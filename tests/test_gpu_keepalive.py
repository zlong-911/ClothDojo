"""Exercise process ownership without loading torch or accessing a real GPU."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def manager(tmp_path: Path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = Path(__file__).parents[1] / "scripts/manage-gpu-keepalive.sh"
    script = scripts / source.name
    shutil.copyfile(source, script)
    worker = scripts / "gpu-keepalive.py"
    worker.write_text(textwrap.dedent("""\
        import os
        import signal
        import sys
        import time
        from pathlib import Path
        signal.signal(signal.SIGTERM, lambda *_: exit(0))
        with Path(os.environ['WORKERS']).open('a') as stream:
            stream.write(f'{os.getpid()}\\n')
        if os.environ.get('FAIL_WORKER'):
            time.sleep(0.4)
            raise RuntimeError('simulated CUDA initialization failure')
        print(f'GPU {sys.argv[2]} keepalive ready', flush=True)
        while True:
            time.sleep(0.01)
        """))
    smi = scripts / "nvidia-smi"
    smi.write_text(f"#!{sys.executable}\n" + textwrap.dedent("""\
        import sys
        import time
        if '--query-gpu=index' in sys.argv:
            print('0\\n1\\n10')
        elif '--query-compute-apps=pid' in sys.argv:
            time.sleep(0.3)
        """))
    smi.chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    workers = tmp_path / "workers"
    env = dict(os.environ, PATH=f"{scripts}:{os.environ['PATH']}",
               CLOTHDOJO_PYTHON=sys.executable,
               CLOTHDOJO_KEEPALIVE_STATE_DIR=str(state), WORKERS=str(workers))

    def run(*args: str):
        return subprocess.run(["bash", str(script), *args], env=env,
                              capture_output=True, text=True, timeout=10)

    try:
        yield script, worker, state, workers, env, run
    finally:
        if workers.exists():
            for pid in workers.read_text().splitlines():
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass


def test_duplicate_targets_do_not_launch_workers(manager):
    _, _, _, workers, _, run = manager
    result = run("start", "0", "0")
    assert result.returncode != 0
    assert "Duplicate GPU" in result.stderr
    assert not workers.exists()


def test_stop_does_not_confuse_gpu_one_with_ten(manager):
    _, worker, state, _, env, run = manager
    with subprocess.Popen([sys.executable, str(worker), "--host-device", "10"], env=env) as other:
        try:
            (state / "gpu1.pid").write_text(str(other.pid))
            result = run("stop", "1")
            assert result.returncode != 0
            assert "Refusing to stop unexpected PID" in result.stderr
            assert other.poll() is None
        finally:
            other.terminate()
            other.wait(timeout=5)


def test_concurrent_starts_share_one_worker_and_release_lock(manager):
    script, _, state, workers, env, run = manager
    command = ["bash", str(script), "start", "0"]
    processes = [subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True) for _ in range(2)]
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stdout + stderr
        assert workers.read_text().splitlines() == [(state / "gpu0.pid").read_text().strip()]
        assert run("status", "0").returncode == 0
        stopped = run("stop", "0")
        assert stopped.returncode == 0, stopped.stderr
        assert not (state / "gpu0.pid").exists()
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_start_reports_initialization_failure(manager):
    _, _, _, _, env, run = manager
    env['FAIL_WORKER'] = '1'
    result = run('start', '0')
    assert result.returncode != 0
    assert 'Keepalive failed to start' in result.stderr
    assert 'Started GPU' not in result.stdout
