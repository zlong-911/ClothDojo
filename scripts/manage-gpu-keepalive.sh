#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/manage-gpu-keepalive.sh {start|stop|status} [GPU ...]

GPU defaults to every GPU visible to nvidia-smi. Configure the worker with:
  CLOTHDOJO_PYTHON                 Python containing torch
  CLOTHDOJO_KEEPALIVE_STATE_DIR   node-local PID/log directory
  CLOTHDOJO_KEEPALIVE_MATRIX_SIZE matrix size (default: 4096)
  CLOTHDOJO_KEEPALIVE_DUTY_CYCLE  duty cycle in (0, 1] (default: 1.0)
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi
action="$1"
shift
case "$action" in
  start|stop|status) ;;
  *) usage; exit 2 ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project="$(dirname -- "$script_dir")"
worker="${script_dir}/gpu-keepalive.py"
python="${CLOTHDOJO_PYTHON:-${project}/.venv/bin/python}"
state_dir="${CLOTHDOJO_KEEPALIVE_STATE_DIR:-/tmp/clothdojo-gpu-keepalive-${UID}}"
matrix_size="${CLOTHDOJO_KEEPALIVE_MATRIX_SIZE:-4096}"
duty_cycle="${CLOTHDOJO_KEEPALIVE_DUTY_CYCLE:-1.0}"

command -v nvidia-smi >/dev/null
if [[ "$action" == "start" && ! -x "$python" ]]; then
  echo "Python is not executable: $python" >&2
  echo "Set CLOTHDOJO_PYTHON to the host's ClothDojo Python." >&2
  exit 1
fi
mkdir -p "$state_dir"
chmod 700 "$state_dir"
# Serialize PID ownership checks and updates across manager invocations.
exec 9>"${state_dir}/manager.lock"
flock -x 9

mapfile -t visible_gpus < <(
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
)
if [[ $# -gt 0 ]]; then
  gpus=("$@")
else
  gpus=("${visible_gpus[@]}")
fi
if [[ ${#gpus[@]} -eq 0 ]]; then
  echo "No visible NVIDIA GPUs" >&2
  exit 1
fi

declare -A selected_gpus=()
for gpu in "${gpus[@]}"; do
  if [[ ! " ${visible_gpus[*]} " =~ " ${gpu} " ]]; then
    echo "GPU is not visible: $gpu" >&2
    exit 1
  fi
  if [[ -n "${selected_gpus[$gpu]:-}" ]]; then
    echo "Duplicate GPU: $gpu" >&2
    exit 2
  fi
  selected_gpus[$gpu]=1
done

pid_file() { printf '%s/gpu%s.pid' "$state_dir" "$1"; }
log_file() { printf '%s/gpu%s.log' "$state_dir" "$1"; }

is_managed_pid() {
  local gpu="$1"
  local pid="$2"
  local -a command_line=()
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  mapfile -d '' -t command_line <"/proc/${pid}/cmdline"
  [[ "${command_line[1]:-}" == "$worker" && \
     "${command_line[2]:-}" == "--host-device" && \
     "${command_line[3]:-}" == "$gpu" ]]
}

read_live_pid() {
  local gpu="$1"
  local file
  local pid
  file="$(pid_file "$gpu")"
  [[ -f "$file" ]] || return 1
  pid="$(<"$file")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    printf '%s' "$pid"
    return 0
  fi
  rm -f "$file"
  return 1
}

gpu_compute_pids() {
  nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d'
}

start_keepalive() {
  local gpu
  local pid
  local other_pids
  local attempt
  local -a launch_gpus=()

  # Preflight every target before launching anything, so a busy card cannot
  # leave a partially protected host behind.
  for gpu in "${gpus[@]}"; do
    if pid="$(read_live_pid "$gpu")"; then
      if ! is_managed_pid "$gpu" "$pid"; then
        echo "Refusing unexpected live PID $pid from $(pid_file "$gpu")" >&2
        exit 1
      fi
      echo "GPU $gpu already protected by PID $pid"
      continue
    fi
    other_pids="$(gpu_compute_pids "$gpu")"
    if [[ -n "$other_pids" ]]; then
      echo "Refusing busy GPU $gpu; compute PIDs: ${other_pids//$'\n'/,}" >&2
      exit 1
    fi
    launch_gpus+=("$gpu")
  done

  for gpu in "${launch_gpus[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$python" "$worker" \
      --host-device "$gpu" \
      --matrix-size "$matrix_size" \
      --duty-cycle "$duty_cycle" \
      >"$(log_file "$gpu")" 2>&1 </dev/null 9>&- &
    pid="$!"
    printf '%s\n' "$pid" >"$(pid_file "$gpu")"
    for attempt in $(seq 1 150); do
      kill -0 "$pid" 2>/dev/null || break
      if grep -Fxq "GPU $gpu keepalive ready" "$(log_file "$gpu")"; then
        break
      fi
      sleep 0.2
    done
    if ! is_managed_pid "$gpu" "$pid" || \
       ! grep -Fxq "GPU $gpu keepalive ready" "$(log_file "$gpu")"; then
      if is_managed_pid "$gpu" "$pid"; then
        kill -TERM "$pid"
      fi
      echo "Keepalive failed to start on GPU $gpu; see $(log_file "$gpu")" >&2
      exit 1
    fi
    echo "Started GPU $gpu keepalive PID $pid"
  done
}

stop_keepalive() {
  local gpu
  local pid
  local attempt
  for gpu in "${gpus[@]}"; do
    if ! pid="$(read_live_pid "$gpu")"; then
      echo "GPU $gpu has no managed keepalive"
      continue
    fi
    if ! is_managed_pid "$gpu" "$pid"; then
      echo "Refusing to stop unexpected PID $pid for GPU $gpu" >&2
      exit 1
    fi
    kill -TERM "$pid"
    for attempt in $(seq 1 60); do
      is_managed_pid "$gpu" "$pid" || break
      sleep 0.5
    done
    if is_managed_pid "$gpu" "$pid"; then
      echo "Keepalive PID $pid did not stop on GPU $gpu" >&2
      exit 1
    fi
    rm -f "$(pid_file "$gpu")"
    echo "Stopped GPU $gpu keepalive PID $pid"
  done
}

show_status() {
  local gpu
  local pid=""
  for gpu in "${gpus[@]}"; do
    if pid="$(read_live_pid "$gpu")" && is_managed_pid "$gpu" "$pid"; then
      echo "GPU $gpu managed_keepalive=running pid=$pid"
    elif [[ -n "$pid" ]]; then
      echo "GPU $gpu managed_keepalive=unexpected pid=$pid"
    else
      echo "GPU $gpu managed_keepalive=stopped"
    fi
    pid=""
  done
  for gpu in "${gpus[@]}"; do
    nvidia-smi -i "$gpu" \
      --query-gpu=index,utilization.gpu,memory.used,power.draw \
      --format=csv,noheader
  done
}

case "$action" in
  start) start_keepalive ;;
  stop) stop_keepalive ;;
  status) show_status ;;
esac
