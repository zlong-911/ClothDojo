from __future__ import annotations

import functools
from types import SimpleNamespace

from clothdojo.sim import runtime as runtime_module
from clothdojo.sim.runtime import ShirtFlattenRuntime


class _CachedView:
    @functools.lru_cache(maxsize=None)  # noqa: B019, UP033 - matches Newton 1.5
    def _get_attribute_array(self, name):
        return name

    @functools.cache  # noqa: B019 - matches Newton 1.5
    def _get_actuator_dof_mapping(self, name):
        return name


def test_close_releases_owned_references_and_is_idempotent(monkeypatch) -> None:
    view = _CachedView()
    view._get_attribute_array("joint_q")
    view._get_actuator_dof_mapping("joint_q")
    scene = SimpleNamespace(
        robot=view,
        arms=view,
        grippers=view,
        end_effectors=view,
    )
    runtime = ShirtFlattenRuntime.__new__(ShirtFlattenRuntime)
    runtime.scene = scene
    runtime.model = SimpleNamespace(device="cuda:7")
    runtime.owned_buffer = object()

    synchronized = []
    collected = []
    monkeypatch.setattr(runtime_module.wp, "synchronize_device", synchronized.append)
    monkeypatch.setattr(runtime_module.gc, "collect", lambda: collected.append(True))

    runtime.close()

    assert runtime.closed
    assert runtime.__dict__ == {"_closed": True}
    assert synchronized == ["cuda:7", "cuda:7"]
    assert collected == [True]
    assert _CachedView._get_attribute_array.cache_info().currsize == 0
    assert _CachedView._get_actuator_dof_mapping.cache_info().currsize == 0

    runtime.close()
    assert synchronized == ["cuda:7", "cuda:7"]
    assert collected == [True]
