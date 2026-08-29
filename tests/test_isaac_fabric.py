from __future__ import annotations

import numpy as np

from clothdojo.teleop.isaac_fabric import _world_to_local_transforms


def test_world_to_local_transforms_respects_installed_root_frame() -> None:
    sine = np.sqrt(0.5)
    installation = np.asarray((1.0, 2.0, 3.0, 0.0, 0.0, sine, sine))
    body_q = np.asarray(
        (
            installation,
            (1.0, 2.5, 3.0, 0.0, 0.0, sine, sine),
        )
    )
    parent_world = np.asarray(
        (
            installation,
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
    )

    local = _world_to_local_transforms(
        body_q,
        np.asarray((-1, 0)),
        parent_world,
    )

    np.testing.assert_allclose(local[0], (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    np.testing.assert_allclose(
        local[1], (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), atol=1.0e-7
    )
