"""Review color fallbacks must respect USD and never replace a bound texture."""

from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom

from clothdojo.sim.workstation_review import _apply_authored_appearance


def _scene():
    from newton import ShapeFlags

    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    attribute = mesh.CreateDisplayColorAttr([Gf.Vec3f(.21404114, .0031308, .5)])
    builder = SimpleNamespace(
        shape_label=["/Mesh"], shape_color=[(.9, .8, .7)],
        shape_flags=[int(ShapeFlags.VISIBLE)],
    )
    return stage, mesh, attribute, builder


@pytest.mark.parametrize("color_space", (None, "raw", "lin_rec709_scene"))
def test_display_color_fallback_converts_linear_rgb(color_space):
    stage, _, attribute, builder = _scene()
    if color_space:
        attribute.SetColorSpace(color_space)
    before = stage.GetRootLayer().ExportToString()
    report = _apply_authored_appearance(builder, stage)
    np.testing.assert_allclose(builder.shape_color[0], (.5, .04044994, .73535698), atol=1e-7)
    assert report["authored_display_color_fallbacks"] == ["/Mesh"]
    assert stage.GetRootLayer().ExportToString() == before


def test_display_color_fallback_keeps_explicit_srgb():
    stage, _, attribute, builder = _scene()
    attribute.SetColorSpace("sRGB")
    _apply_authored_appearance(builder, stage)
    np.testing.assert_array_equal(builder.shape_color[0], tuple(attribute.Get()[0]))


@pytest.mark.parametrize("resolved", (
    {"color": (.1, .2, .3), "texture": None},
    {"color": None, "texture": "bound_albedo.png"},
))
def test_display_color_never_overwrites_bound_appearance(monkeypatch, resolved):
    from newton._src.usd import utils

    stage, _, _, builder = _scene()
    monkeypatch.setattr(utils, "resolve_material_properties_for_prim", lambda _: resolved)
    report = _apply_authored_appearance(builder, stage)
    assert builder.shape_color == [(.9, .8, .7)]
    assert not report["authored_display_color_fallbacks"]


def test_proxy_remains_hidden_even_with_authored_color():
    from newton import ShapeFlags

    stage, mesh, _, builder = _scene()
    physics = int(ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES)
    builder.shape_flags[0] |= physics
    mesh.CreatePurposeAttr("proxy")
    report = _apply_authored_appearance(builder, stage)
    assert not builder.shape_flags[0] & int(ShapeFlags.VISIBLE)
    assert builder.shape_flags[0] & physics == physics
    assert report["hidden_authored_proxy_shapes"] == ["/Mesh"]
