# Historical FR3 Duo review tabletop

`tabletop_context_mm.stl` is the unchanged accepted tabletop mesh with six support
holes, promoted from the FR3 installation review. Coordinates are in millimetres;
the renderer converts them to metres. Bounds match `ShirtFlattenSpec`: centre
`(0.651, 0, 0.775)` m and half extents `(0.631, 0.7, 0.025)` m.

This mesh is visual review context, not a Franka official part and not a robot
body or production collision shape. It preserves the accepted through-table
support appearance; it does not certify a manufactured installation. It belongs
to the external workstation asset package alongside `fr3_duo.usda` and its payload.

The current FR3 installation is back at the accepted table-edge position, with
the 150 mm extrusion and original through-table fasteners. The review still uses
the same simplified, unperforated `ShirtFlattenSpec` table as other robots; this
STL is retained as installation geometry reference and is not loaded by the
renderer. The task table collision shape is unchanged.
