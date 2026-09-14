# Method framework typography specification

The figure is authored on a 504 by 210 bp canvas and processed with pdfcrop
1.42. The crop removes the unused vertical canvas while retaining 3.99 bp side
margins, preserving a 503.998 bp PDF width. The measured height is recorded in
`../figures/clothdojo-method-framework.sources.json`. Insertion at the manuscript's
504 bp text width gives a scale of 1.000004; every value below is therefore the
effective paper size when rounded to 0.001 pt.

| Role | Typeface | Size | Weight | Color |
| --- | --- | ---: | --- | --- |
| Stage number and title | Arial | 8.5 pt | Bold | Stage rail / ink |
| Module heading | Arial | 7.5 pt | Bold | Ink |
| Module body | Arial | 7 pt | Regular | Ink or muted ink |
| Connector annotation | Arial | 7 pt | Regular | Flow ink |
| Explicit math script | Arial Math | 6.3 pt | Regular | Parent label color |

Connector annotations follow the geometry of their shaft:

- Above or below a horizontal shaft: center the annotation on the shaft and
  keep 3.5--4.6 pt between its rendered bounding box and the stroke.
- Beside a vertical shaft: center the annotation within the usable segment and
  keep 6.5--7.5 pt between its rendered bounding box and the stroke.

The state-selection return runs below the first two panels and terminates at
the correction module. No connector passes through a label. The human
supervision cards are centered in their panel; the repeated VR return has been
removed because restoration and VR correction are already one module.

The generator measures rendered glyph bounds after math layout, verifies every
standalone connector annotation against these ranges, and rejects overlaps,
detached endpoints, undersized text, and labels outside their modules.
