# Figure acceptance checklist

Every paper figure must pass both gates below before it is presented as a
candidate or inserted into the manuscript. A failed hard requirement rejects
the figure; it is not treated as a matter of taste.

## Gate 1: correctness and production

- **Data fidelity:** every plotted value is generated from or checked against
  the frozen result table. Missing values are never plotted as zero.
- **Comparison semantics:** color, position, shape, and grouping each have one
  stable meaning. The visually strongest encoding represents the paper's
  primary comparison.
- **Final dimensions:** inspect the figure at its actual `\columnwidth` or
  `\textwidth` insertion size, rather than judging an enlarged preview.
- **Typography:** use the manuscript font. Effective text is at least 8 pt at
  insertion size; ordinary labels should be about 8--9 pt.
- **PDF output:** fonts are embedded, marks remain vector, and `pdfcrop` is run.
  The crop must not cut labels or leave unexplained outer margins.
- **Accessibility:** categories remain distinguishable in grayscale and under
  common color-vision deficiencies. Color is not the sole cue when contrast is
  insufficient.

## Gate 2: visual review at final size

- **Three-second test:** a reader can identify the primary comparison and the
  best-performing condition within three seconds.
- **Reading order:** title/group labels, marks, values, axes, and legend form a
  clear sequence without searching back and forth.
- **Spacing:** bar width, within-group gap, between-group gap, panel gap, and
  outer margins form an intentional hierarchy. Empty space must explain
  grouping rather than expose missing content.
- **Collision check:** labels do not overlap, touch bars, sit ambiguously between
  bars, or require ad-hoc staggering that looks accidental.
- **Visual balance:** no legend, axis, separator, color, or incomplete region
  dominates the data. Sparse panels must still look deliberate.
- **Restraint:** remove guides, borders, labels, and decoration that do not help
  answer the comparison question.
- **Consistency:** palette, stroke weight, capitalization, abbreviations, and
  number precision match the other accepted paper figures.

## Review record

For each final figure, record:

1. source data or result files;
2. intended insertion width and effective minimum font size;
3. cropped PDF dimensions and embedded fonts;
4. color and grayscale previews;
5. Gate 1 and Gate 2 result, including any remaining limitation.

