# UV Shell Gap Overlay

A Blender add-on for the UV Editor that shows, directly on your UV layout, how many texture pixels separate neighboring UV shells, and how far shells sit from their UDIM tile border. Every measurement is color-graded against the padding you need. Problems are called out where they happen: overlapping shells, shells crossing a tile border, and flipped (mirrored) shells.

**Version:** 1.2.0 · **Blender:** 3.6 LTS to 5.2 LTS · **Authors:** Iurii Kruglov & Claude (Anthropic) · **License:** GPL-3.0-or-later

---

## Features

- **Shell-to-shell gaps in pixels.** Measurement points run along every shell border, holes included. Each point draws a line to the nearest neighboring shell and labels it with the distance in texture pixels.
- **Color gradient.** Red at 0 px, yellow at your *Minimal* gap, green at your *Needed* gap and above.
- **Overlap detection.** Points that fall inside another shell are labeled **Overlap**. Overlapping shell pairs are also found between the points: crossing borders, a shell inside another, stacked or mirrored duplicates.
- **UDIM tile borders.** Shells near the edge of their tile show their distance to it, with separate *Border Minimal / Needed* thresholds. Shells crossing a tile line are flagged.
- **Flipped shells.** Mirrored shells are highlighted, and a single button selects them.
- **Adjustable.** Texture resolution (presets, custom non-square, or the image open in the editor), number of points, point position along the border, font size, opacity, line width, and colors.
- **Fast.** Results are cached and recomputed only when the mesh or a relevant setting changes. The heavy lifting is vectorized NumPy code, so large layouts stay usable.

---

## Installation

The add-on comes in two forms with the same code inside. Install only one of them.

| File | Blender | Installs as |
|---|---|---|
| `uv_shell_gap_overlay-1.2.0.zip` | 4.2 and later | Extension |
| `uv_shell_gap_overlay.py` | 3.6 and later | Legacy add-on |

**Blender 4.2 and later (4.5, 5.x)**
1. Open *Edit → Preferences → Add-ons*.
2. Click the drop-down arrow (⌄) in the top-right corner and choose **Install from Disk…**
3. Select the `.zip` (or the `.py`) and make sure **UV Shell Gap Overlay** is enabled.

You can also drag the `.zip` from your file browser into the Blender window.

**Blender 3.6**
1. Open *Edit → Preferences → Add-ons* and click **Install…**
2. Select `uv_shell_gap_overlay.py`, then tick the checkbox next to **UV: UV Shell Gap Overlay**.

**Updating.** Install the new file the same way.
- The extension is replaced in place and stays enabled.
- For the `.py`, turn the add-on off and on again (or restart Blender) so the new code is loaded.
- To switch from the `.py` to the `.zip`, first remove the old add-on: expand it in the Add-ons list and click **Uninstall**.

### Compatibility

| Blender | Status |
|---|---|
| 5.0, 5.1, 5.2 LTS | Tested with 5.0.1, 5.1.2 and 5.2.2 |
| 4.2 LTS, 4.5 LTS | Tested with 4.2.23 and 4.5.14 |
| 3.6 LTS | Supported with the `.py` file, not covered by the automated tests |

"Tested" means the automated test suite passes in that version. It covers measurements, selection, panels and drawing, plus installing, enabling and removing both the `.zip` and the `.py`.

---

## Quick start

1. Select a mesh, enter **Edit Mode**, and open a **UV Editor**.
2. Press **N** to open the sidebar and switch to the **UV Gaps** tab.
3. Set **Texture** to the resolution you bake or paint at, or choose **Active Image**.
4. Set **Minimal** and **Needed** to your padding targets.

The overlay updates as you edit the UVs.

---

## Reading the overlay

| On screen | Meaning |
|---|---|
| Small square on a shell border, a colored line and a label like `12.3 px` | Gap from that point to the nearest other shell. The square marks the point being measured. |
| Colored line ending in a short tick | Distance from the shell to its tile border. The tick sits on the tile edge. |
| Magenta ✕ with **Overlap** | The point lies inside another shell, or two shell borders cross there. |
| Magenta ✕ with **Crosses tile** | The shell straddles a UDIM tile line. |
| Blue tint, blue outline and **Flipped** | The shell's UVs are mirrored. |
| Whole overlay dimmed | A heavy layout is still being edited. It refreshes once you pause. |

**Colors:** red = 0 px (touching) → yellow = Minimal → green = Needed or more. Tile-border distances use the same colors with the *Border Minimal / Needed* values.

**Label priority:** tile crossings, overlaps and flipped shells come first, then the smallest distances relative to their target. With *Hide Overlapping* on, less critical labels that would cover them are skipped. Lines are always drawn.

---

## Settings

Settings are stored per scene and live in the **UV Gaps** tab of the UV Editor sidebar. The checkbox in the panel header turns the whole overlay on or off. The overlay is also hidden when the UV Editor's own *Overlays* are turned off.

### Main panel

| Setting | Default | Description |
|---|---|---|
| Texture | 2048 | Texture resolution used to convert UV distances to pixels. Options: 256–8192, **Custom** (separate Width × Height; non-square textures are supported), or **Active Image** (the size of the image shown in this UV Editor, falling back to Width × Height when there is none). |
| Points per Shell | 24 | Measurement points spread evenly along each shell's whole border, outer border and holes included. |
| Shift Along Border | 0 % | Slides every point along the border. 0–100 % covers one full step between neighboring points, so sweeping the slider checks the entire border. |
| Minimal | 4 px | Smallest acceptable gap between shells (yellow). |
| Needed | 8 px | Target gap between shells (green). |
| Search Radius | 32 px | Only gaps up to this size are measured; shells farther apart are not treated as neighbors. Also decides which shells count as "near" a tile border. |
| Selected Shells Only | off | Measure only from shells with at least one selected UV, as shown in the UV Editor (with UV Sync Selection on in 3.6–4.5: shells with a selected vertex). Distances still go to every shell. |
| Flipped Shells: Show | on | Highlight mirrored shells. |
| Flipped Shells: Select | — | Select all mirrored shells. See [Flipped shells](#flipped-shells). |
| Refresh | — | Force a recompute. Normally not needed. |

### UDIM Tile Borders (subpanel)

| Setting | Default | Description |
|---|---|---|
| Header checkbox | on | Measure shells near the border of their own tile, flag shells crossing a tile line, and measure gaps only between shells in the same tile. |
| Border Minimal | 2 px | Smallest acceptable distance from a shell to its tile border. |
| Border Needed | 4 px | Target distance from a shell to its tile border. |

### Display (subpanel)

| Setting | Default | Description |
|---|---|---|
| Font Size | 12 px | Label size, scaled with Blender's UI scale. |
| Opacity | 0.9 | Opacity of lines, markers and labels. |
| Line Width | 2 px | Width of the measurement lines. |
| Labels: Background | on | Dark box behind each label. When off, text is drawn with a shadow instead. |
| Labels: Hide Overlapping | on | Skip labels that would cover a more important one. |

### Colors (subpanel, collapsed by default)

*Touching (0 px)*, *At Minimal*, *At Needed*, *Overlap* and *Flipped* are all editable.

### Summary box

Shows the number of shells and measured gaps, overlapping shell pairs, gaps and tile-border distances below Minimal / Needed, shells crossing a tile border, and flipped shells.

---

## How it measures

**Which UVs.** Exactly the faces the UV Editor shows. Hidden faces are ignored, and with UV Sync Selection off only faces selected in the mesh are included. All meshes in Edit Mode are measured together, each using its active UV map.

**Shells.** Faces that share a UV vertex (the same mesh vertex with the same UV) belong to one shell. A shell's border is made of the UV edges used by only one face, ordered into continuous loops, holes included.

**Points and gaps.** Points are spaced evenly by length along each shell's whole border. From each point, the add-on finds the closest point on another shell's border, measured in texture pixels, so non-square textures are handled correctly. Only neighbors in front of the border count, within about 75° of the direction the border faces. A measurement therefore never passes through the shell's own body or runs sideways along its edge. Neighbors beyond the Search Radius are ignored. Touching shells read **0.0 px**.

**Overlaps.** A point is an overlap when it lies inside another shell. The inside test uses the even-odd rule, so holes count as empty space. Independently, every pair of shells is checked for crossing borders and for one shell lying inside or stacked on another. That way an overlap is reported even when no point happens to land on it. Shells that only touch are not overlaps.

**Tile borders.** Tile borders are the integer UV lines; the 0–1 square is tile 1001. Each shell belongs to the tile that contains the center of its bounding box. A point measures to an edge of its own tile only when:
- its border faces that edge,
- the edge is within the Search Radius, and
- no shell lies in between. This includes other shells and the shell's own body, such as the far side of a ring.

Shells whose bounds span a tile line are reported as **Crosses tile** instead of being measured.

<a id="flipped-shells"></a>**Flipped shells.** A shell is flipped when its total signed UV area is negative. Its faces run clockwise in UV space, which means the texture appears mirrored on the model.
- **Select** replaces the current selection. Enable *Extend* in the redo panel to add to it instead. The operation can be undone.
- With UV Sync Selection on, it selects mesh faces.
- With UV Sync Selection off, it selects UVs and leaves the mesh selection alone.

---

## Choosing padding values

Bake margin (dilation) grows outward from every shell. Two neighboring shells each need room for their own margin, so the gap between them should be about **twice** the margin. A shell needs only **one** margin to the tile border. That is why the border defaults (2 / 4 px) are half the shell defaults (4 / 8 px).

If you rely on mipmaps, each mip level halves the distance in pixels, so textures seen from far away need more padding.

---

## Performance

Results are cached at three levels:
- Geometry is re-read only when the mesh changes.
- Measurements are redone only when a relevant setting changes.
- Redraws while panning or zooming reuse both.

Measured headless in Blender 4.2 to 5.2 with the default settings. These are CPU-side timings; GPU drawing time is not included.

| Layout | Re-read after an edit | Re-measure after a setting change | Redraw (pan / zoom) |
|---|---|---|---|
| 144 shells, 1.3k faces | ~10–15 ms | ~7–9 ms | ~2–3 ms |
| 1,600 shells, 14.4k faces | ~0.13–0.17 s | ~0.10–0.12 s | ~14–15 ms |

When a step takes longer than about 40 ms, the add-on waits until you pause for ~0.25 s and dims the overlay meanwhile. Editing and dragging sliders stay responsive.

---

## Limitations

- Gaps are sampled at the measurement points. A narrow spot between two points is found by adding points or sweeping *Shift Along Border*. Overlaps are always found.
- Each point reports only its nearest neighbor.
- UVs meant to wrap around (tiling textures) are not treated as wrapping. With Tile Borders on, every tile is a separate texture.
- Flipped detection works per shell, based on its overall orientation. Individual folded faces inside an otherwise normal shell are not flagged.
- Only the active UV map of each mesh is measured.

---

## Troubleshooting

- **Nothing is drawn.** Check that:
  - you are in Edit Mode;
  - the editor is a UV Editor, not the Image Editor in View mode;
  - the panel's header checkbox is on;
  - the UV Editor's *Overlays* are enabled.

  With UV Sync Selection off, only faces selected in the mesh are shown, and only those are measured.
- **Numbers look too small or too large.** Check *Texture*: distances scale with the resolution.
- **The add-on appears twice in the Add-ons list.** Both the `.py` and the `.zip` are installed. Uninstall one of them and restart Blender.
- **Something went wrong.** The add-on prints each error once to the system console. On Windows use *Window → Toggle System Console*; on other systems start Blender from a terminal.

---

## Changelog

**1.2.0**
- Blender 5 support (5.0, 5.1, 5.2 LTS). *Selected Shells Only* and *Select Flipped* work with Blender 5's new UV selection, including UV Sync Selection.
- Also packaged as an extension (`.zip`) for Blender 4.2 and later.
- Gap lines go only to neighbors in front of the border (within about 75° of the direction it faces). Before, a point near a corner could measure sideways along its own shell's edge.
- Label backgrounds are more opaque, so lines no longer show through the text.

**1.1.0**
- Distance from shells to their UDIM tile border, for shells near the border, with *Border Minimal / Needed*.
- Shells crossing a tile line are flagged. While Tile Borders is on, gaps are measured only between shells in the same tile.
- Flipped shells: show (tint, outline, label) and select.

**1.0.0**
- Pixel gaps between neighboring shells along their borders, with a color gradient.
- Overlap detection.
- Sidebar settings and label decluttering.

---

## Authors

Iurii Kruglov & Claude (Anthropic)

## License

GPL-3.0-or-later
