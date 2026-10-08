# UV Shell Gap Overlay

A Blender add-on that shows, directly on your UV layout, how many texture pixels separate neighboring UV shells, how far shells sit from their UDIM tile border, and the texel density of every shell. Gap measurements are color-graded against the padding you need. Texel density is color-graded against the density you need, in the UV Editor and on the mesh in the 3D Viewport. Each shell gets one compact info block with its density, its object's scale, whether it is flipped, and an arrow showing which way is up in the scene. Material sets let you select, hide and reveal the shells of chosen materials and measure gaps only within a material. Problems are called out where they happen: overlapping shells, shells crossing a tile border, and flipped (mirrored) shells. Shells stacked on each other on purpose, such as copies of a part sharing texture space, are told apart from real overlaps: a stack counts as one shell, and its copies can be selected or moved to another UDIM tile with one click.

**Version:** 1.5.0 · **Blender:** 3.6 LTS to 5.2 LTS · **License:** GPL-3.0-or-later

---

## Features

- **Shell-to-shell gaps in pixels.** Measurement points run along every shell border, holes included. Each point draws a line to the nearest neighboring shell and labels it with the distance in texture pixels.
- **Color gradient.** Red at 0 px, yellow at your *Minimal* gap, green at your *Needed* gap and above.
- **Overlap detection.** Points that fall inside another shell are labeled **Overlap**. Overlapping shell pairs are also found between the points: crossing borders, a shell inside another, copies that don't sit exactly on each other.
- **Stacks.** Shells that cover each other almost completely (99.9 % by default) are a stack: copies of a part, mirrored halves, anything laid on the same texture space on purpose. A stack is measured as one shell and is not reported as an overlap. Its info block says how many shells it holds. See [Overlaps and stacks](#overlaps-and-stacks).
- **Select and move overlaps.** Buttons select the overlapping shells, the stacked shells, or all but one shell of every stack and overlap. *Move All but One* sends those to another UDIM tile and leaves one shell in place, an unflipped one where there is one.
- **Same material only.** Gaps and overlaps can be limited to shells of the same material, since each material usually has its own texture.
- **UDIM tile borders.** Shells near the edge of their tile show their distance to it, with separate *Border Minimal / Needed* thresholds. Shells crossing a tile line are flagged.
- **Flipped shells.** Mirrored shells are highlighted, and a single button selects them.
- **Texel density per shell.** Every shell is tinted by its texel density: red at your *Low* value, green at *Needed*, blue at *High*, with a gradient in between. Values are shown in px/cm, px/m, px/in or px/ft. See [Texel density](#texel-density).
- **Auto Low / High.** *Low* and *High* can fill themselves with the lowest and highest density of the material sets you check, and stay editable.
- **Select by texel density.** A From / To range over the analyzed densities selects every shell inside it. The shells in range are outlined.
- **Texel density in the 3D Viewport.** The same colors on the mesh itself. In Edit Mode they cover the objects being edited; in Object Mode, the selected objects, modifiers included.
- **One info block per shell.** Texel density, object scale (when it is not 1), *Flipped*, and an optional orientation arrow share one box per shell, so they never cover each other.
- **Orientation arrows.** An arrow on each shell points where the scene's up direction (+Z) runs across it. Shells lying flat get a +Y arrow instead.
- **Material sets.** A list of the materials on the shown objects, with shell counts and density ranges. Select, hide or reveal the shells of one material or of several checked ones.
- **Adjustable.** Texture resolution (presets, custom non-square, or the image open in the editor), number of points, point position along the border, font size, opacity, line width, and colors.
- **Fast.** Meshes are read as whole arrays and processed with vectorized NumPy code. Only the objects that changed are read again, and only their material is measured again. Texel density colors are computed on the GPU, so changing a threshold is instant. See [Performance](#performance).

---

## Installation

The add-on comes in two forms with the same code inside. Install only one of them.

| File | Blender | Installs as |
|---|---|---|
| `uv_shell_gap_overlay-1.5.0.zip` | 4.2 and later | Extension |
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
| 3.6 LTS | Supported with the `.py` file. Not covered by the tests inside Blender: see below |

"Tested" means the automated test suite passes in that version, run with the NumPy that Blender release ships (1.24 for 4.2, 1.26 for 4.5 and 5.0, 2.3 for 5.1 and 5.2). It covers gap measurements, stacks and overlaps, the select and move buttons, texel density, material sets, selection, hiding and revealing, info blocks, panels and drawing (UV Editor and 3D Viewport), plus installing, enabling and removing both the `.zip` and the `.py`. Two tests check what the add-on keeps between redraws: one makes 200 random edits, the other presses the buttons in random order between edits (runs of 120 to 300 steps), and after every step what is shown must equal a computation from nothing. *Select Flipped*, which now shares its code with the other Select buttons, is checked to select the same faces as in 1.4.0. The drawing was also checked in real renders (Blender 5.2).

There is no Blender 3.6 build the tests can run in. For 3.6, the measuring code (shells, stacks, overlaps, gaps, labels) is run on its own under Python 3.10 with NumPy 1.23.5, the versions Blender 3.6 ships, and gives the same results as in 4.5 and 5.2. The parts that talk to Blender itself are those 1.4 already used in 3.6, plus three calls that the 3.6 documentation lists (a mesh's count of selected faces and vertices, a button's own tooltip, and the view layer's update). Undo of the new buttons relies on Blender's standard operator undo and could not be run in the headless tests.

---

## Quick start

1. Select a mesh, enter **Edit Mode**, and open a **UV Editor**.
2. Press **N** to open the sidebar and switch to the **UV Gaps** tab.
3. Set **Texture** to the resolution you bake or paint at, or choose **Active Image**.
4. Set **Minimal** and **Needed** to your padding targets.
5. For texel density, tick the header checkbox of the **Texel Density** panel. Then pick a unit and set **Needed**. *Low* and *High* fill themselves from your shells while *Auto Low / High* is on.
6. To see the colors on the mesh, tick **Show in 3D Viewport**. The same settings are also in the 3D Viewport's sidebar, **UV Gaps** tab.
7. To work per material, use the **Materials** panel: check sets, then **Select**, **Hide** or **Reveal**.
8. To deal with shells lying on each other, open **Overlaps and Stacks**: select them, or press **Move All but One** to send every copy to the next UDIM tile.
9. For orientation arrows, open **Shell Info** and tick **Orientation Arrows**.

The overlays update as you edit.

---

## Texel density

Texel density is how many texture pixels cover one meter of the model's surface. When it is even across a model, and across the models in a scene, textures look equally sharp everywhere. A shell with too little density looks blurry next to its neighbors; one with too much spends texture space that other shells could use.

The add-on finds the density of every UV shell and shows it in two places: on the UV layout, where you fix it, and on the model in the 3D Viewport, where you see the result. Both use the same settings (*Texture*, *Unit*, *Low*, *Needed* and *High*) and the same colors.

**How the number is found.** The texture pixels a shell covers are divided by its surface area in square meters, and the square root is taken. A 1 × 1 m face whose UVs span a quarter of the width and height of a 2048 px texture covers 512 × 512 pixels, so its density is 512 px/m. Object scale counts, and so does the scene's *Unit Scale*. The details are under [How it measures](#how-it-measures).

**The colors.** Each shell is tinted by how its density compares with your values:

| Color | Density |
|---|---|
| Red | at or below *Low* |
| Yellow | between *Low* and *Needed* |
| Green | at *Needed*, your target |
| Cyan | between *Needed* and *High* |
| Blue | at or above *High* |
| Gray | the shell has no 3D area |

Red and yellow shells get fewer pixels than your target and will look softer; cyan and blue ones use more texture space than they need. *Fill Opacity* sets how strongly the colors cover the UV layout and the shaded model.

### In the UV Editor

Tick the header checkbox of the **Texel Density** panel (UV Editor sidebar, **UV Gaps** tab).
- Every shell the UV Editor shows is tinted, and its info block gives its density in px/cm, px/m, px/in or px/ft.
- The values follow your edits: scale a shell and its color and number change with it.
- The summary box lists the number of shells, their median, lowest and highest density, and how many are at or below Low and at or above High.
- With *Auto Low / High* on, red marks the least dense shells of the checked material sets and blue the densest, so the spread shows at a glance. Type your own Low and High to grade against fixed limits instead.
- **Select by Range** outlines the shells inside a From / To range of densities and selects them with one click. For example, lower *To* until the least dense shells are outlined, select them, and scale them up until they turn green.
- The **Materials** list shows each material's density range, so a texture set that is out of line stands out.

### In the 3D Viewport

Tick **Show in 3D Viewport** in the same panel, or the header checkbox of the **Texel Density** panel in the 3D Viewport sidebar (**UV Gaps** tab). That panel has the same settings and its own summary box.
- Every face takes the color of its UV shell, drawn over the shaded model, so you can see on the model itself where texture resolution falls short or is wasted. Surfaces behind others are not colored through.
- **Edit Mode:** the objects being edited are colored, and the colors follow your UV edits. Hidden faces are left out. The colors sit on the edit cage, not on the modifier result.
- **Object Mode:** the selected mesh objects are colored as they are drawn, modifiers included. This is the way to compare several props, or a whole scene, against one target density.
- Each object uses its active UV map. With *Texture* set to *Active Image*, the size of the image open in a UV Editor is used.
- The 3D Viewport shows only the colors; the numbers are in the UV Editor and in the panel's summary box.
- The colors are part of the viewport's overlays, so its *Overlays* must be on.
- Changing Low, Needed, High, the unit or the texture size recolors at once. With many objects selected, the colors appear over a few frames instead of freezing the viewport.

---

## Reading the overlay

| On screen | Meaning |
|---|---|
| Small square on a shell border, a colored line and a label like `12.3 px` | Gap from that point to the nearest other shell. The square marks the point being measured. |
| Colored line ending in a short tick | Distance from the shell to its tile border. The tick sits on the tile edge. |
| Magenta ✕ with **Overlap** | The point lies inside another shell, or two shell borders cross there. Shells of one stack don't mark each other. |
| Magenta ✕ with **Crosses tile** | The shell straddles a UDIM tile line. |
| Blue tint and blue outline | The shell's UVs are mirrored; its info block says **Flipped**. A mirrored copy lying in a stack with an unflipped shell is not tinted. |
| Shell tinted red to blue | The shell's texel density (see below). Gray: the shell has no 3D area. |
| Info block on a shell | One box per shell: its density (`512 px/m`), `Scale 1.5` in orange when its object's scale is not 1, **Flipped**, and an arrow on the left. |
| `× 4` in an info block | A stack of four shells; the block stands for all of them. `× 4 (1 flipped)` when some, but not all, are mirrored. |
| `256 – 512 px/m` in a stack's block | The shells of the stack differ in texel density: the same UVs are used by parts of different size. |
| `Scale varies` in a stack's block | The stack's shells belong to objects with different scale. |
| Blue arrow in a block | Where the scene's up (+Z) runs across the shell. An arrow pointing up means the texture stands upright on the model. |
| Green arrow in a block | The shell lies flat (a floor or a roof top), so the arrow shows where +Y runs instead. |
| White outline around shells | The shells inside the *Select by Range* From / To range. |
| Colored faces in the 3D Viewport | The texel density of each face's UV shell, in the same colors. |
| Whole overlay dimmed | A heavy layout is still being edited. It refreshes once you pause. |

**Gap colors:** red = 0 px (touching) → yellow = Minimal → green = Needed or more. Tile-border distances use the same colors with the *Border Minimal / Needed* values.

**Texel density colors:** red at Low and below → yellow → green at Needed → cyan → blue at High and above. With Low 100, Needed 300 and High 500 px/m, a shell at 200 px/m is yellow and one at 400 px/m is cyan.

**Label priority:** tile crossings and overlaps come first, then the info blocks (those with a scale note, a density span or *Flipped* first), then the smallest distances relative to their target. With *Hide Overlapping* on, less important labels that would cover them are skipped. Lines and colors are always drawn.

---

## Settings

Settings are stored per scene. They live in the **UV Gaps** tab of the UV Editor sidebar; the texel density settings and the material list are also in the **UV Gaps** tab of the 3D Viewport sidebar. The checkbox in a panel's header turns that overlay on or off, so gaps, texel density and info blocks can be used separately or together. Overlays are also hidden when the editor's own *Overlays* are turned off.

### UV Shell Gaps panel

| Setting | Default | Description |
|---|---|---|
| Header checkbox | on | Show the gap overlay. |
| Texture | 2048 | Texture resolution used to convert UV distances to pixels. Options: 256–8192, **Custom** (separate Width × Height; non-square textures are supported), or **Active Image** (the size of the image shown in this UV Editor, falling back to Width × Height when there is none). |
| Points per Shell | 24 | Measurement points spread evenly along each shell's whole border, outer border and holes included. |
| Shift Along Border | 0 % | Slides every point along the border. 0–100 % covers one full step between neighboring points, so sweeping the slider checks the entire border. |
| Minimal | 4 px | Smallest acceptable gap between shells (yellow). |
| Needed | 8 px | Target gap between shells (green). |
| Search Radius | 32 px | Only gaps up to this size are measured; shells farther apart are not treated as neighbors. Also decides which shells count as "near" a tile border. |
| Selected Shells Only | off | Measure only from shells with at least one selected UV, as shown in the UV Editor (with UV Sync Selection on in 3.6–4.5: shells with a selected vertex). A stack counts as selected when one of its shells is. Distances still go to every shell. With UV Sync Selection off, Blender doesn't announce UV selection changes, so the add-on looks for them itself when the UV Editor redraws without panning or zooming, at most five times a second. |
| Same Material Only | on | Measure gaps and find overlaps only between shells of the same material. Shells of other materials are ignored, including as obstacles between a shell and its tile border. |
| Flipped Shells: Show | on | Highlight mirrored shells. |
| Flipped Shells: Select | — | Select all mirrored shells. See [Flipped shells](#flipped-shells). |
| Refresh | — | Force a recompute. Normally not needed. |

The summary box shows the number of shells and measured gaps, the stacks, overlapping shell pairs, gaps and tile-border distances below Minimal / Needed, shells crossing a tile border, and flipped shells.
- `Flipped shells: 12 (9 in stacks)` means nine of them are mirrored copies lying in a stack with an unflipped shell. The warning icon shows only when a flipped shell is not in such a stack.
- Under `Overlapping pairs`, a line like `0 of them in the 0-1 tile` appears when shells overlap in other tiles too, for example the copies moved there by *Move All but One*. It tells whether the 0–1 tile itself is clean.

### Overlaps and Stacks (subpanel)

| Setting | Default | Description |
|---|---|---|
| Stacked Shells as One | on | Shells covering each other at least as much as *Stack Match* are a stack. A stack is measured once, as one shell, and is not reported as an overlap. Off: every shell is measured on its own and every copy is an overlap, as before 1.5. |
| Stack Match | 99.9 % | How much two shells must cover each other to be a stack. Lower it when your copies are snapped less exactly; 100 % accepts only exact copies. |
| Select: Overlapping | — | Selects the shells that overlap and are not a stack: copies that don't fit, shells lying partly on others. When a stack is involved, all its shells are selected. |
| Select: Stacked | — | Selects every shell that lies in a stack. |
| Select: All but One | — | Selects all the shells lying on others except one of every stack and overlap. See below for which one stays. |
| Tiles | 1 | How many UDIM tiles along U *Move All but One* moves the shells. Negative values move to the left. |
| Move All but One | — | Moves the shells *All but One* would select by *Tiles* tiles and selects them. |
| Only in the 0-1 Tile | on | *All but One* (Select and Move) takes only shells lying in the 0–1 tile (UDIM 1001). Copies moved away earlier stay where they are, so pressing *Move All but One* twice changes nothing the second time. Off: the shells of every tile. |

Hold **Shift** while clicking a Select button to add to the selection. All of these can be undone, and their options can be changed afterwards in the redo panel.

The box at the bottom counts the stacks and the shells lying on another one. When some of those lie outside the 0–1 tile, it also says how many are left in it: `0 of them in the 0-1 tile` after *Move All but One*. With a hundred or more overlapping pairs the box reminds you that copies placed less exactly need a lower *Stack Match*.

### UDIM Tile Borders (subpanel)

| Setting | Default | Description |
|---|---|---|
| Header checkbox | on | Measure shells near the border of their own tile, flag shells crossing a tile line, and measure gaps only between shells in the same tile. |
| Border Minimal | 2 px | Smallest acceptable distance from a shell to its tile border. |
| Border Needed | 4 px | Target distance from a shell to its tile border. |

### Display (subpanel)

Applies to every label, info blocks included.

| Setting | Default | Description |
|---|---|---|
| Font Size | 12 px | Label size, scaled with Blender's UI scale. |
| Opacity | 0.9 | Opacity of lines, markers, labels and arrows. |
| Line Width | 2 px | Width of the measurement lines. |
| Labels: Background | on | Dark box behind each label and info block. When off, text is drawn with a shadow instead. |
| Labels: Hide Overlapping | on | Skip labels that would cover a more important one. |

### Colors (subpanel, collapsed by default)

*Touching (0 px)*, *At Minimal*, *At Needed*, *Overlap* and *Flipped* are all editable.

### Materials panel

A list of the materials used by the shown objects: in Edit Mode the objects being edited, in Object Mode the objects colored in the 3D Viewport. Faces without a material, or on an empty material slot, form the **(No Material)** set.

Each row shows:
- a checkbox,
- the material,
- the number of shells shown in the UV Editor and their texel density range, in the current unit,
- a **select** button (arrow icon) and a **hide / reveal** toggle (eye icon) for that material alone.

| Button | What it does |
|---|---|
| Select | Selects the shells of the checked sets, or of the highlighted row when nothing is checked. Hold **Shift** while clicking (or turn on *Extend* in the redo panel) to add to the selection. |
| Hide | Hides the shells of the checked sets in the UV Editor. |
| Reveal | Shows them again and selects them, as Blender's *Reveal* does. |
| All / None / Invert | Changes which sets are checked. |
| Same Material Only | The same setting as in the gaps panel. |

The checked sets also decide what *Auto Low / High* fills from. How hiding works with and without UV Sync Selection is explained under [Material sets](#material-sets).

### Texel Density panel

| Setting | Default | Description |
|---|---|---|
| Header checkbox | off | Tint every shell by its texel density in the UV Editor and add the density to its info block. |
| Texture | 2048 | The same setting as in the gaps panel. Texel density depends on it. |
| Unit | px/m | px/cm, px/m, px/in or px/ft. Switching the unit converts the values below; the densities they stand for don't change. |
| Low (Red) | 100 px/m | Shells at or below this density are red. |
| Needed (Green) | 300 px/m | The density you aim for. Shells at this density are green. |
| High (Blue) | 500 px/m | Shells at or above this density are blue. |
| Auto Low / High | on | Fills Low and High with the lowest and highest density of the checked material sets, or of all shells when none is checked. The label next to the checkbox says which. See [Auto Low / High](#auto-low--high). |
| Fill button (↻) | — | Fills Low and High once, whether Auto is on or not. |
| Fill Opacity | 0.35 | Opacity of the density colors. |
| Show in 3D Viewport | off | Also color the mesh in 3D Viewports. |

The summary box shows the number of shells, the median, lowest and highest density, and how many shells are at or below Low and at or above High.

### Select by Range (subpanel of Texel Density)

| Setting | Default | Description |
|---|---|---|
| From / To | 0 % / 100 % | The range, as a share of the way from the lowest (0 %) to the highest (100 %) density among the shown shells. Below the sliders are the densities they stand for, and how many shells are in range. The two handles can't pass each other. |
| Highlight Range | on | Outline the shells in range in white while the range is narrower than all shells. |
| Select Shells in Range | — | Selects the shown shells whose density is in range. Hold **Shift** while clicking to add to the selection. |

Blender's sidebar has no slider with two handles, so the range is two linked sliders side by side: *From* on the left, *To* on the right. The lowest and highest densities they refer to are found automatically whenever the shells change.

### Shell Info panel

| Setting | Default | Description |
|---|---|---|
| Header checkbox | on | Show the extra lines and the arrow in the info blocks. |
| Orientation Arrows | off | Add an arrow to each shell's block: blue for the scene's up (+Z), green for +Y on shells lying flat. Arrows also show on their own, with the other overlays off. |
| Object Scale Not 1 | on | Add `Scale …` to the blocks of shells whose object's scale is not 1, for example `Scale 1.5` or `Scale 1 × 2 × 1`. The note joins the blocks while the gap or texel density overlay is on. |

A stack gets one block, on the shell that stands for it, with a `× N` line while the header checkbox is on. *Stacked Shells as One* in the Overlaps and Stacks subpanel decides whether stacks are formed at all.

The texel density line comes from the Texel Density panel and *Flipped* from the gaps panel. Their own header checkboxes decide whether they appear. With the gap overlay, texel density and arrows all off, the add-on does no work in the UV Editor.

### Texel Density and Materials in the 3D Viewport sidebar

The **UV Gaps** tab of the 3D Viewport sidebar has the same texel density settings. Its header checkbox turns the 3D coloring on or off, and *Show in UV Editor* turns on the UV Editor overlay. In Edit Mode it colors the objects being edited. In Object Mode it colors the selected mesh objects. A **Materials** panel (collapsed by default) holds the same material list.

---

## How it measures

**Which UVs.** Exactly the faces the UV Editor shows. Hidden faces are ignored, and with UV Sync Selection off only faces selected in the mesh are included. All meshes in Edit Mode are measured together, each using its active UV map.

**Shells.** Faces that share a UV vertex (the same mesh vertex with the same UV) belong to one shell. A shell's border is made of the UV edges used by only one face, ordered into continuous loops, holes included.

**Points and gaps.** Points are spaced evenly by length along each shell's whole border. From each point, the add-on finds the closest point on another shell's border, measured in texture pixels, so non-square textures are handled correctly. Only neighbors in front of the border count, within about 75° of the direction the border faces. A measurement therefore never passes through the shell's own body or runs sideways along its edge. Neighbors beyond the Search Radius are ignored. Touching shells read **0.0 px**.

**Overlaps.** A point is an overlap when it lies inside another shell. The inside test uses the even-odd rule, so holes count as empty space. Independently, every pair of shells is checked for crossing borders and for one shell lying inside or on another. That way an overlap is reported even when no point happens to land on it. Shells that only touch are not overlaps.

<a id="overlaps-and-stacks"></a>**Stacks.** Reusing texture space is normal: both wheels of an axle, the left and right half of a body, forty identical bolts. Their shells lie exactly on each other and are not a mistake. With *Stacked Shells as One* on, such shells form a stack:
- Two shells belong to one stack when they cover each other by at least *Stack Match*. For copies with the same border points this is checked point by point. For others, the area between the two borders is estimated from the distance between them along the border, and compared with the shells' area.
- A mirrored copy stacks with its unflipped original. Stacks follow *Same Material Only*: with it on, shells of different materials never form a stack.
- One shell stands for the stack: the first one that isn't flipped, or the first one when all are flipped. Gaps are measured from it once. The other shells of the stack are skipped, which is also what makes layouts full of copies fast.
- What is left as an overlap is what needs attention: a copy that sits 97 % on its original, a shell lying partly on another, a shell inside another.
- The shells of a stack can differ in the scene: copies of different size have different texel density. The stack's block then shows the span of densities.

**All but One.** *Select All but One* and *Move All but One* leave one shell of every pile in place, so that afterwards nothing lies on anything else:
- Of a stack, the shell standing for it stays and its copies go.
- Among shells that overlap without being a stack, shells are kept in this order: unflipped before flipped, then a stack before a single shell, then the larger one, then the first. A shell goes when it overlaps one that was already kept. A stack goes or stays as a whole.
- With *Only in the 0-1 Tile*, only shells of that tile go.
- *Move All but One* moves whole tiles along U, so the moved shells keep their place inside the tile and their texel density. It moves the shells as the UV Editor shows them: faces hidden there are not moved.
- The buttons look at all shown shells, not only the selected ones, and follow *Same Material Only* and *Stack Match*.

**Tile borders.** Tile borders are the integer UV lines; the 0–1 square is tile 1001. Each shell belongs to the tile that contains the center of its bounding box. A point measures to an edge of its own tile only when:
- its border faces that edge,
- the edge is within the Search Radius, and
- no shell lies in between. This includes other shells and the shell's own body, such as the far side of a ring.

Shells whose bounds span a tile line are reported as **Crosses tile** instead of being measured.

<a id="flipped-shells"></a>**Flipped shells.** A shell is flipped when its total signed UV area is negative. Its faces run clockwise in UV space, which means the texture appears mirrored on the model.
- **Select** replaces the current selection. Enable *Extend* in the redo panel to add to it instead. The operation can be undone.
- With UV Sync Selection on, it selects mesh faces.
- With UV Sync Selection off, it selects UVs and leaves the mesh selection alone.

**Texel density.** For each shell, texel density = √(texture pixels the shell covers ÷ its surface area in square meters).
- Pixels covered = UV area × Width × Height. For a non-square texture the result is the geometric mean of the horizontal and vertical density.
- The surface area is taken in world space, so object scale is included, non-uniform scale too.
- One Blender unit is one meter. When the scene's unit system is Metric or Imperial, its *Unit Scale* is used instead (a Unit Scale of 0.01 means one unit is 1 cm).
- The info block sits on the face nearest the shell's center of area, so it stays on the shell even for rings and L-shapes.

Which faces are colored:
- **UV Editor:** the faces the UV Editor shows, as for gaps.
- **3D Viewport in Edit Mode:** every visible face of the edited objects (the edit cage).
- **3D Viewport in Object Mode:** the selected objects as they are drawn, modifiers included. With a Mirror modifier, for example, both halves are colored.

<a id="material-sets"></a>**Material sets.** A shell belongs to the material that covers most of its UV area. That matters only for the rare shell whose faces use several materials.
- *Same Material Only* compares each shell only with shells of its own material. With *UDIM Tile Borders* on, a shell's neighbors must share both its tile and its material.
- Select and Hide act on the shells as the UV Editor shows them, with the materials the list shows. Reveal looks at whole shells, hidden faces included, to decide which to bring back.
- Slots holding the same material count as one material set.
- **Select** works like Blender's own selection: with UV Sync Selection on it selects mesh faces, otherwise it selects UVs and leaves the mesh selection alone.
- **Hide** follows Blender's *UV → Hide*. With UV Sync Selection on, the faces are hidden in the mesh, so they disappear in the 3D Viewport too. With it off, the UV Editor shows only the faces selected in the mesh, so hiding deselects them there.
- The selection of other faces is kept when you hide a set.
- **Reveal** undoes either kind of hiding and selects what it reveals.
- All three can be undone.
- A change of what is hidden makes the add-on read only the objects it touched, and measure again only the materials whose shells changed.

<a id="auto-low--high"></a>**Auto Low / High.** While it is on and texel density is shown (in the UV Editor or the 3D Viewport), Low and High are filled with the lowest and highest density of the checked material sets (all shells when none is checked):
- when you turn it on,
- when you check or uncheck a set,
- when other objects are shown: others enter Edit Mode, or another selection in Object Mode,
- when *Texture* or the scene's *Unit Scale* changes, since every density changes with them,
- after Undo or Redo.

They are not refilled while you edit UVs, so the colors stay put while you work. Press the fill button (↻) to refill after edits. Typing a value into Low or High turns Auto off, so what you typed stays, also after reopening the file. Files saved with an earlier version whose Low or High had been changed open with Auto off, keeping those values.

With *Active Image* and two UV Editors showing different images, the filled values follow the texture size the 3D Viewport uses (the first UV Editor's image).

**Orientation arrows.** For each face, the add-on fits how its UVs map onto its 3D surface. The scene's up axis (+Z), projected onto the face, is then taken back into UV space. Faces are weighted by their area and by how much of the axis lies along them. The result is averaged over the shell.
- When up runs mostly across the shell (a wall, a slope), the arrow is blue and points up the surface.
- On shells lying flat (a floor, a table top), up points out of the surface, so the arrow shows +Y instead (green).
- A shell with neither direction clear gets no arrow; a dome seen from above is an example.

---

## Choosing padding values

Bake margin (dilation) grows outward from every shell. Two neighboring shells each need room for their own margin, so the gap between them should be about **twice** the margin. A shell needs only **one** margin to the tile border. That is why the border defaults (2 / 4 px) are half the shell defaults (4 / 8 px).

If you rely on mipmaps, each mip level halves the distance in pixels, so textures seen from far away need more padding.

---

## Performance

How work is saved:
- Meshes are read as whole arrays, never face by face, and only the objects that changed are read again. Small objects are processed together. Shells, borders, materials and texel density are built with vectorized NumPy code.
- An object keeps its shells for as long as its UVs, its materials and what the UV Editor shows of it stay the same. When Blender reports a change that turns out to change nothing, the shells are kept.
- Stacks, overlaps and gaps are found material by material (with *Same Material Only*) and kept per material. After an edit, only the materials whose shells changed are measured again.
- A stack is measured once: no gaps from its copies, no crossing points between them.
- With UV Sync Selection on, a selection click measures nothing again, unless *Selected Shells Only* is on. With it off, the mesh selection decides which faces the UV Editor shows, so the objects you clicked in are checked again.
- Measurements are redone only when a relevant setting changes. The last few are kept, so going back to an earlier value costs nothing.
- Everything drawn for a view is reused until you pan or zoom or something changes.
- Texel density colors are computed on the GPU from each shell's density. Changing Low, Needed or High, the unit, the texture size or the opacity rebuilds nothing. Moving or rotating an object doesn't either; only scaling it does.
- Orientation is found from the first time the arrows are shown. Switching them off and on again costs nothing.
- In the 3D Viewport, objects are prepared a few at a time: each frame spends at most about 30 ms on it, and the rest follow in the next frames. Selecting many objects never freezes the viewport.

All timings below are CPU-side and measured headless with the default settings: the time the add-on's Python code takes. GPU drawing time is not included.

**Many objects in Edit Mode at once.** The test scene is laid out like a vehicle with many repeated parts: 1,307 objects, 49k faces, 8,155 shells of 4 materials. 7,634 of the shells lie in 1,429 stacks. Blender 4.5, UV Sync Selection on, several runs:

| | 1.4.0 | 1.5.0 |
|---|---|---|
| First display | 8.8–10.4 s | 0.8–1.2 s |
| After a selection click | 9.0–10.5 s | 15–27 ms |
| After moving the UVs of one object | 9.4–11.4 s | 0.2–0.3 s |
| Orientation Arrows on | 9.6–11.1 s, every time | 0.3–0.6 s the first time, then 15–70 ms |
| Texel Density on | 0.75–0.9 s | 0.05–0.09 s |
| Points per Shell changed | 6.7–8.3 s | 0.25–0.4 s |
| Hiding one material: the button, then the redraw | 0.6–0.9 s, then 6.1–7.7 s | 0.4–0.6 s, then 0.25–0.3 s |
| Revealing it again | 0.9–1.0 s, then 9.4–9.8 s | 0.45–0.6 s, then 0.35–0.5 s |
| Redraw, same view | 13–15 ms | 2–5 ms |
| Redraw while zooming, per frame | 78–91 ms | 15–25 ms |
| Select Flipped: the button, then the redraw | 0.5–0.6 s, then 9.1–9.4 s | 0.45–0.5 s, then about 0.02 s |
| Select Overlapping, Stacked or All but One: the button, then the redraw | — | 0.45–0.5 s, then about 0.02 s |
| Move All but One (7,317 shells moved): the button, then the redraw | — | 0.7–0.8 s, then 0.9–1.0 s |

- Where the first display's time goes in 1.5.0: reading the 1,307 meshes from Blender 0.25–0.4 s; shells, borders and materials about 0.3 s; stacks, overlaps and gaps about 0.3 s; labels and drawing data about 0.04 s.
- With UV Sync Selection off, a selection click costs about 0.15 s when all 1,307 objects are involved.
- With *Stacked Shells as One* off, the same scene has 61,000–70,000 overlapping pairs, and measuring it takes 1.7–2.6 s.
- The buttons read every object again before they act, so that they work on the meshes as they are. With 1,307 objects that reading is most of a button's time. The redraw after a Select button reads nothing again; after Hide, Reveal and Move it reads the objects that changed.
- Blender 5.1 and 5.2 ship NumPy 2.3, where one function the overlap search of 1.4.0 leaned on (`unique`) is several times slower on large arrays. The first display of this scene took 1.4.0 12.5 s in Blender 5.2 with that NumPy; 1.5.0 takes 0.9 s there.

**One object.** A grid of separate shells without stacks, where nothing can be skipped. Blender 4.5, the whole redraw after each kind of change:

| Redraw after | 144 shells, 1.3k faces: 1.4.0 | 1.5.0 | 1,600 shells, 14.4k faces: 1.4.0 | 1.5.0 |
|---|---|---|---|---|
| a UV edit | ~26 ms | ~23 ms | ~0.34 s | ~0.25 s |
| a setting change (*Shift Along Border*) | ~18 ms | ~15 ms | ~0.28 s | ~0.19 s |
| a selection click | ~25 ms | ~0.4 ms | ~0.34 s | ~0.7 ms |
| nothing (same view) | ~0.3 ms | ~0.3 ms | ~0.5 ms | ~0.5 ms |
| a pan or zoom step | ~3.3 ms | ~3.3 ms | 16–22 ms | 20–22 ms |

The larger layout shows 37,000 gap lines at once. Panning and zooming use the same drawing code as before, so their cost didn't change.

When a step takes longer than about 40 ms, the add-on waits until you pause for ~0.25 s and dims the overlay meanwhile. Editing and dragging sliders stay responsive.

For very large layouts, lower *Points per Shell*: the gap measurement grows with points × shells. *Selected Shells Only* also helps while you work on part of the layout.

**3D Viewport, 150 selected objects with 240k faces** (Blender 5.2, headless with software OpenGL; CPU work only):

| | 1.3.0 | 1.4.0 |
|---|---|---|
| Turning the overlay on | all objects prepared in one frame: ~0.41 s | at most ~30 ms per frame; all objects colored after 12 frames |
| Changing Needed, Low or High | ~76 ms to recolor | nothing to rebuild |
| Python work per redraw (profiled) | ~4.6–4.9 ms | ~3–3.8 ms |

**Why not temporary materials?** Swapping the colors for temporary per-density materials (for example a `200PXmTexelMaterial` shown in the viewport's material color) was measured on the same 150-object scene. It is slower on every change:
- Assigning the materials and letting Blender re-evaluate took 113–145 ms. With one material per 1 px/m it took 522 ms, for 1,103 materials. Restoring took 70–105 ms, or 1.3 s at 1 px/m.
- Every change also makes Blender rebuild each mesh's viewport data, which can't be measured headless.
- Drawing needs one draw call per material slot: 1,800–2,400 per frame, against 150 for the overlay.
- It would also edit your meshes' materials and face assignments, clutter undo, and could leave the temporary materials in a saved file.

The overlay stays, with the GPU coloring above.

### What the GPU does, and what it can't

The add-on draws through Blender's own GPU module, so it runs on whichever graphics backend Blender uses: OpenGL, Metal or Vulkan. Nothing in it is written for OpenGL only, and it needs no setting: when Blender runs on Vulkan (a choice in Blender's Preferences in 4.5 and later), so does the overlay.

What runs on the GPU:
- All drawing. Lines, markers, label boxes, arrows and fills go to the GPU as a few batches per redraw, not shape by shape. Text goes through Blender's font drawing.
- Texel density colors, in the UV Editor and in the 3D Viewport. A small shader turns each shell's density into its color, which is why thresholds, units and opacity change without rebuilding anything.

What stays on the CPU, and why:
- **Reading the meshes.** Blender hands UVs and faces to Python on the CPU; a shader can't fetch them. This is about a third of a first display in 1.5.0.
- **Finding shells, stacks and overlaps.** These are searches over connected faces and candidate pairs, where the amount of work depends on the data at every step. NumPy arrays fit them well; shaders don't.
- **The nearest-border search behind the gap lines.** This is the one step a compute shader would suit. It takes 0.1–0.2 s of the 0.8–1.2 s first display above. Blender 3.6 has no compute shaders in its Python API. The 4.2, 4.5 and 5.2 builds tested have them, but there data goes in as textures or uniform blocks and comes back only as a texture. A second code path for the newer versions would save little, and it would have to give the same results as the first.
- **Placing labels** so that they don't cover each other. That is decided label by label, in order of importance.

So the gains of 1.5.0 come from doing less work, not from moving work to the GPU: reading only what changed, measuring a stack once, measuring each material on its own, and avoiding slow library calls. On the scene above, the first display is about 10 times faster than in 1.4.0 and a selection click several hundred times. A change of graphics API would not have given that, because the time was not spent drawing.

What a shader could still add: while you pan or zoom, the screen positions of all visible lines and markers are computed again in Python for every frame. A custom shader could keep them on the GPU and receive only the new view. Measured on the scene above with Blender 5.2 drawing for real (headless, with a software renderer), that is about 10 ms of the 15–25 ms a frame takes in Python; placing the labels, 7–8 ms, would stay. It is not part of 1.5.0.

---

## Limitations

- Gaps are sampled at the measurement points. A narrow spot between two points is found by adding points or sweeping *Shift Along Border*. Overlaps are always found.
- Each point reports only its nearest neighbor.
- UVs meant to wrap around (tiling textures) are not treated as wrapping. With Tile Borders on, every tile is a separate texture.
- Flipped detection works per shell, based on its overall orientation. Individual folded faces inside an otherwise normal shell are not flagged.
- Texel density is an average per shell. Stretching inside a shell, where one part is denser than another, is not shown.
- Orientation arrows show a shell's average direction. On strongly curved shells, such as a cylinder unwrapped around its axis, the direction changes across the shell.
- A shell whose faces use several materials counts as the material covering most of it.
- Stacks are told by their borders, holes included. Two different parts with the same outline in the same place form a stack too; they do share that texture space. A part reused at another size or turned in UV space doesn't cover its original and is an overlap.
- *Move All but One* moves shells by whole tiles along U from where they are. It doesn't look for an empty tile: shells already in the target tile get the moved ones on top.
- The copies moved away lie on each other in their new tile as they did before, and the overlay keeps counting them there as stacks and overlaps. The panels say how many are left in the 0–1 tile.
- The Select and Move buttons work on the faces the UV Editor shows. With UV Sync Selection off those are the faces selected in the mesh: select all of the mesh first, or only the shown part of a shell is moved.
- Without UV Sync Selection, hiding a material set deselects its faces in the mesh, because that is what the UV Editor shows. In vertex select mode a face whose corners are all shared with selected faces can come back after a later selection change.
- In Edit Mode the 3D Viewport colors the edit cage. With modifiers shown in Edit Mode, such as Subdivision Surface or Mirror, the colors follow the cage, not the modifier result.
- Info blocks appear in the UV Editor only; the 3D Viewport shows the colors.
- Only the active UV map of each mesh is measured.

---

## Troubleshooting

- **Nothing is drawn.** Check that:
  - you are in Edit Mode;
  - the editor is a UV Editor, not the Image Editor in View mode;
  - the panel's header checkbox is on;
  - the UV Editor's *Overlays* are enabled.

  With UV Sync Selection off, only faces selected in the mesh are shown, and only those are measured.
- **No colors in the 3D Viewport.** Check the header checkbox of the Texel Density panel in the 3D Viewport sidebar, and the viewport's *Overlays*. In Object Mode, the objects must be selected and have a UV map.
- **Numbers look too small or too large.** Check *Texture*: distances and densities scale with the resolution. For texel density also check the scene's *Unit Scale* (*Scene Properties → Units*).
- **Low and High keep changing.** *Auto Low / High* is on and the checked material sets, the shown objects, *Texture* or *Unit Scale* changed. Type a value, or untick *Auto Low / High*, to keep your own values.
- **A material is missing from the list.** The list shows materials used by faces of the shown objects: in Edit Mode, the objects being edited. It updates on the next redraw after a change.
- **Copies I stacked are reported as overlaps.** They cover each other by less than *Stack Match*. Lower it (99 %, say), or select them with *Select: Overlapping* and snap them onto each other.
- **Stacked copies are no longer flagged as overlaps.** That is *Stacked Shells as One* in the Overlaps and Stacks subpanel. Turn it off to see every copy as an overlap again.
- **Move All but One says there is nothing to move.** With *Only in the 0-1 Tile* on, it takes shells of the 0–1 tile only, and none of them lies on another: the copies are in another tile already. Turn the option off to treat every tile.
- **Overlaps are still counted after Move All but One.** The moved copies lie on each other in the tile they went to. The line `0 of them in the 0-1 tile` in the summary says the 0–1 tile is clean.
- **A mesh named `.UV Gap Overlay scratch` appears under *Blender File* in the Outliner.** The add-on uses this empty helper mesh to read edit-mode meshes quickly. It has no users, is never saved with your file, and is removed when the add-on is turned off.
- **The add-on appears twice in the Add-ons list.** Both the `.py` and the `.zip` are installed. Uninstall one of them and restart Blender.
- **Something went wrong.** The add-on prints each error once to the system console. On Windows use *Window → Toggle System Console*; on other systems start Blender from a terminal. If your graphics driver can't compile the add-on's texel density shader, the console says so. The colors then come from a slower built-in path that looks the same.

---

## Changelog

**1.5.0**
- Stacks: shells lying on each other on purpose, such as copies of a part or mirrored halves, are measured as one shell and are no longer reported as overlaps (*Stacked Shells as One*, *Stack Match*, 99.9 % by default). What is left as an overlap is what doesn't fit: careless copies, shells lying partly on others.
- A stack's info block shows `× N`, how many of its shells are flipped, the span of its texel densities and `Scale varies`. Mirrored copies in a stack with an unflipped shell are not tinted as flipped.
- New *Overlaps and Stacks* subpanel: select the overlapping shells, the stacked shells, or all but one of every stack and overlap. **Move All but One** takes those to another UDIM tile and leaves one shell in place, an unflipped one where there is one.
- The summaries count the stacks. Once shells lie on each other in other tiles too, they also say how many overlaps and stacked copies are left in the 0–1 tile.
- Much faster with many objects in Edit Mode: only the objects that changed are read again, each material is measured on its own and kept, and a stack is measured once. On a scene of 1,307 objects the first display went from 9–10 s to about 1 s, and the wait after a selection click from about 10 s to about 0.02 s. See [Performance](#performance).
- After *Select Flipped*, after selecting material sets and after *Select by Range*, the overlay no longer computes everything again: it computes nothing, unless *Selected Shells Only* is on. After hiding or revealing a material set, only the objects that changed are worked on.
- Switching *Orientation Arrows* off and on again costs nothing after the first time.
- Faster in Blender 5.1 and 5.2, whose NumPy made one step of 1.4.0 several times slower.
- Fixed: with UV Sync Selection off, *Selected Shells Only* did not follow a change of the UV selection until something else changed. Blender doesn't announce such changes, so the add-on now looks for them.
- Changed: when two materials cover exactly the same area and number of faces of one shell, the shell now takes the one that comes first in its object's material slots (before: the first met among all the objects).
- Looked at and not done: moving the measuring to the GPU. See [What the GPU does, and what it can't](#what-the-gpu-does-and-what-it-cant).

**1.4.0**
- Material sets: a list of the materials on the shown objects, with shell counts and density ranges. Select, hide and reveal their shells, for one material or several checked ones.
- *Same Material Only*: gaps and overlaps are measured only between shells of the same material. On by default.
- One info block per shell holds its texel density, a `Scale …` note when its object's scale is not 1, and *Flipped*, so these no longer cover each other.
- Orientation arrows in the info blocks: where the scene's up (+Z) runs across each shell, or +Y on shells lying flat.
- *Auto Low / High*: Low and High fill themselves with the lowest and highest density of the checked material sets, and stay editable. Files from earlier versions with Low or High changed keep their values (Auto starts off for them).
- *Select by Range*: From / To sliders over the analyzed densities outline the shells in range and select them.
- Texel density colors are computed on the GPU. Changing thresholds, units, texture size or opacity no longer rebuilds anything; nor does moving or rotating objects.
- 3D Viewport: objects are prepared a few per frame, so turning the overlay on with many objects selected no longer freezes the view. Less Python work per redraw.
- The Display settings (font size, opacity, label background) now apply to every label, not only to the gap overlay.
- Evaluated and rejected: temporary per-density materials for the 3D Viewport (slower on every change, and it would edit your data; see *Performance*).

**1.3.0**
- Texel density per UV shell in the UV Editor: a color overlay (red at Low, green at Needed, blue at High, a gradient in between) and a label on every shell, in px/cm, px/m, px/in or px/ft.
- Texel density on the mesh in the 3D Viewport, with its own sidebar panel. Edit Mode colors the edit cage; Object Mode colors the selected objects, modifiers included.
- The gap overlay and the texel density overlay can be turned on independently.
- Faster: meshes are read as whole arrays instead of face by face, so re-reading after an edit is 4–5 times faster. Redrawing an unchanged view reuses everything already drawn.

**1.2.1**
- Fixed: moving UVs with the overlay on could scramble the UV layout. When the overlay refreshed during a drag, Blender moved the mesh data in memory under the running tool. The overlay now reads a private copy of the mesh and never touches the one being edited.

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

## License

GPL-3.0-or-later
