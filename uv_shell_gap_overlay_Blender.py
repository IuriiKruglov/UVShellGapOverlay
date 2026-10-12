# SPDX-License-Identifier: GPL-3.0-or-later
#
# UV Shell Gap Overlay  -  Blender add-on for 3.6 LTS to 5.2 LTS
#
# Draws, along the shell borders in the UV Editor, the pixel distance between
# neighboring UV shells and from shells to their UDIM tile border, graded
# against minimal / needed padding; marks overlapping, tile-crossing and
# flipped (mirrored) shells. Shells stacked on each other on purpose (copies
# sharing texture space) count as one shell; the copies can be selected and
# moved to another UDIM tile. Colors and labels every shell by its texel
# density, in the UV Editor and on the mesh in the 3D Viewport. One info
# block per shell collects its texel density, object scale, flip state and
# an arrow for the scene's up direction. Material sets: select, hide and
# reveal the shells of chosen materials, measure gaps only within a material.
#
# Authors: Iurii Kruglov & Claude (Anthropic)

bl_info = {
    "name": "UV Shell Gap Overlay",
    "author": "Iurii Kruglov, Claude (Anthropic)",
    "version": (1, 5, 0),
    "blender": (3, 6, 0),
    "location": "UV Editor / 3D Viewport > Sidebar (N) > UV Gaps",
    "description": "Pixel gaps between UV shells and to UDIM tile borders, texel density per shell "
                   "(UV Editor and 3D Viewport), per-shell info blocks, material sets; overlap, stack, "
                   "tile-crossing and flipped-shell checks",
    "category": "UV",
}

import time
import traceback

import bpy
import bmesh
import blf
import gpu
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

UV_QUANT = 1.0e6         # loops on one mesh vertex whose UVs agree to 1e-6 share a UV vertex
UV_EPS = 1.0e-6          # inset (UV units) for shell-inside-shell probes
PX_EPS = 1.0e-3          # inset (texture px) for per-point overlap probes
TILE_EPS = 1.0e-6        # UV tolerance: a shell touching a tile line does not cross it
EDGE_FACING = 0.25       # a border point measures only toward what its outward normal faces:
                         # at most ~75 deg off the normal (cos >= 0.25), for tile edges and neighbors
CROSS_TOL = 1.0e-7       # parametric tolerance: meeting at segment ends is touching, not crossing
CHUNK = 1 << 20          # max candidate pairs per vectorized block (bounds memory use)
DENSE_CHUNK = 1 << 19    # max elements per dense distance block
MAX_GRID_ROWS = 1 << 23  # max (box, cell) rows when gridding shell bounding boxes
MAX_PIECES = 200000      # max segment pieces when gridding borders
PROBES_PER_SHELL = 8
MAX_CROSS_MARKS = 48     # crossing markers kept per overlapping shell pair / tile-crossing shell
LIVE_MS = 40.0           # work cheaper than this is redone on every change
DEBOUNCE_S = 0.25        # heavier work waits until changes pause this long
TD3D_BUDGET_MS = 30.0    # 3D view: objects are (re)built until a frame has used this much, the rest next frame
ORIENT_MIN = 0.3         # an arrow needs the axis to lie this much along the shell (area-weighted, 0..1)
FONT_ID = 0
NO_MATERIAL = "(No Material)"
BATCH_MESH_FACES = 20000  # objects with fewer faces are worked on together, in batches
BATCH_FACES = 300000      # of at most this many faces
SIG_CHECK_S = 0.5         # how often the objects are checked for changes Blender doesn't announce

# Blender 5.0 moved UV selection from BMLoopUV.select / select_edge to
# BMLoop.uv_select_vert / uv_select_edge and BMFace.uv_select (shared by all UV maps).
_UV_SELECT_ON_LOOP = hasattr(bmesh.types.BMLoop, "uv_select_vert")

_EMPTY2 = np.empty((0, 2))
_EMPTY_F = np.empty(0)
_EMPTY_I = np.empty(0, dtype=np.int64)
_EMPTY_I32 = np.empty(0, dtype=np.int32)
_KEY_OFF = 1 << 30
_KEY_MUL = 1 << 31
_TILE_OFF = 1 << 20
_TILE_MUL = 1 << 21


# ---------------------------------------------------------------------------
# Runtime state (caches; nothing here is saved)
# ---------------------------------------------------------------------------

class _State:
    draw_handle = None
    draw3d_handle = None
    geo = None
    geo_sig = None
    geo_flags = None
    geo_dirty = True     # a mesh or an object changed since the shells were read
    sel_dirty = False    # only a selection changed: matters where it decides what is shown or measured
    geo_version = 0
    sel_version = 0      # bumped when only the selected shells of the geometry changed
    quick_key = None     # what the objects were last checked for (see _ensure_geometry)
    sig_time = 0.0
    sel_poll = 0.0       # when the UV selection may be read again (see _poll_uv_selection)
    sel_poll_ms = 0.0
    view_key = {}        # per UV editor region: its view, to tell panning from other redraws
    orient_seen = False  # orientation arrows were shown since the file was opened
    epoch = 0            # bumped when nothing read so far can be trusted (undo, Refresh, file load)
    data_gen = 0         # bumped on every mesh / object change
    id_gen = {}          # original ID pointer -> geometry change count
    touch = {}           # original ID pointer -> change count, selection changes included
    own_sel = None       # IDs whose update, just now, is a selection the add-on wrote (_announce_selection)
    last_change = 0.0
    last_geo_ms = 0.0
    geo_stats = (0, 0)   # objects whose shells were found again / objects, in the last update
    meas = None
    meas_key = None
    meas_version = 0
    meas_geo_version = -1
    last_meas_ms = 0.0
    pending_key = None
    pending_since = 0.0
    meas_cache = {}      # small per-key caches, so several UV editors don't evict each other
    labels_cache = {}
    arrays = {}          # edit-mode object -> (epoch, geometry count, change count, _Arrays, UV Sync
                         # Selection state its selection flags are of, how much of them is right)
    parts = {}           # edit-mode object -> _Parts: its shells, kept while it doesn't change
    mcache = None        # _MeasureCache: stacks, overlaps and gaps per material, kept by content
    stacks = None        # stacks of the geometry (_Stacks) when no measurement provides them
    stacks_key = None
    home = None          # (what for, counts): see _home_counts
    td3d = {}            # object -> _TD3D (3D view texel density)
    td_values = {}       # texel density per shell, per (geometry, texture, unit scale)
    td_fill = {}         # UV editor texel density fill batches
    frame = {}           # per-view draw data, reused while the view and data are unchanged
    td_uv_stats = None   # summary for the panels
    td_3d_stats = None
    td3d_frame = None    # 3D view: (key, stats) of the last frame, reused while nothing changed
    pub = None           # per-shell densities and materials for the panels (_Published):
                         # from the UV editor in Edit Mode, from the 3D view in Object Mode
    auto_key = None      # what Low / High were last auto-filled for
    pending = {}         # scene property writes queued for a timer (not allowed while drawing)
    fast_read = None     # None: untested, True: scratch-mesh reading works, False: fall back
    shader_names = {}
    td_shader = None     # None: not tried yet, False: unavailable (per-vertex colors instead)
    numpy_buffers = True
    last_error = None
    reported = set()

    @classmethod
    def invalidate(cls, hard=False):
        """Read every object again. What turns out unchanged keeps its shells and measurements
        unless `hard` (Refresh): then everything is computed again."""
        cls.geo_dirty = True
        cls.geo_sig = None
        cls.quick_key = None
        cls.meas_key = None
        cls.epoch += 1
        cls.data_gen += 1
        cls.meas_cache.clear()
        cls.labels_cache.clear()
        cls.arrays.clear()
        cls.td3d.clear()
        cls.td_values.clear()
        cls.td_fill.clear()
        cls.frame.clear()
        cls.td3d_frame = None
        cls.stacks = cls.stacks_key = cls.home = None
        if hard:
            cls.parts.clear()
            if cls.mcache is not None:
                cls.mcache.clear()

    @classmethod
    def reset(cls):
        cls.invalidate(hard=True)
        cls.geo = None
        cls.geo_flags = None
        cls.sel_dirty = False
        cls.orient_seen = False
        cls.meas = None
        cls.last_geo_ms = 0.0
        cls.last_meas_ms = 0.0
        cls.pending_key = None
        cls.id_gen.clear()
        cls.touch.clear()
        cls.view_key.clear()
        cls.td_uv_stats = None
        cls.td_3d_stats = None
        cls.pub = None
        cls.auto_key = None
        cls.pending = {}
        cls.fast_read = None


def _report(what, detail=""):
    """Print each distinct problem once to the system console."""
    key = (what, detail)
    if key not in _State.reported:
        _State.reported.add(key)
        print("[UV Shell Gap Overlay] %s%s" % (what, (":\n" + detail) if detail else ""))


def _unique(x):
    """Sorted unique values of an integer array, by sorting: np.unique hashes since NumPy 2.3,
    which takes many times longer on the large arrays of codes met here."""
    x = np.sort(np.asarray(x).reshape(-1))
    if x.size < 2:
        return x
    return x[np.concatenate(([True], x[1:] != x[:-1]))]


def _cache_put(cache, key, value, size=4):
    cache[key] = value
    while len(cache) > size:
        cache.pop(next(iter(cache)))


def _redraw_areas(types=('IMAGE_EDITOR',)):
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for win in wm.windows:
        screen = win.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type in types:
                area.tag_redraw()


def _redraw_uv_editors(*_args):
    _redraw_areas(('IMAGE_EDITOR',))


def _on_setting_changed(self, context):
    # texture size and the like also feed the texel density shown in 3D views
    _redraw_areas(('IMAGE_EDITOR', 'VIEW_3D') if getattr(self, "td_show_3d", False) else ('IMAGE_EDITOR',))


def _on_td_changed(self, context):
    _redraw_areas(('IMAGE_EDITOR', 'VIEW_3D'))


def _on_auto_range(self, context):
    if self.td_auto_range:
        _State.auto_key = None  # fill again on the next redraw
    _redraw_areas(('IMAGE_EDITOR', 'VIEW_3D'))


def _on_range_from(self, context):
    if self.td_sel_to < self.td_sel_from:
        self.td_sel_to = self.td_sel_from  # the handles don't pass each other
    _redraw_areas(('IMAGE_EDITOR', 'VIEW_3D'))


def _on_range_to(self, context):
    if self.td_sel_from > self.td_sel_to:
        self.td_sel_from = self.td_sel_to
    _redraw_areas(('IMAGE_EDITOR', 'VIEW_3D'))


# ---------------------------------------------------------------------------
# Settings (Scene.uv_gap_overlay)
# ---------------------------------------------------------------------------

_RES_ITEMS = [
    ('256', "256", "256 x 256 px"),
    ('512', "512", "512 x 512 px"),
    ('1024', "1024", "1024 x 1024 px"),
    ('2048', "2048", "2048 x 2048 px"),
    ('4096', "4096", "4096 x 4096 px"),
    ('8192', "8192", "8192 x 8192 px"),
    ('CUSTOM', "Custom", "Custom width and height (non-square textures supported)"),
    ('IMAGE', "Active Image", "Size of the image shown in this UV editor"),
]

# texel density units: pixels per unit = pixels per meter * factor
_TD_UNITS = (
    ('PX_CM', "px/cm", "Pixels per centimeter", 0.01),
    ('PX_M', "px/m", "Pixels per meter", 1.0),
    ('PX_IN', "px/in", "Pixels per inch", 0.0254),
    ('PX_FT', "px/ft", "Pixels per foot", 0.3048),
)
_TD_FACTOR = {u[0]: u[3] for u in _TD_UNITS}
_TD_LABEL = {u[0]: u[1] for u in _TD_UNITS}


def _td_value(stored, name, description, manual=False):
    """A texel density shown in the chosen unit but stored in px/m, so switching the unit
    converts the value instead of reinterpreting it. `manual`: typing a value turns the
    Low / High auto-fill off, so the typed value stays (also after reopening the file)."""
    def get_value(self):
        return float(getattr(self, stored)) * _TD_FACTOR.get(self.td_unit, 1.0)

    def set_value(self, value):
        setattr(self, stored, max(0.0, float(value)) / _TD_FACTOR.get(self.td_unit, 1.0))
        if manual and self.td_auto_range:
            self.td_auto_range = False

    return FloatProperty(name=name, description=description, get=get_value, set=set_value,
                         min=0.0, soft_max=100000.0, step=100, precision=2, update=_on_td_changed)


class UVGAP_MaterialItem(bpy.types.PropertyGroup):
    """One material set of the Materials list (`name` is what the list shows)."""
    key: StringProperty(
        name="Material", description="Full name of the material; empty for faces without a material",
        options={'HIDDEN'})
    checked: BoolProperty(
        name="Use",
        description="Include this material set: Low / High auto-fill, Select, Hide and Reveal act on "
                    "the checked sets",
        default=False, update=_on_td_changed)


class UVGAP_Settings(bpy.types.PropertyGroup):
    show: BoolProperty(
        name="Show Overlay", description="Draw the UV shell gap overlay",
        default=True, update=_on_setting_changed)

    # texture resolution
    resolution: EnumProperty(
        name="Texture", description="Texture resolution used to convert UV distances to pixels",
        items=_RES_ITEMS, default='2048', update=_on_setting_changed)
    res_x: IntProperty(
        name="Width", description="Custom texture width",
        default=2048, min=1, soft_max=16384, subtype='PIXEL', update=_on_setting_changed)
    res_y: IntProperty(
        name="Height", description="Custom texture height",
        default=2048, min=1, soft_max=16384, subtype='PIXEL', update=_on_setting_changed)

    # measurement points
    points: IntProperty(
        name="Points per Shell",
        description="Measurement points spread evenly along each shell's border (outer border and holes)",
        default=24, min=1, max=512, soft_max=128, update=_on_setting_changed)
    shift: FloatProperty(
        name="Shift Along Border",
        description="Slide all measurement points along the shell borders, "
                    "as a fraction of the spacing between neighboring points",
        default=0.0, min=0.0, max=100.0, subtype='PERCENTAGE', precision=0,
        update=_on_setting_changed)

    # shell-to-shell thresholds
    min_px: FloatProperty(
        name="Minimal", description="Smallest acceptable gap between shells, in texture pixels",
        default=4.0, min=0.0, soft_max=256.0, step=100, precision=1, subtype='PIXEL',
        update=_on_setting_changed)
    needed_px: FloatProperty(
        name="Needed", description="Target gap between shells, in texture pixels",
        default=8.0, min=0.0, soft_max=256.0, step=100, precision=1, subtype='PIXEL',
        update=_on_setting_changed)
    search_px: FloatProperty(
        name="Search Radius",
        description="Gaps up to this size are measured; shells farther apart (or farther from "
                    "their tile border) are not treated as neighbors",
        default=32.0, min=0.1, soft_max=512.0, step=100, precision=1, subtype='PIXEL',
        update=_on_setting_changed)
    selected_only: BoolProperty(
        name="Selected Shells Only",
        description="Measure only from shells that have selected UVs (distances still go to every shell)",
        default=False, update=_on_setting_changed)
    same_material: BoolProperty(
        name="Same Material Only",
        description="Measure gaps and find overlaps only between shells of the same material (each "
                    "material has its own texture). A shell's material is the one covering most of "
                    "its UV area",
        default=True, update=_on_setting_changed)

    # stacks: shells lying on each other on purpose
    stacks: BoolProperty(
        name="Stacked Shells as One",
        description="Shells covering each other at least as much as Stack Match - copies of a part sharing "
                    "texture space - count as one shell: gaps are measured once per stack and they aren't "
                    "reported as overlaps. Only shells that don't fit (a copy placed carelessly, shells "
                    "overlapping partly) are overlaps",
        default=True, update=_on_setting_changed)
    stack_match: FloatProperty(
        name="Stack Match",
        description="Shells covering each other at least this much are a stack; less is an overlap",
        default=99.9, min=50.0, max=100.0, soft_min=90.0, step=10, precision=2, subtype='PERCENTAGE',
        update=_on_setting_changed)
    stack_tiles: IntProperty(
        name="Tiles",
        description="How many UDIM tiles along U Move All but One moves the shells (negative: to the left)",
        default=1, min=-99, max=99)
    stack_home_only: BoolProperty(
        name="Only in the 0-1 Tile",
        description="All but One (Select and Move) takes only shells lying in the 0-1 tile (UDIM 1001): copies "
                    "moved to another tile earlier stay where they are, so pressing Move All but One again "
                    "changes nothing. Off: the shells of every tile",
        default=True)

    # UDIM tile borders
    tile_border: BoolProperty(
        name="UDIM Tile Borders",
        description="Measure shells near the border of their UDIM tile and flag shells crossing a tile "
                    "border. Each tile is treated as its own texture: gaps are measured only between "
                    "shells in the same tile",
        default=True, update=_on_setting_changed)
    border_min_px: FloatProperty(
        name="Border Minimal",
        description="Smallest acceptable distance from a shell to its tile border, in texture pixels "
                    "(a shell dilates toward the border from one side only, so usually half the shell gap)",
        default=2.0, min=0.0, soft_max=256.0, step=100, precision=1, subtype='PIXEL',
        update=_on_setting_changed)
    border_needed_px: FloatProperty(
        name="Border Needed",
        description="Target distance from a shell to its tile border, in texture pixels",
        default=4.0, min=0.0, soft_max=256.0, step=100, precision=1, subtype='PIXEL',
        update=_on_setting_changed)

    # flipped shells
    show_flipped: BoolProperty(
        name="Show Flipped Shells",
        description="Highlight shells whose UVs are mirrored (clockwise winding in UV space)",
        default=True, update=_on_setting_changed)

    # display
    font_size: IntProperty(
        name="Font Size", default=12, min=6, max=72, subtype='PIXEL', update=_on_setting_changed)
    opacity: FloatProperty(
        name="Opacity", default=0.9, min=0.0, max=1.0, subtype='FACTOR', update=_on_setting_changed)
    line_width: FloatProperty(
        name="Line Width", default=2.0, min=1.0, max=10.0, precision=1, subtype='PIXEL',
        update=_on_setting_changed)
    label_background: BoolProperty(
        name="Label Background", description="Dark box behind labels for readability over textures",
        default=True, update=_on_setting_changed)
    declutter: BoolProperty(
        name="Hide Overlapping Labels",
        description="Skip labels that would cover a more critical one (problems and smallest gaps win)",
        default=True, update=_on_setting_changed)

    # colors
    color_overlap: FloatVectorProperty(
        name="Overlap", subtype='COLOR_GAMMA', size=3, min=0.0, max=1.0,
        default=(1.0, 0.2, 0.9), update=_on_setting_changed)
    color_zero: FloatVectorProperty(
        name="Touching (0 px)", subtype='COLOR_GAMMA', size=3, min=0.0, max=1.0,
        default=(1.0, 0.1, 0.1), update=_on_setting_changed)
    color_min: FloatVectorProperty(
        name="At Minimal", subtype='COLOR_GAMMA', size=3, min=0.0, max=1.0,
        default=(1.0, 0.8, 0.1), update=_on_setting_changed)
    color_needed: FloatVectorProperty(
        name="At Needed", subtype='COLOR_GAMMA', size=3, min=0.0, max=1.0,
        default=(0.25, 0.95, 0.35), update=_on_setting_changed)
    color_flipped: FloatVectorProperty(
        name="Flipped", subtype='COLOR_GAMMA', size=3, min=0.0, max=1.0,
        default=(0.2, 0.65, 1.0), update=_on_setting_changed)

    # texel density
    td_show: BoolProperty(
        name="Texel Density",
        description="Color every UV shell by its texel density (red at Low, green at Needed, blue at "
                    "High, a gradient in between) and label it with its density",
        default=False, update=_on_td_changed)
    td_show_3d: BoolProperty(
        name="Show in 3D Viewport",
        description="Color mesh faces in the 3D viewport by the texel density of their UV shell "
                    "(objects in Edit Mode, or the selected objects in Object Mode)",
        default=False, update=_on_td_changed)
    td_unit: EnumProperty(
        name="Unit", description="Unit of the texel density values",
        items=[(k, label, desc) for k, label, desc, _f in _TD_UNITS], default='PX_M',
        update=_on_td_changed)
    td_needed_m: FloatProperty(default=300.0, min=0.0, options={'HIDDEN'})
    td_low_m: FloatProperty(default=100.0, min=0.0, options={'HIDDEN'})
    td_high_m: FloatProperty(default=500.0, min=0.0, options={'HIDDEN'})
    td_needed: _td_value("td_needed_m", "Needed", "Target texel density, shown green")
    td_low: _td_value("td_low_m", "Low", "Texel density shown red; lower densities are red too. "
                      "Typing a value turns Auto Low / High off", manual=True)
    td_high: _td_value("td_high_m", "High", "Texel density shown blue; higher densities are blue too. "
                       "Typing a value turns Auto Low / High off", manual=True)
    td_fill_opacity: FloatProperty(
        name="Fill Opacity", description="Opacity of the texel density colors",
        default=0.35, min=0.0, max=1.0, subtype='FACTOR', update=_on_td_changed)
    td_auto_range: BoolProperty(
        name="Auto Low / High",
        description="Fill Low and High with the lowest and highest texel density of the checked "
                    "material sets (all shells when none is checked) whenever the checked sets or the "
                    "objects change. Typing into Low or High turns this off and keeps the typed values",
        default=True, update=_on_auto_range)

    # texel density range selection (percent of the analyzed lowest..highest density)
    td_sel_from: FloatProperty(
        name="From", description="Lower end of the range: 0% is the lowest texel density of the shells, "
                                 "100% the highest",
        default=0.0, min=0.0, max=100.0, subtype='PERCENTAGE', precision=1, update=_on_range_from)
    td_sel_to: FloatProperty(
        name="To", description="Upper end of the range: 0% is the lowest texel density of the shells, "
                               "100% the highest",
        default=100.0, min=0.0, max=100.0, subtype='PERCENTAGE', precision=1, update=_on_range_to)
    td_range_highlight: BoolProperty(
        name="Highlight Range",
        description="Outline the shells inside the range in the UV Editor while the range is narrower "
                    "than all shells",
        default=True, update=_on_td_changed)

    # per-shell info blocks
    info_show: BoolProperty(
        name="Shell Info",
        description="One info block per shell with its texel density, object scale, flip state and "
                    "orientation arrow, placed so blocks don't overlap",
        default=True, update=_on_setting_changed)
    show_orientation: BoolProperty(
        name="Orientation Arrows",
        description="Arrow in each shell's info block pointing where the scene's up (+Z) runs across "
                    "the shell (blue); on shells lying flat, where +Y runs (green)",
        default=False, update=_on_setting_changed)
    show_scale: BoolProperty(
        name="Object Scale",
        description="Note the object scale on the shells of objects whose scale is not 1",
        default=True, update=_on_setting_changed)

    # material sets
    materials: CollectionProperty(type=UVGAP_MaterialItem)
    material_index: IntProperty(name="Active Material Set", default=0, min=0)

    # 140: Low / High managed by 1.4 (see _migrate_settings)
    settings_version: IntProperty(default=0, options={'HIDDEN'})


def _resolve_resolution(st, space):
    """(width, height, image name or None)."""
    mode = st.resolution
    if mode == 'CUSTOM':
        return max(1, st.res_x), max(1, st.res_y), None
    if mode == 'IMAGE':
        img = getattr(space, "image", None) if space is not None else None
        if img is not None:
            try:
                w, h = int(img.size[0]), int(img.size[1])
            except Exception:
                w = h = 0
            if w > 0 and h > 0:
                return w, h, img.name
        return max(1, st.res_x), max(1, st.res_y), None
    n = int(mode)
    return n, n, None


def _first_uv_editor_image():
    """Texture shown in a UV editor (texture size for the 3D view in Active Image mode):
    UV editors first, then other image editors; render results don't count."""
    wm = getattr(bpy.context, "window_manager", None)
    found = []
    for win in (wm.windows if wm is not None else ()):
        screen = win.screen
        for area in (screen.areas if screen is not None else ()):
            if area.type == 'IMAGE_EDITOR':
                space = area.spaces.active
                img = getattr(space, "image", None)
                if img is not None and img.type == 'IMAGE' and img.size[0] > 0 and img.size[1] > 0:
                    found.append((getattr(space, "mode", None) != 'UV', len(found), img))
    return min(found)[2] if found else None


def _resolve_resolution_3d(st):
    if st.resolution == 'IMAGE':
        img = _first_uv_editor_image()
        if img is not None:
            return int(img.size[0]), int(img.size[1]), img.name
    return _resolve_resolution(st, None)


def _unit_scale(scene):
    """Meters per Blender unit."""
    us = getattr(scene, "unit_settings", None)
    if us is None or us.system == 'NONE' or not us.scale_length > 0.0:
        return 1.0
    return float(us.scale_length)


# ---------------------------------------------------------------------------
# UV shells from the edit mesh
# ---------------------------------------------------------------------------

def _edit_mesh_objects(context):
    objs = getattr(context, "objects_in_mode_unique_data", None)
    if objs is None:
        view_layer = getattr(context, "view_layer", None)
        objs = [o for o in view_layer.objects if o.mode == 'EDIT'] if view_layer is not None else []
    out, seen = [], set()
    for obj in objs:
        if obj is None or obj.type != 'MESH':
            continue
        me = obj.data
        if me is None or not me.is_editmode:
            continue
        ptr = me.as_pointer()
        if ptr not in seen:
            seen.add(ptr)
            out.append(obj)
    return out


def _chain_border(edges):
    """Order border edges (a, b, sign) into continuous loops.

    `sign` tells on which side of a->b the shell interior lies (+1 left, -1 right,
    0 unknown); it is flipped when an edge has to be walked backwards.
    """
    out_map, in_map = {}, {}
    for i, (a, b, _s) in enumerate(edges):
        out_map.setdefault(a, []).append(i)
        in_map.setdefault(b, []).append(i)
    used = [False] * len(edges)
    ordered = []
    for i0 in range(len(edges)):
        if used[i0]:
            continue
        used[i0] = True
        a, b, s = edges[i0]
        ordered.append((a, b, s))
        start, cur = a, b
        while cur != start:
            nxt, rev = -1, False
            for j in out_map.get(cur, ()):
                if not used[j]:
                    nxt = j
                    break
            if nxt < 0:
                for j in in_map.get(cur, ()):
                    if not used[j]:
                        nxt, rev = j, True
                        break
            if nxt < 0:
                break  # open border (non-manifold UVs): start a new chain
            used[nxt] = True
            a2, b2, s2 = edges[nxt]
            if rev:
                ordered.append((b2, a2, -s2))
                cur = a2
            else:
                ordered.append((a2, b2, s2))
                cur = b2
    return ordered


# ---------------------------------------------------------------------------
# Reading meshes into arrays
# ---------------------------------------------------------------------------
#
# Fast path: a private copy of the edit mesh is written into a scratch mesh (C code) and
# read with foreach_get, so no Python object is created per face or corner. Walking the
# copy in Python is the fallback. Either way the edit mesh itself is never walked from a
# draw callback (see _read_edit_arrays).

_SCRATCH_MESH = ".UV Gap Overlay scratch"


class _Arrays:
    """Raw data of one mesh: corners (loops), faces, vertices, optionally 3D data."""
    __slots__ = ("uv_name", "sync_valid", "sel_readable", "loop_vert", "uv", "f_start", "f_total",
                 "f_hide", "f_sel", "f_mat", "v_sel", "uv_sel", "co", "tri_loops", "tri_face", "fast")


def _create_scratch_mesh():
    """Timer callback: data-blocks are created here rather than inside a draw callback."""
    try:
        if bpy.data.meshes.get(_SCRATCH_MESH) is None:
            me = bpy.data.meshes.new(_SCRATCH_MESH)
            if me.name != _SCRATCH_MESH:  # the name is taken (linked data): leave it alone
                bpy.data.meshes.remove(me)
    except Exception:
        _report("could not create the helper mesh", traceback.format_exc())
    return None


def _scratch_mesh():
    """A private mesh without users that edit-mesh copies are written into. Looked up by
    name every time: undo and file loads replace data-blocks. Never saved (no users).
    When it is missing, it is created from a timer and this read takes the slow path."""
    me = bpy.data.meshes.get(_SCRATCH_MESH)
    if me is None:
        if not bpy.app.timers.is_registered(_create_scratch_mesh):
            bpy.app.timers.register(_create_scratch_mesh, first_interval=0.0)
        return None
    if me.users or me.library is not None or me.is_editmode:
        return None
    return me


def _get(coll, attr, n, dtype, width=1):
    buf = np.empty(n * width, dtype=dtype)
    if n:
        coll.foreach_get(attr, buf)
    return buf if width == 1 else buf.reshape(n, width)


_ATTR_READ = {'BOOLEAN': ("value", bool, 1), 'INT': ("value", np.int32, 1),
              'FLOAT2': ("vector", np.float32, 2), 'FLOAT_VECTOR': ("vector", np.float32, 3)}
_FLAGS_AS_ATTRIBUTES = bpy.app.version >= (3, 5, 0)  # hide / select stored as bool attributes
_FACE_OFFSETS = bpy.app.version >= (4, 0, 0)         # faces stored as offsets into the corners


def _attr(me, name, domain, data_type, n):
    """Generic attribute as an array (a plain memory copy, far faster than reading through
    the per-item collections), or None when the mesh has no such attribute."""
    att = me.attributes.get(name)
    if att is None or att.domain != domain or att.data_type != data_type:
        return None
    field, dtype, width = _ATTR_READ[data_type]
    return _get(att.data, field, n, dtype, width)


def _flags(me, name, domain, n, coll, field):
    """Hide / select flags: bool attributes that are left out while all False."""
    got = _attr(me, name, domain, 'BOOLEAN', n)
    if got is not None:
        return got
    return np.zeros(n, dtype=bool) if _FLAGS_AS_ATTRIBUTES else _get(coll, field, n, bool)


def _fan_triangles(start, total):
    """(corner triples, face) of a fan triangulation."""
    nt = np.maximum(total.astype(np.int64) - 2, 0)
    face = np.repeat(np.arange(total.size, dtype=np.int64), nt)
    k = np.arange(int(nt.sum()), dtype=np.int64) - np.repeat(np.cumsum(nt) - nt, nt) + 1
    s = start.astype(np.int64)[face]
    return np.column_stack((s, s + k, s + k + 1)).astype(np.int32), face.astype(np.int32)


def _arrays_from_mesh(me, uv_name, want_3d):
    A = _Arrays()
    nL, nF, nV = len(me.loops), len(me.polygons), len(me.vertices)
    A.loop_vert = _attr(me, ".corner_vert", 'CORNER', 'INT', nL)
    if A.loop_vert is None:
        A.loop_vert = _get(me.loops, "vertex_index", nL, np.int32)
    A.uv = _attr(me, uv_name, 'CORNER', 'FLOAT2', nL)
    if A.uv is None:
        A.uv = _get(me.uv_layers[uv_name].data, "uv", nL, np.float32, 2)
    A.f_start = _get(me.polygons, "loop_start", nF, np.int32)
    if _FACE_OFFSETS:  # face corners are contiguous and in face order
        A.f_total = np.diff(np.append(A.f_start, np.int32(nL))).astype(np.int32)
    else:
        A.f_total = _get(me.polygons, "loop_total", nF, np.int32)
    A.f_hide = _flags(me, ".hide_poly", 'FACE', nF, me.polygons, "hide")
    A.f_sel = _flags(me, ".select_poly", 'FACE', nF, me.polygons, "select")
    A.f_mat = _attr(me, "material_index", 'FACE', 'INT', nF)  # left out while every index is 0
    if A.f_mat is None:
        A.f_mat = _get(me.polygons, "material_index", nF, np.int32)
    A.v_sel = _flags(me, ".select_vert", 'POINT', nV, me.vertices, "select")
    # UV selection of the face corners (Blender 5.0+ / Blender 3.5 - 4.x); left out while empty
    A.uv_sel = _attr(me, ".uv_select_vert" if _UV_SELECT_ON_LOOP else ".vs." + uv_name, 'CORNER', 'BOOLEAN', nL)
    if A.uv_sel is None:
        A.uv_sel = np.zeros(nL, dtype=bool)
    A.sel_readable = True
    A.co = A.tri_loops = A.tri_face = None
    if want_3d:
        A.co = _attr(me, "position", 'POINT', 'FLOAT_VECTOR', nV)
        if A.co is None:
            A.co = _get(me.vertices, "co", nV, np.float32, 3)
        me.calc_loop_triangles()
        nT = len(me.loop_triangles)
        A.tri_loops = _get(me.loop_triangles, "loops", nT, np.int32, 3)
        if hasattr(me, "loop_triangle_polygons"):
            A.tri_face = _get(me.loop_triangle_polygons, "value", nT, np.int32)
        else:
            A.tri_face = _get(me.loop_triangles, "polygon_index", nT, np.int32)
    A.fast = True
    return A


def _arrays_from_bmesh(bm, uvl, want_3d):
    """The same arrays, walked in Python (slow path). `bm` must be a private copy."""
    A = _Arrays()
    bm.verts.index_update()
    lv, uv, tot, hide, sel, mat, usel = [], [], [], [], [], [], []
    readable = True
    for f in bm.faces:
        loops = f.loops
        tot.append(len(loops))
        hide.append(f.hide)
        sel.append(f.select)
        mat.append(f.material_index)
        for loop in loops:
            lv.append(loop.vert.index)
            luv = loop[uvl]
            uv.extend(luv.uv)
            if readable:
                try:
                    usel.append(loop.uv_select_vert if _UV_SELECT_ON_LOOP else luv.select)
                except AttributeError:
                    readable = False
    A.loop_vert = np.asarray(lv, dtype=np.int32)
    A.uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    A.f_total = np.asarray(tot, dtype=np.int32)
    A.f_start = np.zeros(len(tot), dtype=np.int32)
    if len(tot) > 1:
        np.cumsum(A.f_total[:-1], out=A.f_start[1:])
    A.f_hide = np.asarray(hide, dtype=bool)
    A.f_sel = np.asarray(sel, dtype=bool)
    A.f_mat = np.asarray(mat, dtype=np.int32)
    A.v_sel = np.asarray([v.select for v in bm.verts], dtype=bool)
    A.sel_readable = readable
    A.uv_sel = np.asarray(usel, dtype=bool) if readable else np.zeros(len(lv), dtype=bool)
    A.co = A.tri_loops = A.tri_face = None
    if want_3d:
        A.co = np.asarray([c for v in bm.verts for c in v.co], dtype=np.float32).reshape(-1, 3)
        A.tri_loops, A.tri_face = _fan_triangles(A.f_start, A.f_total)
    A.fast = False
    return A


def _read_edit_arrays(obj, want_3d):
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    uvl = bm.loops.layers.uv.active
    if uvl is None:
        return None
    uv_name = uvl.name
    sync_valid = bool(getattr(bm, "uv_select_sync_valid", False))
    # Never walk the elements of the edit mesh here. The first Python wrapper of a BMesh
    # element makes Blender add a hidden per-element layer, and dropping the BMesh wrapper
    # removes it again: both reallocate every vertex / edge / corner / face data block,
    # UVs included. A running UV transform (or UV sculpt, edge slide, ...) keeps raw
    # pointers into those blocks, so doing that from a draw callback mid-drag makes the
    # tool write into freed memory and scrambles the UVs. BMesh.copy() reads the edit
    # mesh in C without wrappers; only the private copy is touched afterwards.
    bmc = bm.copy()
    del bm, uvl
    A = None
    try:
        if _State.fast_read is not False:
            scratch = None
            try:
                scratch = _scratch_mesh()
                if scratch is not None:
                    bmc.to_mesh(scratch)
                    A = _arrays_from_mesh(scratch, uv_name, want_3d)
                    _State.fast_read = True
            except Exception:
                A = None
                _State.fast_read = False
                _report("fast mesh read unavailable, using the slow path", traceback.format_exc())
            finally:
                if scratch is not None:
                    try:
                        scratch.clear_geometry()
                    except Exception:
                        pass
        if A is None:
            A = _arrays_from_bmesh(bmc, bmc.loops.layers.uv.get(uv_name), want_3d)
    finally:
        bmc.free()
    A.uv_name, A.sync_valid = uv_name, sync_valid
    return A


def _obj_gens(obj):
    """(geometry change count, change count with selection changes) of an object and its mesh,
    as counted from Blender's update notifications."""
    op, mp = obj.as_pointer(), obj.data.as_pointer()
    g, t = _State.id_gen, _State.touch
    return (g.get(op, 0), g.get(mp, 0)), (t.get(op, 0), t.get(mp, 0))


SEL_NONE, SEL_FACES, SEL_ALL = 0, 1, 2  # how much of the selection is looked at (_need_selection)


def _need_selection(sync, want_sel):
    """Which selection flags matter: all of them for Selected Shells Only; else, without UV
    Sync Selection, which faces are selected (the UV editor shows only those); else none."""
    if want_sel:
        return SEL_ALL
    return SEL_NONE if sync else SEL_FACES


def _selection_without_reading(obj, A, need_sel, sync):
    """Bring the selection flags of arrays read earlier up to date where Blender's counts of
    selected elements tell them: nothing selected, or (when only the faces matter) every
    shown face selected. Returns how much of the selection is then right (SEL_NONE: read the
    object). A click selecting something in one object deselects everything in the others:
    of a thousand objects announced as changed, all but one are told apart here."""
    me = obj.data
    try:
        if sync:
            if int(me.total_vert_sel) == 0:
                A.f_sel = np.zeros(A.f_sel.size, dtype=bool)
                A.v_sel = np.zeros(A.v_sel.size, dtype=bool)
                A.uv_sel = np.zeros(A.uv_sel.size, dtype=bool)
                return SEL_ALL
            return SEL_NONE
        n = int(me.total_face_sel)
        if n == 0:  # nothing of the object is shown: no other flag is looked at
            A.f_sel = np.zeros(A.f_sel.size, dtype=bool)
            return SEL_ALL
        if need_sel == SEL_FACES and n == A.f_hide.size - int(np.count_nonzero(A.f_hide)):
            A.f_sel = ~A.f_hide
            return SEL_FACES
    except Exception:
        pass
    return SEL_NONE


def _active_uv_name(obj):
    """Name of the edit mesh's active UV map (None: it has none). Touches no mesh element."""
    uvl = bmesh.from_edit_mesh(obj.data).loops.layers.uv.active
    return uvl.name if uvl is not None else None


def _object_arrays(obj, want_3d, reuse=True, need_sel=SEL_ALL, sync=None):
    """Arrays of an edit-mode object, shared by the UV editor and 3D views until the object
    changes (changes arrive as depsgraph updates, counted per object; `reuse=False` always
    reads again). `need_sel`: how much of the selection must be up to date, with `sync`, the
    UV Sync Selection state it is wanted for: switching that rewrites the UV selection
    (Blender 5) without an update being announced."""
    key = obj.as_pointer()
    geom, touch = _obj_gens(obj)
    hit = _State.arrays.get(key)
    if (reuse and hit is not None and hit[0] == _State.epoch and hit[1] == geom
            and (hit[3].co is not None or not want_3d) and hit[3].uv_name == _active_uv_name(obj)):
        if not need_sel:
            return hit[3]
        if hit[4] == sync:
            if hit[2] == touch and hit[5] >= need_sel:
                return hit[3]
            level = _selection_without_reading(obj, hit[3], need_sel, sync)  # (only the selection changed)
            if level >= need_sel:
                _State.arrays[key] = (hit[0], geom, touch, hit[3], sync, level)
                return hit[3]
    A = _read_edit_arrays(obj, want_3d)
    if A is None:
        _State.arrays.pop(key, None)
    else:
        _State.arrays[key] = (_State.epoch, geom, touch, A, sync, SEL_ALL)
    return A


def _prune_objects(objects):
    """Forget what was read from objects that are no longer in Edit Mode."""
    if len(_State.arrays) > len(objects) or len(_State.parts) > len(objects):
        alive = {o.as_pointer() for o in objects}
        for cache in (_State.arrays, _State.parts):
            for k in [k for k in cache if k not in alive]:
                del cache[k]


# ---------------------------------------------------------------------------
# UV shells from the arrays (vectorized)
# ---------------------------------------------------------------------------

class _Shells:
    """Faces visible in one view of a mesh, grouped into UV shells. Corners are kept in
    compact order: visible faces in mesh order, corners in face order."""
    __slots__ = ("fidx", "tot", "off", "lid", "lface", "nxt", "vid", "vx", "vy",
                 "area2", "face_shell", "nshell")


def _face_offsets(tot):
    off = np.zeros(tot.size, dtype=np.int64)
    if tot.size > 1:
        np.cumsum(tot[:-1], out=off[1:])
    return off


def _uv_vertex_ids(vert, u, v):
    """Ids of (mesh vertex, UV rounded to 1e-6), numbered by first appearance, and the
    position of each first appearance (same rule as the Python walk)."""
    n = vert.size
    qu = np.rint(u * UV_QUANT).astype(np.int64)
    qv = np.rint(v * UV_QUANT).astype(np.int64)
    order = np.lexsort((qv, qu, vert))  # stable: ties keep corner order
    sv, su, sw = vert[order], qu[order], qv[order]
    brk = np.ones(n, dtype=bool)
    if n > 1:
        brk[1:] = (sv[1:] != sv[:-1]) | (su[1:] != su[:-1]) | (sw[1:] != sw[:-1])
    first = order[brk]
    grp = np.empty(n, dtype=np.int64)
    grp[order] = np.cumsum(brk) - 1
    rank = np.argsort(first, kind='stable')
    new = np.empty(rank.size, dtype=np.int64)
    new[rank] = np.arange(rank.size)
    return new[grp], first[rank]


def _components(n, a, b):
    """Connected components of n nodes joined by edges a-b; label = smallest node index."""
    p = np.arange(n, dtype=np.int64)
    while a.size:
        pa, pb = p[a], p[b]
        m = pa != pb
        if not m.any():
            break
        a, b, pa, pb = a[m], b[m], pa[m], pb[m]
        np.minimum.at(p, np.maximum(pa, pb), np.minimum(pa, pb))  # hook a root to a smaller one
        while True:  # pointer jumping until every node points at its root
            q = p[p]
            if np.array_equal(q, p):
                break
            p = q
    return p


def _face_sums_in_order(values, off, tot):
    """values[off : off + tot] summed per face, corner by corner in order (bit-exact with
    a plain Python loop, unlike numpy's pairwise sums)."""
    out = np.zeros(tot.size)
    if not tot.size:
        return out
    tmin, tmax = int(tot.min()), int(tot.max())
    for k in range(tmin):
        out += values[off + k]
    if tmax > tmin:
        order = np.argsort(-tot, kind='stable')
        neg_sorted = -tot[order]
        for k in range(tmin, tmax):
            idx = order[:int(np.searchsorted(neg_sorted, -k, side='left'))]  # faces with > k corners
            out[idx] += values[off[idx] + k]
    return out


def _shells(A, vis):
    S = _Shells()
    fidx = np.flatnonzero(vis)
    nf = fidx.size
    tot = A.f_total[fidx].astype(np.int64)
    off = _face_offsets(tot)
    L = int(tot.sum()) if nf else 0
    lid = np.repeat(A.f_start[fidx].astype(np.int64) - off, tot) + np.arange(L, dtype=np.int64)
    lface = np.repeat(np.arange(nf, dtype=np.int64), tot)
    nxt = np.arange(1, L + 1, dtype=np.int64)
    if nf:
        nxt[off + tot - 1] = off
    uv = A.uv[lid].astype(np.float64)
    vid, first = _uv_vertex_ids(A.loop_vert[lid].astype(np.int64), uv[:, 0], uv[:, 1])
    vx, vy = uv[first, 0], uv[first, 1]
    xa, ya = vx[vid], vy[vid]
    area2 = _face_sums_in_order(xa * ya[nxt] - xa[nxt] * ya, off, tot)
    # faces sharing a UV vertex belong to one shell
    b = lface[first][vid]
    m = lface != b
    root = _components(nf, lface[m], b[m])
    is_root = root == np.arange(nf)  # roots are the first face of each shell, in face order
    label = np.cumsum(is_root) - 1
    S.fidx, S.tot, S.off, S.lid, S.lface, S.nxt = fidx, tot, off, lid, lface, nxt
    S.vid, S.vx, S.vy, S.area2 = vid, vx, vy, area2
    S.face_shell = label[root].astype(np.int64)
    S.nshell = int(np.count_nonzero(is_root))
    return S


def _chain_all(sh, a, b, sg, nshell, nv):
    """Border edges ordered into loops per shell, exactly like _chain_border: vectorized
    for shells whose border is a set of simple, consistently wound loops, _chain_border
    itself for the others (mixed winding, several loops meeting at a vertex)."""
    if not a.size:
        return sh, a, b, sg
    bad_v = (np.bincount(a, minlength=nv) != 1) | (np.bincount(b, minlength=nv) != 1)
    bad_shell = np.zeros(nshell, dtype=bool)
    bad_shell[sh[bad_v[a] | bad_v[b]]] = True
    pick = []
    se = np.flatnonzero(~bad_shell[sh])
    if se.size:
        ne = se.size
        ids = np.arange(ne, dtype=np.int64)
        start_of = np.full(nv, -1, dtype=np.int64)
        start_of[a[se]] = ids
        succ = start_of[b[se]]
        head, p, span = ids.copy(), succ.copy(), 1
        while span < ne:  # smallest edge index of each loop, by pointer doubling
            head = np.minimum(head, head[p])
            p = p[p]
            span *= 2
        s = succ.copy()
        pred = np.empty(ne, dtype=np.int64)
        pred[succ] = ids
        s[pred[head == ids]] = -1  # cut each loop just before its first edge
        d = (s >= 0).astype(np.int64)
        while True:  # list ranking: steps left to the end of the loop
            m = np.flatnonzero(s >= 0)
            if not m.size:
                break
            sm = s[m]
            d[m] += d[sm]
            s[m] = s[sm]
        g = se[np.lexsort((d[head] - d, head, sh[se]))]
        pick.append((sh[g], a[g], b[g], sg[g]))
    bad = np.flatnonzero(bad_shell[sh])
    if bad.size:
        bad = bad[np.argsort(sh[bad], kind='stable')]
        cut = np.flatnonzero(np.diff(sh[bad])) + 1
        for grp in np.split(bad, cut):
            chain = _chain_border(list(zip(a[grp].tolist(), b[grp].tolist(), sg[grp].tolist())))
            ca, cb, cs = zip(*chain)
            pick.append((np.full(len(chain), sh[grp[0]], dtype=np.int64),
                         np.asarray(ca, dtype=np.int64), np.asarray(cb, dtype=np.int64),
                         np.asarray(cs, dtype=np.float64)))
    sh2, a2, b2, sg2 = (np.concatenate(x) for x in zip(*pick))
    o = np.argsort(sh2, kind='stable')
    return sh2[o], a2[o], b2[o], sg2[o]


def _sel_kind(want_sel, sync, sync_valid):
    """Where "selected" comes from (Selected Shells Only)."""
    if not want_sel:
        return 0
    if _UV_SELECT_ON_LOOP and (not sync or sync_valid):
        return 1  # UV selection stored on the face corner (Blender 5.0+)
    if sync:
        return 2  # mesh selection
    return 3      # UV selection stored on the UV layer (Blender 3.x / 4.x)


def _corner_selection(A, sel_kind):
    """Whether each face corner counts as selected; None when every shell does (the selection
    isn't asked for or can't be read)."""
    if sel_kind == 0 or not A.sel_readable:
        return None
    return A.v_sel[A.loop_vert] if sel_kind == 2 else A.uv_sel


def _shell_selection(ls, off, face_shell, nshell):
    """Shells with a selected corner. `ls`: per corner of the shown faces, in their order."""
    face_sel = np.logical_or.reduceat(ls, off) if off.size else np.zeros(0, dtype=bool)
    return np.bincount(face_shell, weights=face_sel, minlength=nshell) > 0


def _gap_pieces(S, csel):
    """Ordered shell borders, per-shell selection and area, fan triangles of mirrored
    shells: the input of the gap measurement. `csel`: see _corner_selection."""
    vid, nxt, nv = S.vid, S.nxt, S.vx.size
    eb = vid[nxt]
    key = np.minimum(vid, eb) * max(nv, 1) + np.maximum(vid, eb)
    o = np.argsort(key, kind='stable')
    ks = key[o]
    brk = np.ones(ks.size, dtype=bool)
    if ks.size > 1:
        brk[1:] = ks[1:] != ks[:-1]
    run = np.cumsum(brk) - 1
    uses = np.empty(ks.size, dtype=np.int64)
    uses[o] = np.bincount(run)[run] if ks.size else run
    bk = np.flatnonzero(uses == 1)  # UV edges used by a single face
    b_face = S.lface[bk]
    fa = S.area2[b_face]
    b_sign = np.where(fa > 0.0, 1.0, np.where(fa < 0.0, -1.0, 0.0))
    ch_sh, ch_a, ch_b, ch_sg = _chain_all(S.face_shell[b_face], vid[bk], eb[bk], b_sign, S.nshell, nv)

    counts = np.bincount(ch_sh, minlength=S.nshell)
    shells_e = np.flatnonzero(counts)  # shells with a border, in shell order
    shell_area2 = np.bincount(S.face_shell, weights=S.area2, minlength=S.nshell)
    if csel is None:
        shell_sel = np.ones(S.nshell, dtype=bool)
    else:
        shell_sel = _shell_selection(csel[S.lid], S.off, S.face_shell, S.nshell)

    gid_of = np.full(S.nshell, -1, dtype=np.int64)
    gid_of[shells_e] = np.arange(shells_e.size)
    fsh = S.face_shell
    ff = np.flatnonzero((shell_area2[fsh] < 0.0) & (gid_of[fsh] >= 0))  # faces of mirrored shells
    nt = np.maximum(S.tot[ff] - 2, 0)
    tf = np.repeat(ff, nt)
    k = np.arange(int(nt.sum()), dtype=np.int64) - np.repeat(np.cumsum(nt) - nt, nt) + 1
    l0 = S.off[tf]
    v0, v1, v2 = vid[l0], vid[l0 + k], vid[l0 + k + 1]
    tris = np.stack((np.column_stack((S.vx[v0], S.vy[v0])), np.column_stack((S.vx[v1], S.vy[v1])),
                     np.column_stack((S.vx[v2], S.vy[v2]))), axis=1)
    return {
        "shells": shells_e,
        "seg_a": np.column_stack((S.vx[ch_a], S.vy[ch_a])),
        "seg_b": np.column_stack((S.vx[ch_b], S.vy[ch_b])),
        "sign": ch_sg.astype(np.float64),
        "count": counts[shells_e],
        "sel": shell_sel[shells_e],
        "area2": shell_area2[shells_e],
        "tris": tris.reshape(-1, 3, 2),
        "tri_gid": gid_of[fsh[tf]],
    }


def _cofactor(M3):
    """Cofactor matrix: maps area (normal) vectors the way M maps points."""
    c0, c1, c2 = M3[:, 0], M3[:, 1], M3[:, 2]
    return np.column_stack((np.cross(c1, c2), np.cross(c2, c0), np.cross(c0, c1)))


def _cofactors(M):
    """_cofactor of every matrix of M (k, 3, 3), with the same arithmetic."""
    def cross(a, b):
        return np.stack((a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1], a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                         a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), axis=1)
    c0, c1, c2 = M[:, :, 0], M[:, :, 1], M[:, :, 2]
    return np.stack((cross(c1, c2), cross(c2, c0), cross(c0, c1)), axis=2)


def _xform3(V, M, idx=None):
    """M applied to the rows of V: out[k] = M @ V[k]. M: one 3x3 matrix, or one per object
    with idx telling each row's object. Element by element (no BLAS), so one mesh alone and
    the same mesh in a batch give the same bits."""
    out = np.empty((V.shape[0], 3))
    for i in range(3):
        if idx is None:
            m0, m1, m2 = M[i, 0], M[i, 1], M[i, 2]
        else:
            m0, m1, m2 = M[idx, i, 0], M[idx, i, 1], M[idx, i, 2]
        out[:, i] = V[:, 0] * m0 + V[:, 1] * m1 + V[:, 2] * m2
    return out


def _world_face_areas(nvec, M3, face_obj=None):
    """World-space area of faces from their object-space area vectors. M3: the object's 3x3
    matrix, or one per object of a batch with face_obj, the object of each face."""
    wv = _xform3(nvec, _cofactor(M3) if face_obj is None else _cofactors(M3), face_obj)
    return np.sqrt(wv[:, 0] * wv[:, 0] + wv[:, 1] * wv[:, 1] + wv[:, 2] * wv[:, 2])


def _face_area_vectors(S, A):
    """Newell area vector of each face, in object space."""
    P = A.co[A.loop_vert[S.lid]].astype(np.float64)
    if not S.off.size:
        return np.empty((0, 3))
    Q = P[S.nxt]
    C = np.empty_like(P)  # the cross product P x Q, as np.cross computes it (without its overhead)
    np.subtract(P[:, 1] * Q[:, 2], P[:, 2] * Q[:, 1], out=C[:, 0])
    np.subtract(P[:, 2] * Q[:, 0], P[:, 0] * Q[:, 2], out=C[:, 1])
    np.subtract(P[:, 0] * Q[:, 1], P[:, 1] * Q[:, 0], out=C[:, 2])
    return np.add.reduceat(C, S.off, axis=0) * 0.5


def _material_keys(obj):
    """Material of each slot (full name, "" for an empty slot); [""] for an object without slots."""
    keys = [slot.material.name_full if slot.material is not None else "" for slot in obj.material_slots]
    return keys or [""]


def _material_sets(obj):
    """(material keys without repeats, set id of each slot): slots holding the same material,
    or several empty slots, are one material set."""
    keys = _material_keys(obj)
    uniq = list(dict.fromkeys(keys))
    return uniq, np.asarray([uniq.index(k) for k in keys], dtype=np.int64)


def _face_materials(A, lut):
    """Material id of every face; indices past the last slot use the last slot, as Blender draws them."""
    return lut[np.clip(A.f_mat, 0, lut.size - 1)]


def _dominant(fsh, fmat, weight, ns):
    """The material of each shell: the one covering most of its UV area (then more faces, lower id)."""
    out = np.zeros(ns, dtype=np.int64)
    if not fsh.size:
        return out
    nm = int(fmat.max()) + 1
    key = fsh * nm + fmat
    o = np.argsort(key, kind='stable')
    k = key[o]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    area = np.add.reduceat(weight[o], starts)
    count = np.diff(np.r_[starts, k.size])
    ush, um = k[starts] // nm, k[starts] % nm
    o2 = np.lexsort((um, -count, -area, ush))
    first = np.r_[True, ush[o2][1:] != ush[o2][:-1]]
    out[ush[o2][first]] = um[o2][first]
    return out


def _shell_orientation(S, A, M3, cx, cy, face_area, face_obj=None):
    """Where the scene's up axis (+Z) runs across each shell, as a unit direction in UV space.

    Each face's UV -> 3D mapping is fitted to its corners (least squares); the axis, projected
    onto the face, is taken back to UV space and weighted by how much of it lies along the face
    and by the face's area. Shells lying flat (+Z mostly along their normal) use +Y instead.
    Returns directions (NaN: no arrow) and the axis (0 none, 1 +Z, 2 +Y) per shell."""
    ns, fsh, lf = S.nshell, S.face_shell, S.lface
    du = S.vx[S.vid] - cx[lf]
    dv = S.vy[S.vid] - cy[lf]
    # world orientation and scale
    P = _xform3(A.co[A.loop_vert[S.lid]].astype(np.float64), M3, None if face_obj is None else face_obj[lf])
    dP = P - (np.add.reduceat(P, S.off, axis=0) / S.tot[:, None])[lf]
    suu = np.add.reduceat(du * du, S.off)
    suv = np.add.reduceat(du * dv, S.off)
    svv = np.add.reduceat(dv * dv, S.off)
    spu = np.add.reduceat(dP * du[:, None], S.off, axis=0)
    spv = np.add.reduceat(dP * dv[:, None], S.off, axis=0)
    orient = np.full((ns, 2), np.nan)
    axis = np.zeros(ns, dtype=np.int8)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        det = suu * svv - suv * suv
        ok = det > 1e-12 * (suu + svv) ** 2
        Ju = (spu * svv[:, None] - spv * suv[:, None]) / det[:, None]  # dP/du
        Jv = (spv * suu[:, None] - spu * suv[:, None]) / det[:, None]  # dP/dv
        g11 = (Ju * Ju).sum(axis=1)
        g12 = (Ju * Jv).sum(axis=1)
        g22 = (Jv * Jv).sum(axis=1)
        gdet = g11 * g22 - g12 * g12
        ok &= gdet > 1e-12 * (g11 + g22) ** 2
        total = np.bincount(fsh, weights=face_area, minlength=ns)
        for code, comp in ((1, 2), (2, 1)):  # +Z, then +Y for what is left
            b1, b2 = Ju[:, comp], Jv[:, comp]
            wu = (g22 * b1 - g12 * b2) / gdet  # UV step whose 3D image is the axis on the face
            wv = (g11 * b2 - g12 * b1) / gdet
            along = b1 * wu + b2 * wv          # squared length of the axis on the face, 0..1
            wn = np.hypot(wu, wv)
            good = ok & np.isfinite(wn) & (wn > 0.0) & (along > 0.0)
            wt = np.where(good, face_area * np.sqrt(np.clip(along, 0.0, 1.0)) / np.where(good, wn, 1.0), 0.0)
            vx = np.bincount(fsh, weights=np.where(good, wu, 0.0) * wt, minlength=ns)
            vy = np.bincount(fsh, weights=np.where(good, wv, 0.0) * wt, minlength=ns)
            mag = np.hypot(vx, vy)
            pick = (axis == 0) & (mag > 0.0) & (mag >= ORIENT_MIN * total)
            orient[pick, 0] = vx[pick] / mag[pick]
            orient[pick, 1] = vy[pick] / mag[pick]
            axis[pick] = code
    return orient, axis


def _shell_pieces(S, A, M3, fmat, want_3d, want_orient=False, face_obj=None):
    """Per shell: label point, UV area and material; with 3D data also the world area (texel
    density), the triangles for the colored fill and, if wanted, the up direction in UV space.
    M3: the object's 3x3 matrix, or one per object of a batch with face_obj, the object of
    each shown face."""
    uv_abs = np.abs(S.area2) * 0.5
    fsh, ns = S.face_shell, S.nshell
    # label point: centroid of the face nearest to the shell's area-weighted centroid,
    # which is always on the shell (also for rings and concave shells)
    xa, ya = S.vx[S.vid], S.vy[S.vid]
    cx = np.add.reduceat(xa, S.off) / S.tot
    cy = np.add.reduceat(ya, S.off) / S.tot
    w = np.bincount(fsh, weights=uv_abs, minlength=ns)
    n = np.bincount(fsh, minlength=ns)
    wx = np.bincount(fsh, weights=uv_abs * cx, minlength=ns)
    wy = np.bincount(fsh, weights=uv_abs * cy, minlength=ns)
    mx = np.bincount(fsh, weights=cx, minlength=ns) / n
    my = np.bincount(fsh, weights=cy, minlength=ns) / n
    with np.errstate(divide='ignore', invalid='ignore'):
        sx = np.where(w > 0.0, wx / w, mx)
        sy = np.where(w > 0.0, wy / w, my)
    d2 = (cx - sx[fsh]) ** 2 + (cy - sy[fsh]) ** 2
    o = np.lexsort((d2, fsh))
    nearest = o[np.concatenate(([True], fsh[o][1:] != fsh[o][:-1]))]
    out = {
        "nshell": ns,
        "uv_area": w,
        "anchor": np.column_stack((cx[nearest], cy[nearest])),
        "mat": _dominant(fsh, fmat, uv_abs, ns),
    }
    if want_3d:
        world = _world_face_areas(_face_area_vectors(S, A), M3, face_obj)
        out["world_area"] = np.bincount(fsh, weights=world, minlength=ns)
        if want_orient:
            out["orient"], out["axis"] = _shell_orientation(S, A, M3, cx, cy, world, face_obj)
        cmap = np.full(A.f_total.size, -1, dtype=np.int64)
        cmap[S.fidx] = np.arange(S.fidx.size)
        tc = cmap[A.tri_face]
        keep = tc >= 0
        out["tri_uv"] = A.uv[A.tri_loops[keep]].astype(np.float32)
        out["tri_shell"] = fsh[tc[keep]]
    return out


class _Geometry:
    __slots__ = ("seg_a", "seg_b", "seg_sign", "seg_shell", "shell_off", "shell_sel",
                 "bmin", "bmax", "pairs", "nshells", "nfaces",
                 "shell_area", "flipped", "flip_tris", "flip_tri_shell",
                 "tile", "tile_cross", "tile_cross_pts",
                 "td_nshells", "td_uv_area", "td_world_area", "td_anchor", "td_tri_uv", "td_tri_shell",
                 # every shown shell (the arrays above cover the shells with a border, "gap shells")
                 "sh_n", "sh_obj", "sh_mat", "sh_anchor", "sh_orient", "sh_axis", "sh_gid",
                 "gap_shell", "shell_mat", "mat_keys", "mat_total", "mat_vis", "obj_names", "obj_scale",
                 # the objects' own parts, for the caches and the operators
                 "obj_parts", "obj_src", "obj_uvsig", "sh_base", "src_serials")

    def __init__(self):
        self.seg_a = _EMPTY2
        self.seg_b = _EMPTY2
        self.seg_sign = _EMPTY_F
        self.seg_shell = _EMPTY_I
        self.shell_off = np.zeros(1, dtype=np.int64)
        self.shell_sel = np.empty(0, dtype=bool)
        self.bmin = _EMPTY2
        self.bmax = _EMPTY2
        self.pairs = None          # overlapping pairs (_Pairs), when found for exactly these shells
        self.nshells = 0
        self.nfaces = 0
        self.shell_area = _EMPTY_F
        self.flipped = np.empty(0, dtype=bool)
        self.flip_tris = np.empty((0, 3, 2))
        self.flip_tri_shell = _EMPTY_I
        self.tile = np.empty((0, 2), dtype=np.int64)
        self.tile_cross = np.empty(0, dtype=bool)
        self.tile_cross_pts = {}
        self.td_nshells = 0
        self.td_uv_area = _EMPTY_F
        self.td_world_area = _EMPTY_F
        self.td_anchor = _EMPTY2
        self.td_tri_uv = np.empty((0, 3, 2), dtype=np.float32)
        self.td_tri_shell = _EMPTY_I
        self.sh_n = 0
        self.sh_obj = _EMPTY_I
        self.sh_mat = _EMPTY_I
        self.sh_anchor = _EMPTY2
        self.sh_orient = _EMPTY2
        self.sh_axis = np.empty(0, dtype=np.int8)
        self.sh_gid = _EMPTY_I
        self.gap_shell = _EMPTY_I
        self.shell_mat = _EMPTY_I
        self.mat_keys = []
        self.mat_total = _EMPTY_I
        self.mat_vis = _EMPTY_I
        self.obj_names = []
        self.obj_scale = []
        self.obj_parts = []        # per object with shown faces: its _Parts
        self.obj_src = []          # ... and its index in the objects the geometry was read from
        self.obj_uvsig = []        # ... and what its shells depend on (measurement caches)
        self.sh_base = []          # ... and the all-shell index of its first shell
        self.src_serials = ()      # serial numbers of all the parts assembled (also without faces)


class _Parts:
    """One object's share of a _Geometry: its shells, their borders and per-shell data, kept
    while the object doesn't change. `uv_sig` changes with the UVs, the faces shown and the
    materials - what gaps, overlaps and stacks depend on -, `serial` with every extraction.
    Material ids are local to the object (`uniq`: its material keys, `lut`: set of each slot).
    fidx / face_shell / tot: mesh face, shell and corner count of each shown face; lid: the
    mesh corner of each of their corners."""
    __slots__ = ("serial", "uv_sig", "A", "vis", "csel", "fm", "flags", "m3", "name", "scale", "fast",
                 "uniq", "lut", "total", "shown", "nfaces", "gap", "sh", "fidx", "face_shell", "lid", "tot")


_SERIAL = [0]


def _same(a, b):
    return a is b or (a.shape == b.shape and bool(np.array_equal(a, b)))


def _same_uv_arrays(P, A, vis):
    """Whether the arrays A (faces shown: vis) hold the same UVs, faces and material slots as
    the ones P was made from."""
    B = P.A
    return (_same(P.vis, vis) and _same(B.f_total, A.f_total) and _same(B.f_start, A.f_start)
            and _same(B.loop_vert, A.loop_vert) and _same(B.uv, A.uv) and _same(B.f_mat, A.f_mat))


def _parts_selection(P, csel):
    """Whether each shell with a border of a kept object is selected now."""
    if csel is None:
        return np.ones(P.gap["shells"].size, dtype=bool)
    sel = _shell_selection(csel[P.lid], _face_offsets(P.tot), P.face_shell, P.sh["nshell"])
    return sel[P.gap["shells"]]


def _extract_batch(todo, want_td, want_orient):
    """Fill in the _Parts of `todo` (A, vis, csel, fm, uniq and m3 set). Several objects are
    worked on in one pass: their arrays are laid one after another as one mesh (corners, faces
    and vertices renumbered), the shells are found once and the results are cut back into one
    _Parts per object - a thousand small objects cost about as much as one mesh with their
    faces, with the same results as one by one."""
    n = len(todo)
    if n == 1:
        P = todo[0]
        X, vis, fm, csel, M3, fobj = P.A, P.vis, P.fm, P.csel, P.m3, None
        foff = loff = np.zeros(2, dtype=np.int64)
        kbase = np.array([0, len(P.uniq)], dtype=np.int64)
    else:
        arrs = [P.A for P in todo]
        nf = np.array([A.f_total.size for A in arrs], dtype=np.int64)
        foff = np.concatenate(([0], np.cumsum(nf)))
        loff = np.concatenate(([0], np.cumsum([A.loop_vert.size for A in arrs])))
        voff = np.concatenate(([0], np.cumsum([A.v_sel.size for A in arrs])))
        X = _Arrays()
        X.f_total = np.concatenate([A.f_total for A in arrs])
        X.f_start = np.concatenate([A.f_start.astype(np.int64) + int(o) for A, o in zip(arrs, loff[:-1])])
        X.loop_vert = np.concatenate([A.loop_vert.astype(np.int64) + int(o) for A, o in zip(arrs, voff[:-1])])
        X.uv = np.concatenate([A.uv for A in arrs]).reshape(-1, 2)
        X.co = X.tri_loops = X.tri_face = None
        if want_td:
            X.co = np.concatenate([A.co for A in arrs]).reshape(-1, 3)
            X.tri_loops = np.concatenate([A.tri_loops.astype(np.int64) + int(o)
                                          for A, o in zip(arrs, loff[:-1])]).reshape(-1, 3)
            X.tri_face = np.concatenate([A.tri_face.astype(np.int64) + int(o) for A, o in zip(arrs, foff[:-1])])
        vis = np.concatenate([P.vis for P in todo])
        kbase = np.concatenate(([0], np.cumsum([len(P.uniq) for P in todo]))).astype(np.int64)
        fm = np.concatenate([P.fm + int(b) for P, b in zip(todo, kbase[:-1])])
        csel = None
        if any(P.csel is not None for P in todo):
            csel = np.concatenate([P.csel if P.csel is not None else np.ones(A.loop_vert.size, dtype=bool)
                                   for P, A in zip(todo, arrs)])
        fobj = np.repeat(np.arange(n, dtype=np.int64), nf)
        M3 = np.stack([P.m3 for P in todo]) if want_td else None

    total = np.bincount(fm, minlength=int(kbase[-1]))
    shown = np.bincount(fm[vis], minlength=int(kbase[-1]))
    for i, P in enumerate(todo):
        P.total = total[kbase[i]:kbase[i + 1]].astype(np.int64)
        P.shown = shown[kbase[i]:kbase[i + 1]].astype(np.int64)
        P.nfaces = 0
        P.gap = P.sh = None
        P.fidx = P.face_shell = P.lid = P.tot = _EMPTY_I32
    S = _shells(X, vis)
    if not S.fidx.size:
        return
    fo = fobj[S.fidx] if fobj is not None else np.zeros(S.fidx.size, dtype=np.int64)  # object of each shown face
    gap = _gap_pieces(S, csel)
    sh = _shell_pieces(S, X, M3, fm[S.fidx], want_td, want_td and want_orient, None if fobj is None else fo)
    every = np.arange(n + 1)
    f_off = np.searchsorted(fo, every)                       # shown faces of each object
    c_off = np.concatenate((S.off, [S.vid.size]))[f_off]     # their corners
    sobj = np.zeros(S.nshell, dtype=np.int64)
    sobj[S.face_shell] = fo                                  # shells: numbered by their first face
    s_off = np.searchsorted(sobj, every)
    g_off = np.searchsorted(gap["shells"], s_off)            # shells with a border
    seg_off = np.concatenate(([0], np.cumsum(gap["count"])))[g_off]
    none = np.zeros(n + 1, dtype=np.int64)
    t_off = np.searchsorted(sobj[gap["shells"]][gap["tri_gid"]], every) if gap["tri_gid"].size else none
    q_off = np.searchsorted(sobj[sh["tri_shell"]], every) if want_td and sh["tri_shell"].size else none
    for i, P in enumerate(todo):
        f0, f1 = int(f_off[i]), int(f_off[i + 1])
        if f1 == f0:
            continue
        c0, c1, s0, s1 = int(c_off[i]), int(c_off[i + 1]), int(s_off[i]), int(s_off[i + 1])
        g0, g1, e0, e1 = int(g_off[i]), int(g_off[i + 1]), int(seg_off[i]), int(seg_off[i + 1])
        t0, t1, q0, q1 = int(t_off[i]), int(t_off[i + 1]), int(q_off[i]), int(q_off[i + 1])
        P.nfaces = f1 - f0
        P.gap = {"shells": gap["shells"][g0:g1] - s0, "seg_a": gap["seg_a"][e0:e1], "seg_b": gap["seg_b"][e0:e1],
                 "sign": gap["sign"][e0:e1], "count": gap["count"][g0:g1], "sel": gap["sel"][g0:g1],
                 "area2": gap["area2"][g0:g1], "tris": gap["tris"][t0:t1], "tri_gid": gap["tri_gid"][t0:t1] - g0}
        P.sh = {"nshell": s1 - s0, "uv_area": sh["uv_area"][s0:s1], "anchor": sh["anchor"][s0:s1],
                "mat": sh["mat"][s0:s1] - int(kbase[i])}
        if want_td:
            P.sh["world_area"] = sh["world_area"][s0:s1]
            P.sh["tri_uv"] = sh["tri_uv"][q0:q1]
            P.sh["tri_shell"] = sh["tri_shell"][q0:q1] - s0
            if want_orient:
                P.sh["orient"], P.sh["axis"] = sh["orient"][s0:s1], sh["axis"][s0:s1]
        P.fidx = (S.fidx[f0:f1] - int(foff[i])).astype(np.int32)
        P.face_shell = (S.face_shell[f0:f1] - s0).astype(np.int32)
        P.lid = (S.lid[c0:c1] - int(loff[i])).astype(np.int32)
        P.tot = S.tot[f0:f1].astype(np.int32)


def _object_parts(objects, sync, want_sel, want_td=False, reuse=False, want_orient=False, sigs=None):
    """One _Parts per edit-mode object with a UV map, in order: from the cache where the
    object is unchanged, found again otherwise. Unchanged means: no update was announced for
    it and its entry of the signature (`sigs`, see _geometry_signature) is the same; or, when
    its arrays were read again, they compare equal. Returns (parts, index into `objects` of
    each, whether the selected shells of a kept object changed, whether the name or the scale
    of one did)."""
    st = _State
    flags = (bool(want_td), bool(want_td and want_orient))
    need_sel = _need_selection(sync, want_sel)
    out, src, todo = [], [], []
    sel_changed = meta_changed = False
    for oi, obj in enumerate(objects):
        me = obj.data
        if not me.is_editmode:
            continue
        key = obj.as_pointer()
        fast = None
        if sigs is not None:
            # what the object's parts are of (the change counts as they are before it is read:
            # also parts read anew, for an operator, are known by them afterwards)
            geom, touch = _obj_gens(obj)
            fast = (st.epoch, flags, bool(sync), bool(want_sel), geom, touch if need_sel else None, sigs[oi])
            old = st.parts.get(key) if reuse else None
            if old is not None and old.fast == fast:
                out.append(old)
                src.append(oi)
                continue
        A = _object_arrays(obj, want_td, reuse, need_sel, bool(sync))
        if A is None:
            continue
        uniq, lut = _material_sets(obj)
        m3 = np.array(obj.matrix_world.to_3x3(), dtype=np.float64) if want_td else None
        vis = ~A.f_hide if sync else (~A.f_hide & A.f_sel)  # faces shown in the UV editor
        csel = _corner_selection(A, _sel_kind(want_sel, sync, A.sync_valid))
        name, scale = obj.name, tuple(obj.scale)
        old = st.parts.get(key)
        same_uv = (old is not None and old.uniq == uniq and _same(old.lut, lut)
                   and (_same(old.vis, vis) if old.A is A else _same_uv_arrays(old, A, vis)))
        if same_uv and old.flags == flags and (not want_td or (
                _same(old.m3, m3) and (old.A is A or _same(old.A.co, A.co)))):
            # the same shells: only the selection, the name or the scale can differ
            P = old
            if P.gap is not None:
                sel = _parts_selection(P, csel)
                if not np.array_equal(sel, P.gap["sel"]):
                    P.gap["sel"] = sel
                    sel_changed = True
            if P.name != name or P.scale != scale:
                P.name, P.scale = name, scale
                meta_changed = True
            P.A, P.vis = A, vis
        else:
            P = _Parts()
            P.uv_sig = old.uv_sig if same_uv else None
            P.A, P.vis, P.csel, P.flags, P.m3 = A, vis, csel, flags, m3
            P.name, P.scale, P.uniq, P.lut = name, scale, uniq, lut
            P.fm = _face_materials(A, lut)
            todo.append(P)
            st.parts[key] = P
        P.fast = fast
        out.append(P)
        src.append(oi)

    batch, faces = [], 0
    for P in todo:  # small meshes together, large ones on their own
        nf = int(P.A.f_total.size)
        if nf >= BATCH_MESH_FACES:
            _extract_batch([P], want_td, want_orient)
            continue
        if batch and faces + nf > BATCH_FACES:
            _extract_batch(batch, want_td, want_orient)
            batch, faces = [], 0
        batch.append(P)
        faces += nf
    if batch:
        _extract_batch(batch, want_td, want_orient)
    for P in todo:
        _SERIAL[0] += 1
        P.serial = _SERIAL[0]
        if P.uv_sig is None:
            P.uv_sig = ("uv", P.serial)
        P.csel = P.fm = None
    st.geo_stats = (len(todo), len(out))
    return out, src, sel_changed, meta_changed


def _assemble(parts, src=None, want_td=False, want_orient=False):
    """A _Geometry from the objects' parts, in order. Material ids are global: the order in
    which the keys first appear."""
    geo = _Geometry()
    gaps, shs = [], []
    mat_id, mat_total, mat_vis = {}, [], []
    for k, P in enumerate(parts):
        ids = []
        for key in P.uniq:
            i = mat_id.get(key)
            if i is None:
                i = mat_id[key] = len(mat_total)
                mat_total.append(0)
                mat_vis.append(0)
            ids.append(i)
        for i, t, s in zip(ids, P.total.tolist(), P.shown.tolist()):
            mat_total[i] += t
            mat_vis[i] += s
        if P.gap is None:
            continue
        geo.nfaces += P.nfaces
        gaps.append(P.gap)
        p = dict(P.sh)
        p["mat"] = np.asarray(ids, dtype=np.int64)[p["mat"]]
        p["obj"] = len(geo.obj_names)
        shs.append(p)
        geo.obj_names.append(P.name)
        geo.obj_scale.append(P.scale)
        geo.obj_parts.append(P)
        geo.obj_src.append(src[k] if src is not None else k)
        geo.obj_uvsig.append(P.uv_sig)

    geo.mat_keys = list(mat_id)
    geo.mat_total = np.asarray(mat_total, dtype=np.int64)
    geo.mat_vis = np.asarray(mat_vis, dtype=np.int64)
    geo.src_serials = tuple(P.serial for P in parts)
    if not shs:
        return geo
    base = np.cumsum([0] + [p["nshell"] for p in shs])
    n = geo.sh_n = int(base[-1])
    geo.sh_base = base[:-1].tolist()
    geo.sh_obj = np.repeat(np.asarray([p["obj"] for p in shs], dtype=np.int64), np.diff(base))
    geo.sh_mat = np.concatenate([p["mat"] for p in shs])
    geo.sh_anchor = np.concatenate([p["anchor"] for p in shs])
    geo.sh_gid = np.full(n, -1, dtype=np.int64)
    if want_td:
        geo.td_nshells = n
        geo.td_uv_area = np.concatenate([p["uv_area"] for p in shs])
        geo.td_world_area = np.concatenate([p["world_area"] for p in shs])
        geo.td_anchor = geo.sh_anchor
        geo.td_tri_uv = np.concatenate([p["tri_uv"] for p in shs])
        geo.td_tri_shell = np.concatenate([p["tri_shell"] + b for p, b in zip(shs, base[:-1])])
    if want_td and want_orient:
        geo.sh_orient = np.concatenate([p["orient"] for p in shs])
        geo.sh_axis = np.concatenate([p["axis"] for p in shs])
    else:
        geo.sh_orient = np.full((n, 2), np.nan)
        geo.sh_axis = np.zeros(n, dtype=np.int8)

    S = int(sum(g["count"].size for g in gaps))
    if S == 0:
        return geo
    geo.gap_shell = np.concatenate([g["shells"] + b for g, b in zip(gaps, base[:-1])]).astype(np.int64)
    geo.sh_gid[geo.gap_shell] = np.arange(S)
    geo.shell_mat = geo.sh_mat[geo.gap_shell]
    gbase = np.cumsum([0] + [g["count"].size for g in gaps])
    geo.seg_a = np.concatenate([g["seg_a"] for g in gaps])
    geo.seg_b = np.concatenate([g["seg_b"] for g in gaps])
    geo.seg_sign = np.concatenate([g["sign"] for g in gaps])
    geo.shell_off = np.concatenate(([0], np.cumsum(np.concatenate([g["count"] for g in gaps])))).astype(np.int64)
    geo.shell_sel = np.concatenate([g["sel"] for g in gaps]).astype(bool)
    geo.seg_shell = np.repeat(np.arange(S, dtype=np.int64), np.diff(geo.shell_off))
    lo = np.minimum(geo.seg_a, geo.seg_b)
    hi = np.maximum(geo.seg_a, geo.seg_b)
    geo.bmin = np.minimum.reduceat(lo, geo.shell_off[:-1], axis=0)
    geo.bmax = np.maximum.reduceat(hi, geo.shell_off[:-1], axis=0)
    geo.nshells = S
    geo.shell_area = np.concatenate([g["area2"] for g in gaps]) * 0.5
    geo.flipped = geo.shell_area < 0.0
    geo.flip_tris = np.concatenate([g["tris"] for g in gaps]).astype(np.float64).reshape(-1, 3, 2)
    geo.flip_tri_shell = np.concatenate([g["tri_gid"] + b for g, b in zip(gaps, gbase[:-1])]).astype(np.int64)
    geo.tile = np.floor((geo.bmin + geo.bmax) * 0.5).astype(np.int64)
    geo.tile_cross = (np.floor(geo.bmin + TILE_EPS) < np.floor(geo.bmax - TILE_EPS)).any(axis=1)
    geo.tile_cross_pts = _tile_crossings(geo)
    return geo


def _extract(objects, sync, want_sel, want_td=False, reuse=False, want_orient=False):
    """Shells, their borders and per-shell data of the faces shown in the UV editor. Objects
    that didn't change since they were last read keep their shells (`reuse` False: every
    object is read again, and compared with what was read before)."""
    parts, src, _sel, _meta = _object_parts(objects, sync, want_sel, want_td, reuse, want_orient)
    return _assemble(parts, src, want_td, want_orient)


def _tile_crossings(geo):
    """{shell: points} where the border of a shell spanning a tile line crosses that line (UV)."""
    out = {}
    for s in np.flatnonzero(geo.tile_cross).tolist():
        o0, o1 = int(geo.shell_off[s]), int(geo.shell_off[s + 1])
        a, b = geo.seg_a[o0:o1], geo.seg_b[o0:o1]
        found = []
        for axis in (0, 1):
            k0 = int(np.floor(geo.bmin[s, axis] + TILE_EPS)) + 1
            k1 = int(np.floor(geo.bmax[s, axis] - TILE_EPS))
            for k in range(k0, k1 + 1):
                da, db = a[:, axis] - k, b[:, axis] - k
                hit = (da < 0.0) != (db < 0.0)
                if hit.any():
                    t = da[hit] / (da[hit] - db[hit])
                    found.append(a[hit] + (b[hit] - a[hit]) * t[:, None])
        if found:
            pts = np.concatenate(found)
            if len(pts) > MAX_CROSS_MARKS:
                pts = pts[np.linspace(0, len(pts) - 1, MAX_CROSS_MARKS).astype(np.int64)]
            out[s] = pts
    return out


# ---------------------------------------------------------------------------
# Vectorized spatial helpers (uniform grid + sorted cell keys)
# ---------------------------------------------------------------------------

def _ranges(starts, ends):
    """Concatenate index ranges [starts[i], ends[i])."""
    starts = np.asarray(starts, dtype=np.int64)
    lens = np.asarray(ends, dtype=np.int64) - starts
    total = int(lens.sum())
    if total <= 0:
        return _EMPTY_I
    return np.arange(total, dtype=np.int64) + np.repeat(starts - (np.cumsum(lens) - lens), lens)


def _cell_rows(x0, y0, x1, y1, cell):
    gx0 = np.floor(x0 / cell)
    gy0 = np.floor(y0 / cell)
    return float(((np.floor(x1 / cell) - gx0 + 1) * (np.floor(y1 / cell) - gy0 + 1)).sum())


def _box_cells(x0, y0, x1, y1, cell):
    """(box index, cell key) rows for every grid cell each axis-aligned box touches."""
    gx0 = np.floor(x0 / cell).astype(np.int64)
    gy0 = np.floor(y0 / cell).astype(np.int64)
    wx = np.floor(x1 / cell).astype(np.int64) - gx0 + 1
    wy = np.floor(y1 / cell).astype(np.int64) - gy0 + 1
    cnt = wx * wy
    box = np.repeat(np.arange(gx0.size, dtype=np.int64), cnt)
    local = np.arange(box.size, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    w = wx[box]
    key = (gx0[box] + local % w + _KEY_OFF) * _KEY_MUL + (gy0[box] + local // w + _KEY_OFF)
    return box, key


class _Grid:
    """Items with bounding boxes, bucketed by the grid cells they touch."""
    __slots__ = ("cell", "keys", "items")

    def __init__(self, cell, items, lo, hi):
        box, key = _box_cells(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1], cell)
        order = np.argsort(key, kind='stable')
        self.cell = cell
        self.keys = key[order]
        self.items = items[box[order]]

    def query(self, lo, hi):
        """Rows (query index, first position, count) into `items`, sorted by query index."""
        qbox, qkey = _box_cells(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1], self.cell)
        first = np.searchsorted(self.keys, qkey, side='left')
        count = np.searchsorted(self.keys, qkey, side='right') - first
        return qbox, first, count


def _expand(qbox, first, count):
    """Candidate pairs (query index, position in grid items), grouped by query."""
    total = int(count.sum())
    q = np.repeat(qbox, count)
    pos = np.arange(total, dtype=np.int64) + np.repeat(first - (np.cumsum(count) - count), count)
    return q, pos


def _query_chunks(qbox, count, nq, limit=CHUNK):
    """(row0, row1) ranges covering whole queries with roughly `limit` pairs each."""
    per_q = np.bincount(qbox, weights=count, minlength=nq)
    csum = np.cumsum(per_q)
    q0 = 0
    while q0 < nq:
        base = csum[q0 - 1] if q0 else 0.0
        q1 = min(nq, max(q0 + 1, int(np.searchsorted(csum, base + limit, side='right'))))
        yield (int(np.searchsorted(qbox, q0, side='left')),
               int(np.searchsorted(qbox, q1, side='left')))
        q0 = q1


def _segment_pieces(A, B, L, cell):
    """Split segments into pieces no longer than `cell`: (segment index, piece lo, piece hi)."""
    D = B - A
    k = np.maximum(1, np.ceil(L / cell)).astype(np.int64)
    seg = np.repeat(np.arange(len(A), dtype=np.int64), k)
    j = (np.arange(seg.size, dtype=np.int64) - np.repeat(np.cumsum(k) - k, k)).astype(np.float64)
    kk = k[seg].astype(np.float64)
    p0 = A[seg] + D[seg] * (j / kk)[:, None]
    p1 = A[seg] + D[seg] * ((j + 1.0) / kk)[:, None]
    return seg, np.minimum(p0, p1), np.maximum(p0, p1)


def _shell_grid(bmin, bmax):
    ext = (bmax - bmin).max(axis=1)
    cell = max(float(np.median(ext)), float(ext.max()) * 1e-3, 1e-12)
    while _cell_rows(bmin[:, 0], bmin[:, 1], bmax[:, 0], bmax[:, 1], cell) > MAX_GRID_ROWS:
        cell *= 2.0
    return _Grid(cell, np.arange(len(bmin), dtype=np.int64), bmin, bmax)


def _slopes(A, B):
    D = B - A
    return np.divide(D[:, 0], D[:, 1], out=np.zeros(len(D)), where=D[:, 1] != 0.0)


class _Bands:
    """Border segments bucketed by shell and horizontal band: a ray from a point only crosses
    segments spanning the point's height, so only its band's segments need testing."""
    __slots__ = ("h", "y0", "nb", "keys", "seg")

    def __init__(self, A, B, seg_shell):
        lo = np.minimum(A[:, 1], B[:, 1])
        hi = np.maximum(A[:, 1], B[:, 1])
        self.y0 = float(lo.min()) if lo.size else 0.0
        span = float(hi.max()) - self.y0 if hi.size else 0.0
        # bands as tall as the mean segment: each segment lands in about two bands, a shell's
        # band holds a handful of its segments
        h = max(float(np.mean(hi - lo)) if lo.size else 0.0, span / (1 << 20), 1e-12)
        while lo.size and float(np.sum(np.floor((hi - self.y0) / h) - np.floor((lo - self.y0) / h) + 1.0)) > MAX_GRID_ROWS:
            h *= 2.0
        self.h = h
        self.nb = int(np.floor(span / h)) + 2
        b0 = np.floor((lo - self.y0) / h).astype(np.int64)
        cnt = np.floor((hi - self.y0) / h).astype(np.int64) - b0 + 1
        seg = np.repeat(np.arange(lo.size, dtype=np.int64), cnt)
        band = b0[seg] + (np.arange(seg.size, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        key = seg_shell[seg] * self.nb + band
        o = np.argsort(key, kind='stable')  # (segments of a band in border order)
        self.keys, self.seg = key[o], seg[o]

    def query(self, shells, py):
        """(first position, count) into seg of the segments of shells[i] in the band of py[i]."""
        band = np.clip(np.floor((py - self.y0) / self.h).astype(np.int64), 0, self.nb - 1)
        key = shells * self.nb + band
        first = np.searchsorted(self.keys, key, side='left')
        return first, np.searchsorted(self.keys, key, side='right') - first


def _probe_hits(pts, own, sgrid, bmin, bmax, A, B, slope, off, bands=None):
    """Every (point index, shell) where the point lies inside a shell other than own[point].

    Candidates come from shell bounding boxes; the even-odd rule over the
    candidate shell's border decides (holes count as outside) - over the segments in the
    point's band (_Bands), the only ones a horizontal ray can cross.
    Results are sorted by point index.
    """
    if len(pts) == 0:
        return _EMPTY_I, _EMPTY_I
    qbox, first, count = sgrid.query(pts, pts)  # a point touches exactly one cell
    q, pos = _expand(qbox, first, count)
    c = sgrid.items[pos]
    px, py = pts[q, 0], pts[q, 1]
    keep = ((c != own[q]) & (px >= bmin[c, 0]) & (px <= bmax[c, 0]) &
            (py >= bmin[c, 1]) & (py <= bmax[c, 1]))
    q, c = q[keep], c[keep]
    if not q.size:
        return _EMPTY_I, _EMPTY_I
    if bands is None:
        seg_shell = np.repeat(np.arange(off.size - 1, dtype=np.int64), np.diff(off))
        bands = _Bands(A, B, seg_shell)
    bfirst, nseg = bands.query(c, pts[q, 1])
    csum = np.cumsum(nseg)
    hp, hs = [], []
    i0 = 0
    while i0 < q.size:
        base = csum[i0 - 1] if i0 else 0
        i1 = min(q.size, max(i0 + 1, int(np.searchsorted(csum, base + CHUNK, side='right'))))
        qq, cc, ns = q[i0:i1], c[i0:i1], nseg[i0:i1]
        has = ns > 0
        qq, cc, ns, bf = qq[has], cc[has], ns[has], bfirst[i0:i1][has]
        if qq.size:
            seg = bands.seg[_ranges(bf, bf + ns)]
            row = np.repeat(qq, ns)
            ppx, ppy = pts[row, 0], pts[row, 1]
            ay, by = A[seg, 1], B[seg, 1]
            cross = ((ay > ppy) != (by > ppy)) & (ppx < A[seg, 0] + (ppy - ay) * slope[seg])
            odd = (np.add.reduceat(cross, np.cumsum(ns) - ns, dtype=np.int32) & 1).astype(bool)
            hp.append(qq[odd])
            hs.append(cc[odd])
        i0 = i1
    if not hp:
        return _EMPTY_I, _EMPTY_I
    return np.concatenate(hp), np.concatenate(hs)


# ---------------------------------------------------------------------------
# Overlapping shell pairs (UV space)
# ---------------------------------------------------------------------------

def _candidate_segment_pairs(A, B, L, seg_shell):
    """Border segment pairs (i, j) of different shells sharing a grid cell; shell(i) < shell(j)."""
    M = len(A)
    cell = max(2.0 * float(np.median(L)), float(L.sum()) / MAX_PIECES, 1e-12)
    seg, plo, phi = _segment_pieces(A, B, L, cell)
    box, key = _box_cells(plo[:, 0], plo[:, 1], phi[:, 0], phi[:, 1], cell)
    order = np.argsort(key, kind='stable')
    key = key[order]
    item = seg[box[order]]
    n = key.size
    gstart = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    gend = np.r_[gstart[1:], n]
    later = np.repeat(gend, gend - gstart) - np.arange(n) - 1  # entries after this one in its cell
    csum = np.cumsum(later)
    codes = []
    p0 = 0
    while p0 < n:
        base = csum[p0 - 1] if p0 else 0
        p1 = min(n, max(p0 + 1, int(np.searchsorted(csum, base + CHUNK, side='right'))))
        c = later[p0:p1]
        tot = int(c.sum())
        if tot:
            a = np.repeat(np.arange(p0, p1, dtype=np.int64), c)
            b = a + 1 + (np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(c) - c, c))
            i, j = item[a], item[b]
            si, sj = seg_shell[i], seg_shell[j]
            keep = si != sj
            i, j, swap = i[keep], j[keep], (si > sj)[keep]
            codes.append(np.where(swap, j, i) * M + np.where(swap, i, j))
        p0 = p1
    if not codes:
        return _EMPTY_I, _EMPTY_I
    code = _unique(np.concatenate(codes))
    return code // M, code % M


def _pair_crossings(A, B, i, j):
    """Proper crossings of segment pairs (i[k], j[k]): (pair indices k, crossing points).

    Segments meeting only at their ends, or lying on each other, just touch."""
    ks, pts = [], []
    for k0 in range(0, i.size, CHUNK):
        ii, jj = i[k0:k0 + CHUNK], j[k0:k0 + CHUNK]
        p, q = A[ii], A[jj]
        r, s = B[ii] - p, B[jj] - q
        den = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
        qp = q - p
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / den
            u = (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / den
        ok = np.abs(den) > 1e-9 * np.hypot(r[:, 0], r[:, 1]) * np.hypot(s[:, 0], s[:, 1])
        ok &= (t > CROSS_TOL) & (t < 1.0 - CROSS_TOL) & (u > CROSS_TOL) & (u < 1.0 - CROSS_TOL)
        if ok.any():
            sel = np.flatnonzero(ok)
            ks.append(k0 + sel)
            pts.append(p[sel] + r[sel] * t[sel][:, None])
    if not ks:
        return _EMPTY_I, _EMPTY2
    return np.concatenate(ks), np.concatenate(pts)


def _shell_probes(A, B, sign, off, eps):
    """Up to PROBES_PER_SHELL points per shell, just inside it next to spread-out border segments."""
    m = np.diff(off)
    k = np.minimum(PROBES_PER_SHELL, m)
    owner = np.repeat(np.arange(m.size, dtype=np.int64), k)
    j = (np.arange(owner.size, dtype=np.int64) - np.repeat(np.cumsum(k) - k, k)).astype(np.float64)
    kk, mm = k[owner], m[owner]
    step = np.where(kk > 1, (mm - 1) / np.maximum(kk - 1, 1), 0.0)
    idx = off[:-1][owner] + np.round(j * step).astype(np.int64)
    a = A[idx]
    d = B[idx] - a
    sg = sign[idx]
    ln = np.hypot(d[:, 0], d[:, 1])
    ok = (ln > 0.0) & (sg != 0.0)
    a, d, ln, sg, owner = a[ok], d[ok], ln[ok], sg[ok], owner[ok]
    inward = np.column_stack((-d[:, 1], d[:, 0])) * (sg / ln)[:, None]
    return a + d * 0.5 + inward * np.minimum(eps, 0.05 * ln)[:, None], owner


class _Pairs:
    """Overlapping shell pairs (s < c, gap-shell indices) in the order they are labeled: pairs
    whose borders cross (by s, c), then shells inside or stacked on another (in probe order).
    label: (k, 2) UV label points; crossing marks of pair i: cross_pts[cross_off[i]:cross_off[i + 1]].
    kind: 0 crossing borders, 1 inside; own / j / oth: for kind 1, the shell whose probe j lies
    inside shell oth (for ordering pairs measured in groups)."""
    __slots__ = ("s", "c", "label", "kind", "own", "j", "oth", "cross_off", "cross_pts")

    def __init__(self):
        self.s = self.c = self.kind = self.own = self.j = self.oth = _EMPTY_I
        self.label = _EMPTY2
        self.cross_off = np.zeros(1, dtype=np.int64)
        self.cross_pts = _EMPTY2

    def __len__(self):
        return int(self.s.size)

    def cross(self, i):
        return self.cross_pts[self.cross_off[i]:self.cross_off[i + 1]]

    def as_dict(self):
        """{(s, c): (label_uv, crossing_points_uv)}."""
        return {(int(s), int(c)): (self.label[i].copy(), self.cross(i).copy())
                for i, (s, c) in enumerate(zip(self.s.tolist(), self.c.tolist()))}


def _find_overlapping_pairs(geo):
    """Every overlapping shell pair (s < c), as _Pairs.

    Borders that cross -> overlap, labeled at the crossing nearest their center.
    No crossing, but a point just inside one shell lies inside the other -> the
    shell is contained in or stacked on the other, labeled at that point.
    Shells that only touch are not overlaps.
    """
    S = geo.nshells
    out = _Pairs()
    if S < 2:
        return out
    A, B, off, seg_shell = geo.seg_a, geo.seg_b, geo.shell_off, geo.seg_shell
    L = np.hypot(B[:, 0] - A[:, 0], B[:, 1] - A[:, 1])

    ccode, clabel, ccount, cpts = _EMPTY_I, _EMPTY2, _EMPTY_I, _EMPTY2
    i, j = _candidate_segment_pairs(A, B, L, seg_shell)
    if i.size:
        k, pts = _pair_crossings(A, B, i, j)
        if k.size:
            code = seg_shell[i[k]] * S + seg_shell[j[k]]
            order = np.argsort(code, kind='stable')
            code, pts = code[order], pts[order]
            starts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
            cnt = np.diff(np.r_[starts, code.size])
            grp = np.repeat(np.arange(starts.size, dtype=np.int64), cnt)
            # label: the crossing nearest the crossings' center (the first of equally near ones);
            # the center summed point by point in order, as mean() does (np.add.reduceat sums
            # pairwise, which can tip a tie between symmetric crossings)
            ctr = np.column_stack((_face_sums_in_order(pts[:, 0], starts, cnt),
                                   _face_sums_in_order(pts[:, 1], starts, cnt))) / cnt[:, None]
            d2 = ((pts - ctr[grp]) ** 2).sum(axis=1)
            o = np.lexsort((d2, grp))
            first = o[np.r_[True, grp[o][1:] != grp[o][:-1]]]
            ccode, clabel = code[starts], pts[first]
            # marks: every crossing, or MAX_CROSS_MARKS spread evenly along the pair's crossings
            take = np.minimum(cnt, MAX_CROSS_MARKS)
            pos = np.arange(int(take.sum()), dtype=np.int64) - np.repeat(np.cumsum(take) - take, take)
            n_of = np.repeat(cnt, take)
            step = np.where(n_of > MAX_CROSS_MARKS, (n_of - 1.0) / (MAX_CROSS_MARKS - 1), 1.0)
            pick = (pos * step).astype(np.int64)  # (as np.linspace(0, n - 1, MAX_CROSS_MARKS))
            pick = np.where((n_of > MAX_CROSS_MARKS) & (pos == MAX_CROSS_MARKS - 1), n_of - 1, pick)
            cpts = pts[np.repeat(starts, take) + pick]
            ccount = take

    pcode, plabel, pown, pj, poth = _EMPTY_I, _EMPTY2, _EMPTY_I, _EMPTY_I, _EMPTY_I
    probes, owner = _shell_probes(A, B, geo.seg_sign, off, UV_EPS)
    if len(probes):
        hp, hs = _probe_hits(probes, owner, _shell_grid(geo.bmin, geo.bmax),
                             geo.bmin, geo.bmax, A, B, _slopes(A, B), off)
        if hp.size:
            s = owner[hp]
            code = np.minimum(s, hs) * S + np.maximum(s, hs)
            _u, f = np.unique(code, return_index=True)
            f = np.sort(f)  # first time each pair is seen, in probe order
            f = f[~np.isin(code[f], ccode)]
            if f.size:
                pcode, plabel = code[f], probes[hp[f]]
                pown, poth = s[f], hs[f]
                first_probe = np.searchsorted(owner, np.arange(S), side='left')
                pj = hp[f] - first_probe[pown]

    nc, npr = ccode.size, pcode.size
    code = np.concatenate((ccode, pcode))
    out.s, out.c = code // S, code % S
    out.label = np.concatenate((clabel, plabel)).reshape(-1, 2)
    out.kind = np.concatenate((np.zeros(nc, dtype=np.int64), np.ones(npr, dtype=np.int64)))
    out.own = np.concatenate((np.full(nc, -1, dtype=np.int64), pown))
    out.j = np.concatenate((np.full(nc, -1, dtype=np.int64), pj))
    out.oth = np.concatenate((np.full(nc, -1, dtype=np.int64), poth))
    out.cross_off = np.concatenate(([0], np.cumsum(np.concatenate((ccount, np.zeros(npr, dtype=np.int64)))))).astype(np.int64)
    out.cross_pts = cpts.reshape(-1, 2)
    return out


# ---------------------------------------------------------------------------
# Measurement (texture-pixel space; results stored back in UV space)
# ---------------------------------------------------------------------------

class _Measurement:
    __slots__ = ("p_uv", "q_uv", "dist", "src", "dst",
                 "bp_uv", "bf_uv", "bdist", "baxis", "bsrc",
                 "ov_uv", "ov_src", "pair_label_uv", "pair_label_pi", "cross_uv", "cross_pi",
                 "overlap_codes", "tile_mark_uv", "tile_label_uv", "tile_crossing",
                 "shells_total", "shells_sampled", "samples", "overlap_pairs", "version", "stacks")

    def __init__(self):
        self.p_uv = _EMPTY2          # shell-to-shell gaps: border point, nearest neighbor point
        self.q_uv = _EMPTY2
        self.dist = _EMPTY_F
        self.src = _EMPTY_I
        self.dst = _EMPTY_I
        self.bp_uv = _EMPTY2         # tile-border distances: border point, foot on the tile edge
        self.bf_uv = _EMPTY2
        self.bdist = _EMPTY_F
        self.baxis = _EMPTY_I        # 0: vertical tile edge (u = const), 1: horizontal (v = const)
        self.bsrc = _EMPTY_I
        self.ov_uv = _EMPTY2         # measurement points inside another shell
        self.ov_src = _EMPTY_I       # (their shells)
        self.pair_label_uv = _EMPTY2
        self.pair_label_pi = _EMPTY_I  # (their pairs: index into geo.pairs)
        self.cross_uv = _EMPTY2
        self.cross_pi = _EMPTY_I
        self.overlap_codes = _EMPTY_I  # overlapping pairs s * S + c (s < c), sorted
        self.tile_mark_uv = _EMPTY2  # where shells cross tile lines
        self.tile_label_uv = _EMPTY2
        self.tile_crossing = 0
        self.shells_total = 0
        self.shells_sampled = 0
        self.samples = 0
        self.overlap_pairs = 0
        self.version = 0
        self.stacks = None           # which shells were measured as one (_Stacks), if any


def _shell_candidates(sh, lo, hi, A, B, L, seg_shell, tile_key=None):
    """Sorted pairs (row k, segment j): border segments of shells other than sh[k]
    (in the same group - tile and / or material - when tile_key is given) whose bounding
    box meets [lo[k], hi[k]]."""
    M = len(A)
    cell = max(0.5 * float(np.median((hi - lo).max(axis=1))), float(np.median(L)),
               float(L.sum()) / MAX_PIECES, 1e-9)
    while _cell_rows(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1], cell) > MAX_GRID_ROWS:
        cell *= 2.0
    seg, plo, phi = _segment_pieces(A, B, L, cell)
    grid = _Grid(cell, seg, plo, phi)
    qbox, first, count = grid.query(lo, hi)
    codes = []
    for r0, r1 in _query_chunks(qbox, count, len(sh)):
        k, pos = _expand(qbox[r0:r1], first[r0:r1], count[r0:r1])
        j = grid.items[pos]
        keep = ((seg_shell[j] != sh[k]) &
                (np.minimum(A[j, 0], B[j, 0]) <= hi[k, 0]) & (np.maximum(A[j, 0], B[j, 0]) >= lo[k, 0]) &
                (np.minimum(A[j, 1], B[j, 1]) <= hi[k, 1]) & (np.maximum(A[j, 1], B[j, 1]) >= lo[k, 1]))
        if tile_key is not None:
            keep &= tile_key[seg_shell[j]] == tile_key[sh[k]]
        codes.append(k[keep] * M + j[keep])
    if not codes:
        return _EMPTY_I, _EMPTY_I
    code = _unique(np.concatenate(codes))
    return code // M, code % M


def _dense_blocks(cand_k, cand_j, K, n):
    """Yield (rows, idx, valid): the candidate segments of several rows as one dense block.

    Rows with similar candidate counts share a power-of-two width (idx is padded,
    `valid` marks real entries), so a few large array operations cover every shell.
    """
    counts = np.bincount(cand_k, minlength=K)
    starts = np.cumsum(counts) - counts
    rows = np.flatnonzero(counts)
    if rows.size == 0:
        return
    width = np.left_shift(1, np.ceil(np.log2(counts[rows])).astype(np.int64))
    last = cand_j.size - 1
    for w in np.unique(width).tolist():
        rw = rows[width == w]
        step = max(1, DENSE_CHUNK // (n * w))
        col = np.arange(w)
        for c0 in range(0, rw.size, step):
            r = rw[c0:c0 + step]
            valid = col[None, :] < counts[r][:, None]
            idx = cand_j[np.minimum(starts[r][:, None] + col[None, :], last)]
            yield r, idx, valid


def _nearest_batched(P3, N3, cand_k, cand_j, A, D, L, R):
    """Nearest candidate border point within R, in front of the border, for every sample.

    P3, N3: (K, n, 2) sample points and outward normals of K shells.
    Returns (K, n) distance or inf, (K, n, 2) closest point, (K, n) segment index.
    """
    K, n = P3.shape[0], P3.shape[1]
    best = np.full((K, n), np.inf)
    bq = np.zeros((K, n, 2))
    bj = np.zeros((K, n), dtype=np.int64)
    if K == 0 or cand_k.size == 0:
        return best, bq, bj
    L2s = np.where(L > 0.0, L * L, 1.0)
    R2 = R * R * (1.0 + 1e-12)
    for r, idx, valid in _dense_blocks(cand_k, cand_j, K, n):
        ax, ay = A[idx, 0][:, None, :], A[idx, 1][:, None, :]
        dx, dy = D[idx, 0][:, None, :], D[idx, 1][:, None, :]
        px, py = P3[r, :, 0][:, :, None], P3[r, :, 1][:, :, None]
        # same arithmetic as t = clip(((p - a) . d) / |d|^2), v = a + t d - p, done in place
        t = np.subtract(px, ax)
        np.multiply(t, dx, out=t)
        vy = np.subtract(py, ay)
        np.multiply(vy, dy, out=vy)
        np.add(t, vy, out=t)
        np.divide(t, L2s[idx][:, None, :], out=t)
        np.clip(t, 0.0, 1.0, out=t)
        vx = np.multiply(t, dx)
        np.add(vx, ax, out=vx)
        np.subtract(vx, px, out=vx)
        np.multiply(t, dy, out=vy)
        np.add(vy, ay, out=vy)
        np.subtract(vy, py, out=vy)
        d2 = np.multiply(vx, vx)
        np.multiply(vy, vy, out=t)
        np.add(d2, t, out=d2)
        # padding, beyond R (not a neighbor), or not in front of the border: behind it would
        # measure through its own shell, sideways would run along its own edge
        np.multiply(vx, N3[r, :, 0][:, :, None], out=t)
        front = np.multiply(vy, N3[r, :, 1][:, :, None])
        np.add(t, front, out=t)
        np.sqrt(d2, out=front)
        np.multiply(front, EDGE_FACING, out=front)
        np.subtract(front, 1e-9, out=front)
        bad = t < front
        bad |= d2 > R2
        bad |= ~valid[:, None, :]
        d2[bad] = np.inf
        jj = d2.argmin(axis=2)[:, :, None]
        best[r] = np.take_along_axis(d2, jj, 2)[:, :, 0]
        bq[r, :, 0] = np.take_along_axis(vx, jj, 2)[:, :, 0] + P3[r, :, 0]
        bq[r, :, 1] = np.take_along_axis(vy, jj, 2)[:, :, 0] + P3[r, :, 1]
        bj[r] = np.take_along_axis(idx, jj[:, :, 0], 1)
    return np.sqrt(best), bq, bj


def _border_distances(P3, N3, edges, cand_k, cand_j, A, D, R):
    """Distance from each sample to the nearest edge of its shell's tile that the border
    faces (within R), unless a shell border - its own or another's - lies in between.

    P3, N3: (K, n, 2) points and outward normals; edges: (K, 4) tile lines x0, x1, y0, y1.
    Returns (K, n) distance or inf, (K, n, 2) foot point on the edge, (K, n) edge axis
    (0 = vertical line u = const, 1 = horizontal line v = const).
    """
    K, n = P3.shape[0], P3.shape[1]
    px, py = P3[:, :, 0], P3[:, :, 1]
    nx, ny = N3[:, :, 0], N3[:, :, 1]
    x0, x1 = edges[:, 0][:, None], edges[:, 1][:, None]
    y0, y1 = edges[:, 2][:, None], edges[:, 3][:, None]
    d = np.stack((px - x0, x1 - px, py - y0, y1 - py), axis=2)
    facing = np.stack((-nx, nx, -ny, ny), axis=2) >= EDGE_FACING
    d = np.where(facing & (d <= R), np.maximum(d, 0.0), np.inf)
    e = d.argmin(axis=2)
    dist = np.take_along_axis(d, e[:, :, None], 2)[:, :, 0]
    F = np.empty((K, n, 2))
    F[:, :, 0] = np.where(e == 0, x0, np.where(e == 1, x1, px))
    F[:, :, 1] = np.where(e == 2, y0, np.where(e == 3, y1, py))
    axis = (e >= 2).astype(np.int64)
    if cand_k.size and np.isfinite(dist).any():
        for r, idx, valid in _dense_blocks(cand_k, cand_j, K, n):
            live = np.isfinite(dist[r])
            if not live.any():
                continue
            ppx, ppy = px[r][:, :, None], py[r][:, :, None]
            rx = (F[r, :, 0] - px[r])[:, :, None]
            ry = (F[r, :, 1] - py[r])[:, :, None]
            ax, ay = A[idx, 0][:, None, :], A[idx, 1][:, None, :]
            sx, sy = D[idx, 0][:, None, :], D[idx, 1][:, None, :]
            qpx, qpy = ax - ppx, ay - ppy
            den = rx * sy - ry * sx
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (qpx * sy - qpy * sx) / den
                u = (qpx * ry - qpy * rx) / den
            hit = np.abs(den) > 1e-12 * np.hypot(rx, ry) * np.hypot(sx, sy)
            hit &= (t > 1e-6) & (t <= 1.0 + 1e-9) & (u >= -1e-9) & (u <= 1.0 + 1e-9)
            hit &= valid[:, None, :]
            blocked = hit.any(axis=2) & live
            dist[r] = np.where(blocked, np.inf, dist[r])
    return dist, F, axis


def _measure(geo, W, H, points, shift, radius, selected_only, tiles=False, same_mat=False):
    m = _Measurement()
    S = geo.nshells
    m.shells_total = S
    if S == 0:
        return m
    mat = geo.shell_mat if same_mat and geo.shell_mat.size == S else None
    scale = np.array((float(W), float(H)))
    A = geo.seg_a * scale
    B = geo.seg_b * scale
    D = B - A
    L = np.hypot(D[:, 0], D[:, 1])
    Ls = np.where(L > 0.0, L, 1.0)
    inward = np.column_stack((-D[:, 1], D[:, 0])) * (geo.seg_sign / Ls)[:, None]
    off = geo.shell_off
    seg_shell = geo.seg_shell
    bmin, bmax = geo.bmin * scale, geo.bmax * scale
    R = max(0.0, float(radius))
    n = max(1, int(points))
    tile_key = ((geo.tile[:, 0] + _TILE_OFF) * _TILE_MUL + geo.tile[:, 1] + _TILE_OFF) if tiles else None
    group = tile_key  # gaps are measured only within a group: same tile and / or same material
    if mat is not None:
        group = mat if group is None else group * (int(mat.max()) + 1) + mat

    # evenly spaced points along each shell's whole border (all loops), shifted along it
    cum0 = np.concatenate(([0.0], np.cumsum(L)))
    start = cum0[off[:-1]]
    total = cum0[off[1:]] - start
    act = total > 1e-9
    if selected_only:
        act &= geo.shell_sel
    sh = np.flatnonzero(act)
    covered = _EMPTY_I  # pairs (s * S + c) with a measurement point of one inside the other
    if sh.size:
        frac = (np.arange(n, dtype=np.float64) + (float(shift) % 1.0)) / n
        T = (start[sh][:, None] + total[sh][:, None] * frac[None, :]).ravel()
        s_of = np.repeat(sh, n)
        g = np.searchsorted(cum0, T, side='right') - 1
        g = np.clip(g, off[s_of], off[s_of + 1] - 1)
        f = np.clip((T - cum0[g]) / Ls[g], 0.0, 1.0)
        P = A[g] + D[g] * f[:, None]
        m.shells_sampled = int(sh.size)
        m.samples = int(P.shape[0])

        # 1) overlap: the point, nudged into its own shell, lies inside another shell.
        #    Step off the segment ends first: at a corner, the inward normal of one
        #    edge runs along the other edge and could land on a touching neighbor's border.
        end = np.minimum(2.0 * PX_EPS / Ls[g], 0.5)
        probe = A[g] + D[g] * np.clip(f, end, 1.0 - end)[:, None] + inward[g] * PX_EPS
        hp, hs = _probe_hits(probe, s_of, _shell_grid(bmin, bmax), bmin, bmax, A, B, _slopes(A, B), off)
        if mat is not None and hp.size:  # shells of other materials use another texture
            keep = mat[s_of[hp]] == mat[hs]
            hp, hs = hp[keep], hs[keep]
        ov = np.zeros(P.shape[0], dtype=bool)
        if hp.size:
            ov[hp] = True
            a = s_of[hp]
            covered = _unique(np.minimum(a, hs) * S + np.maximum(a, hs))
            m.ov_uv = P[ov] / scale
            m.ov_src = s_of[ov]

        # 2) gap: nearest border point of another shell within the radius, in front of this border
        #    (not searched from shells whose points all lie inside other shells: overlaps)
        P3 = P.reshape(-1, n, 2)
        N3 = (-inward[g]).reshape(-1, n, 2)
        live = ~ov.reshape(-1, n).all(axis=1)
        rows = np.flatnonzero(live)
        ck = cj = _EMPTY_I
        if rows.size:
            ck, cj = _shell_candidates(sh[rows], bmin[sh[rows]] - R, bmax[sh[rows]] + R, A, B, L, seg_shell, group)
            ck = rows[ck]
        if R > 0.0:
            dist, Q, j = _nearest_batched(P3, N3, ck, cj, A, D, L, R)
            dist, Q, j = dist.ravel(), Q.reshape(-1, 2), j.ravel()
            hit = np.flatnonzero(np.isfinite(dist) & ~ov)
            m.p_uv = P[hit] / scale
            m.q_uv = Q[hit] / scale
            m.dist = dist[hit]
            m.src = s_of[hit]
            m.dst = seg_shell[j[hit]]

        # 3) tile border: shells within R of an edge of their own tile, measured where the
        #    border faces that edge and no shell border lies in between
        if tiles and R > 0.0:
            tl = geo.tile[sh].astype(np.float64)
            edges = np.column_stack((tl[:, 0] * W, (tl[:, 0] + 1.0) * W, tl[:, 1] * H, (tl[:, 1] + 1.0) * H))
            room = np.minimum(np.minimum(bmin[sh, 0] - edges[:, 0], edges[:, 1] - bmax[sh, 0]),
                              np.minimum(bmin[sh, 1] - edges[:, 2], edges[:, 3] - bmax[sh, 1]))
            near = np.flatnonzero((room <= R) & ~geo.tile_cross[sh] & live)
            if near.size:
                M = len(A)
                remap = np.full(sh.size, -1, dtype=np.int64)
                remap[near] = np.arange(near.size)
                keep = remap[ck] >= 0
                own = sh[near]
                own_len = off[own + 1] - off[own]
                code = _unique(np.concatenate((
                    remap[ck[keep]] * M + cj[keep],
                    np.repeat(np.arange(near.size, dtype=np.int64), own_len) * M
                    + _ranges(off[own], off[own + 1]))))
                bd, F, axis = _border_distances(P3[near], N3[near], edges[near], code // M, code % M, A, D, R)
                ok = np.isfinite(bd) & ~ov.reshape(-1, n)[near]
                if ok.any():
                    m.bp_uv = P3[near][ok] / scale
                    m.bf_uv = F[ok] / scale
                    m.bdist = bd[ok]
                    m.baxis = axis[ok]
                    m.bsrc = np.repeat(own, n).reshape(-1, n)[ok]

    prs = geo.pairs if geo.pairs is not None else _find_overlapping_pairs(geo)
    k = np.arange(len(prs), dtype=np.int64)
    if selected_only and k.size:
        k = k[geo.shell_sel[prs.s] | geo.shell_sel[prs.c]]
    if mat is not None and k.size:
        k = k[mat[prs.s[k]] == mat[prs.c[k]]]
    pcode = prs.s[k] * S + prs.c[k]
    m.overlap_codes = _unique(np.concatenate((covered, pcode))).astype(np.int64)
    m.overlap_pairs = int(m.overlap_codes.size)
    lab = k[~np.isin(pcode, covered)]  # no measurement point caught these overlaps: label them anyway
    if lab.size:
        m.pair_label_uv = prs.label[lab]
        m.pair_label_pi = lab
    if k.size:
        n_cross = prs.cross_off[k + 1] - prs.cross_off[k]
        if n_cross.any():
            m.cross_uv = prs.cross_pts[_ranges(prs.cross_off[k], prs.cross_off[k + 1])]
            m.cross_pi = np.repeat(k, n_cross)
    if tiles:
        _tile_marks(geo, selected_only, m)
    return m


def _tile_marks(geo, selected_only, m):
    """Marks and labels of the shells crossing a tile line."""
    if not geo.tile_cross_pts:
        return
    tmarks, tlabels = [], []
    for s, pts in geo.tile_cross_pts.items():
        if selected_only and not geo.shell_sel[s]:
            continue
        tmarks.append(pts)
        tlabels.append(pts[0])
    m.tile_crossing = len(tlabels)
    if tlabels:
        m.tile_mark_uv = np.concatenate(tmarks)
        m.tile_label_uv = np.array(tlabels, dtype=np.float64).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Measuring material by material, stacks as one shell, results kept between measurements
# ---------------------------------------------------------------------------

class _MeasureCache:
    """Results kept between measurements, per group of shells (one material, or all shells):
    stacks and overlapping pairs by the group's content, gap measurements by content and
    settings. An entry is dropped when the last `keep` computations using its table didn't
    need it."""
    TABLES = ("stacks", "pairs", "meas")

    def __init__(self, keep=4):
        self.keep = int(keep)
        self.tables = {name: {} for name in self.TABLES}
        self.calls = {name: 0 for name in self.TABLES}
        self.stats = (0, 0)  # groups measured / groups, in the last measurement

    def begin(self, *names):
        for name in names:
            self.calls[name] += 1

    def get(self, name, key):
        table = self.tables[name]
        hit = table.get(key)
        if hit is None:
            return None
        table[key] = (hit[0], self.calls[name])
        return hit[0]

    def put(self, name, key, value):
        self.tables[name][key] = (value, self.calls[name])

    def trim(self, *names):
        for name in names:
            old = self.calls[name] - self.keep
            table = self.tables[name]
            for k in [k for k, (_v, c) in table.items() if c <= old]:
                del table[k]

    def clear(self):
        for table in self.tables.values():
            table.clear()


def _shell_groups(geo, same_mat):
    """[(material key or None, gap-shell indices, ascending)]: one group per material when
    shells of other materials don't count (Same Material Only), else one group of all shells."""
    S = geo.nshells
    if not S:
        return []
    if not same_mat or geo.shell_mat.size != S:
        return [(None, np.arange(S, dtype=np.int64))]
    order = np.argsort(geo.shell_mat, kind='stable')
    m = geo.shell_mat[order]
    starts = np.flatnonzero(np.r_[True, m[1:] != m[:-1]])
    ends = np.r_[starts[1:], S]
    return [(geo.mat_keys[int(m[a])], order[a:b]) for a, b in zip(starts.tolist(), ends.tolist())]


def _group_sig(geo, gkey, g):
    """What a group's shells depend on: its material and the UV data of its objects."""
    objs = _unique(geo.sh_obj[geo.gap_shell[g]])
    return (gkey, tuple(geo.obj_uvsig[i] for i in objs.tolist()))


def _with_selection(geo, shell_sel):
    """A shallow copy of geo with other selected shells (shell_sel, per gap shell)."""
    out = _Geometry()
    for f in _Geometry.__slots__:
        setattr(out, f, getattr(geo, f))
    out.shell_sel = np.asarray(shell_sel, dtype=bool)
    return out


def _sub_geometry(geo, g):
    """The gap shells g (ascending gap-shell indices) as a _Geometry of their own: what the
    gap measurement and the overlap search need."""
    sub = _Geometry()
    S = geo.nshells
    whole = g.size == S
    if whole:
        sub.seg_a, sub.seg_b, sub.seg_sign = geo.seg_a, geo.seg_b, geo.seg_sign
        sub.shell_off, sub.seg_shell = geo.shell_off, geo.seg_shell
    else:
        seg = _ranges(geo.shell_off[g], geo.shell_off[g + 1])
        cnt = geo.shell_off[g + 1] - geo.shell_off[g]
        sub.seg_a, sub.seg_b, sub.seg_sign = geo.seg_a[seg], geo.seg_b[seg], geo.seg_sign[seg]
        sub.shell_off = np.concatenate(([0], np.cumsum(cnt))).astype(np.int64)
        sub.seg_shell = np.repeat(np.arange(g.size, dtype=np.int64), cnt)
    for f in ("shell_sel", "bmin", "bmax", "shell_mat", "tile", "tile_cross", "shell_area", "flipped"):
        a = getattr(geo, f)
        setattr(sub, f, a if whole else a[g])
    sub.nshells = int(g.size)
    sub.tile_cross_pts = {}  # (tile crossings: marked once for all groups)
    return sub


# ---------------------------------------------------------------------------
# Stacks: shells lying on each other on purpose (copies of a part sharing texture space)
# ---------------------------------------------------------------------------

STACK_MATCH = 0.999      # shells covering each other at least this much are one stack
_STACK_SAMPLES = 256     # border samples per shell when two shells are compared by distance
_STACK_ROUNDS = 16
_STACK_WINDOW = 64


class _Stacks:
    """Per gap shell: `rep`, the shell that stands for its stack in the measurements (itself
    when it lies alone; a shell that isn't flipped where the stack has one), `size`, the
    number of shells in its stack, and `nflip`, how many of them are flipped."""
    __slots__ = ("rep", "size", "nflip")

    def __init__(self, n=0):
        self.rep = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)
        self.nflip = np.zeros(n, dtype=np.int64)

    def unique(self):
        """Gap shells that stand for themselves: every stack once."""
        return np.flatnonzero(self.rep == np.arange(self.rep.size))

    def copies(self):
        """Gap shells lying on their stack's representative: all but one of every stack."""
        return np.flatnonzero(self.rep != np.arange(self.rep.size))

    def members(self, reps):
        """Every shell of the stacks of `reps`."""
        pick = np.zeros(self.rep.size, dtype=bool)
        pick[np.asarray(reps, dtype=np.int64)] = True
        return np.flatnonzero(pick[self.rep])

    def counts(self):
        """(stacks of two or more shells, shells in them)."""
        u = self.unique()
        big = self.size[u] > 1
        return int(np.count_nonzero(big)), int(self.size[u][big].sum())


def _border_offsets(A, B, L, off, s, c):
    """Per pair (s[k], c[k]): the distance from s's border to c's border, summed along s's
    border - about the area between the two borders where they nearly coincide. Borders are
    sampled at segment middles (at most _STACK_SAMPLES per shell)."""
    out = np.zeros(s.size)
    if not s.size:
        return out
    D = B - A
    L2 = np.where(L > 0.0, L * L, 1.0)
    ns, mc = off[s + 1] - off[s], off[c + 1] - off[c]
    stride = np.maximum(1, -(-ns // _STACK_SAMPLES))
    nsamp = -(-ns // stride)
    cls = np.ceil(np.log2(nsamp)).astype(np.int64) * 64 + np.ceil(np.log2(mc)).astype(np.int64)
    for k in _unique(cls).tolist():
        rows = np.flatnonzero(cls == k)
        NS, MC = int(nsamp[rows].max()), int(mc[rows].max())
        step = max(1, DENSE_CHUNK // (NS * MC))
        cs, cm = np.arange(NS), np.arange(MC)
        for r0 in range(0, rows.size, step):
            r = rows[r0:r0 + step]
            si = off[s[r]][:, None] + cs[None, :] * stride[r][:, None]
            vs = cs[None, :] < nsamp[r][:, None]
            si = np.where(vs, si, off[s[r]][:, None])
            P = A[si] + D[si] * 0.5                              # (K, NS, 2) border samples of s
            w = np.where(vs, L[si], 0.0)
            # each sample stands for its share of the whole border
            wsum = w.sum(axis=1)
            w = w * np.where(wsum > 0.0, _shell_perimeter(L, off, s[r]) / np.where(wsum > 0.0, wsum, 1.0), 0.0)[:, None]
            ci = off[c[r]][:, None] + cm[None, :]
            vc = cm[None, :] < mc[r][:, None]
            ci = np.where(vc, ci, off[c[r]][:, None])
            ax, ay = A[ci, 0][:, None, :], A[ci, 1][:, None, :]
            dx, dy = D[ci, 0][:, None, :], D[ci, 1][:, None, :]
            px, py = P[:, :, 0][:, :, None], P[:, :, 1][:, :, None]
            t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2[ci][:, None, :], 0.0, 1.0)
            vx, vy = ax + t * dx - px, ay + t * dy - py
            d2 = vx * vx + vy * vy
            d2[np.broadcast_to(~vc[:, None, :], d2.shape)] = np.inf
            out[r] = (np.sqrt(d2.min(axis=2)) * w).sum(axis=1)
    return out


def _shell_perimeter(L, off, shells):
    csum = np.concatenate(([0.0], np.cumsum(L)))
    return csum[off[shells + 1]] - csum[off[shells]]


def _stack_matcher(sub, match):
    """A function (s, c) -> which pairs of shells cover each other by at least `match`."""
    A, B, off = sub.seg_a, sub.seg_b, sub.shell_off
    L = np.hypot(B[:, 0] - A[:, 0], B[:, 1] - A[:, 1])
    n = sub.nshells
    all_shells = np.arange(n, dtype=np.int64)
    perim = _shell_perimeter(L, off, all_shells)
    area = np.abs(sub.shell_area)
    slack = 1.0 - float(match)
    # how far two borders may be apart on average before they differ by half the allowed area
    eps = np.maximum(slack * area / np.where(perim > 0.0, perim, 1.0), 1e-9)
    # border vertices sorted within each shell: copies list the same vertices
    order = np.lexsort((A[:, 1], A[:, 0], sub.seg_shell))
    V = A[order]

    def matches(s, c):
        ok = np.zeros(s.size, dtype=bool)
        if not s.size:
            return ok
        ns, nc = off[s + 1] - off[s], off[c + 1] - off[c]
        big = np.maximum(area[s], area[c])
        near = np.abs(area[s] - area[c]) <= 2.0 * slack * big + 1e-18
        same = np.flatnonzero(near & (ns == nc))
        if same.size:
            ia, ib = _ranges(off[s[same]], off[s[same] + 1]), _ranges(off[c[same]], off[c[same] + 1])
            d = np.abs(V[ia] - V[ib]).max(axis=1)
            worst = np.maximum.reduceat(d, np.cumsum(ns[same]) - ns[same])
            ok[same] = worst <= np.minimum(eps[s[same]], eps[c[same]])
        rest = np.flatnonzero(near & ~ok & (big > 0.0))
        if rest.size:
            rs, rc = s[rest], c[rest]
            delta = np.maximum(_border_offsets(A, B, L, off, rs, rc), _border_offsets(A, B, L, off, rc, rs))
            ok[rest] = (area[rs] + area[rc] - delta) >= 2.0 * float(match) * big[rest]
        return ok
    return matches


def _find_stacks(sub, match):
    """Stacks among the shells of `sub`: (rep, size, nflip) per shell (local indices)."""
    n = sub.nshells
    st = _Stacks(n)
    if n < 2:
        st.nflip = sub.flipped.astype(np.int64) if sub.flipped.size == n else st.nflip
        return st
    matches = _stack_matcher(sub, match)
    box = np.column_stack((sub.bmin, sub.bmax))
    # 1) shells with the same bounds (rounded): each is compared with the first of them; those
    #    that differ from it are compared among themselves next
    key = np.rint(box / 1e-6).astype(np.int64)
    ea, eb = [], []
    todo = np.arange(n, dtype=np.int64)
    for _round in range(_STACK_ROUNDS):
        if todo.size < 2:
            break
        k = key[todo]
        o = np.lexsort((todo, k[:, 3], k[:, 2], k[:, 1], k[:, 0]))
        u, kk = todo[o], k[o]
        first = np.r_[True, (kk[1:] != kk[:-1]).any(axis=1)]
        leader = u[np.maximum.accumulate(np.where(first, np.arange(u.size), 0))]
        s, c = leader[~first], u[~first]
        if not s.size:
            break
        ok = matches(s, c)
        ea.append(s[ok])
        eb.append(c[ok])
        todo = c[~ok]
    comp = _components(n, np.concatenate(ea) if ea else _EMPTY_I, np.concatenate(eb) if eb else _EMPTY_I)
    # 2) stacks whose bounds differ by less than the match allows (copies that aren't exact):
    #    each is compared with the next ones along U, then along V (shells lined up along one
    #    of them are many; along both, only shells lying on each other)
    tol = np.maximum(2.0 * (1.0 - float(match)) * (sub.bmax - sub.bmin), 1e-9)
    for axis in (0, 1):
        roots = np.flatnonzero(comp == np.arange(n))
        if roots.size < 2:
            break
        o = np.argsort(sub.bmin[roots, axis], kind='stable')
        r = roots[o]
        xs = sub.bmin[r, axis]
        hi = np.searchsorted(xs, xs + tol[r, axis], side='right')
        cnt = np.minimum(hi - np.arange(r.size) - 1, _STACK_WINDOW)
        tot = int(cnt.sum())
        if not tot:
            continue
        i = np.repeat(np.arange(r.size, dtype=np.int64), cnt)
        j = i + 1 + (np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        a, b = r[i], r[j]
        t = np.maximum(tol[a], tol[b])
        keep = ((np.abs(box[a] - box[b]) <= np.concatenate((t, t), axis=1)).all(axis=1) &
                ((key[a] != key[b]).any(axis=1)))
        a, b = a[keep], b[keep]
        ok = matches(a, b)
        if ok.any():
            ea.append(a[ok])
            eb.append(b[ok])
            comp = _components(n, np.concatenate(ea), np.concatenate(eb))
    # the stack's representative: a shell that isn't flipped where there is one, then the first
    flipped = sub.flipped if sub.flipped.size == n else np.zeros(n, dtype=bool)
    idx = np.arange(n, dtype=np.int64)
    o = np.lexsort((idx, flipped, comp))
    firsts = o[np.r_[True, comp[o][1:] != comp[o][:-1]]]
    rep_of_comp = np.empty(n, dtype=np.int64)
    rep_of_comp[comp[firsts]] = firsts
    st.rep = rep_of_comp[comp]
    st.size = np.bincount(st.rep, minlength=n)[st.rep]
    st.nflip = np.bincount(st.rep, weights=flipped, minlength=n).astype(np.int64)[st.rep]
    return st


def _grouped(geo, same_mat, stack_match, cache):
    """[(signature, shells measured (ascending gap-shell indices), _Stacks of the group or
    None, all the group's shells)]: per material (or for all shells), with stacks reduced to
    their representatives when `stack_match` is given."""
    out = []
    for gkey, g in _shell_groups(geo, same_mat):
        sig = _group_sig(geo, gkey, g)
        if stack_match is None:
            out.append((sig, g, None, g))
            continue
        skey = (sig, round(float(stack_match), 9))
        st = cache.get("stacks", skey) if cache is not None else None
        if st is None:
            st = _find_stacks(_sub_geometry(geo, g), stack_match)
            if cache is not None:
                cache.put("stacks", skey, st)
        out.append((skey, g[st.unique()], st, g))
    return out


def _merge_stacks(geo, grouped):
    st = _Stacks(geo.nshells)
    if geo.flipped.size == geo.nshells:
        st.nflip = geo.flipped.astype(np.int64)
    for _sig, _gu, local, g in grouped:
        if local is not None:
            st.rep[g] = g[local.rep]
            st.size[g] = local.size
            st.nflip[g] = local.nflip
    return st


def _stacks_of(geo, same_mat, stack_match=STACK_MATCH, cache=None):
    """The stacks of the geometry (_Stacks, over the gap shells): of the same material only
    when `same_mat`."""
    if cache is not None:
        cache.begin("stacks")
    grouped = _grouped(geo, same_mat, stack_match, cache)
    if cache is not None:
        cache.trim("stacks")
    return _merge_stacks(geo, grouped)


def _merge_pairs(S, parts):
    """Global _Pairs from per-group ones: parts [(g, pairs)]. Returns (pairs, index of each
    group's pairs in the result)."""
    out = _Pairs()
    if not parts:
        return out, []
    s = np.concatenate([g[p.s] for g, p in parts])
    c = np.concatenate([g[p.c] for g, p in parts])
    kind = np.concatenate([p.kind for _g, p in parts])
    key = np.concatenate([np.where(p.kind == 0, g[p.s] * S + g[p.c],
                                   (g[np.maximum(p.own, 0)] * PROBES_PER_SHELL + p.j) * S + g[np.maximum(p.oth, 0)])
                          for g, p in parts])
    o = np.lexsort((key, kind))
    inv = np.empty(o.size, dtype=np.int64)
    inv[o] = np.arange(o.size)
    out.s, out.c, out.kind = s[o], c[o], kind[o]
    out.label = np.concatenate([p.label for _g, p in parts]).reshape(-1, 2)[o]
    own = np.concatenate([np.where(p.own >= 0, g[np.maximum(p.own, 0)], -1) for g, p in parts])
    oth = np.concatenate([np.where(p.oth >= 0, g[np.maximum(p.oth, 0)], -1) for g, p in parts])
    out.own, out.oth = own[o], oth[o]
    out.j = np.concatenate([p.j for _g, p in parts])[o]
    lo = np.concatenate([p.cross_off[:-1] for _g, p in parts])
    hi = np.concatenate([p.cross_off[1:] for _g, p in parts])
    base = np.repeat(np.cumsum([0] + [p.cross_pts.shape[0] for _g, p in parts])[:-1], [len(p) for _g, p in parts])
    lo, hi = (lo + base)[o], (hi + base)[o]
    pts = np.concatenate([p.cross_pts for _g, p in parts]).reshape(-1, 2)
    out.cross_pts = pts[_ranges(lo, hi)]
    out.cross_off = np.concatenate(([0], np.cumsum(hi - lo))).astype(np.int64)
    offs = np.cumsum([0] + [len(p) for _g, p in parts])
    return out, [inv[a:b] for a, b in zip(offs[:-1].tolist(), offs[1:].tolist())]


def _group_pairs(geo, grouped, cache):
    out = []
    for sig, g, _st, _all in grouped:
        prs = cache.get("pairs", sig) if cache is not None else None
        if prs is None:
            prs = _find_overlapping_pairs(_sub_geometry(geo, g))
            if cache is not None:
                cache.put("pairs", sig, prs)
        out.append((sig, g, prs))
    return out


def _overlap_pairs(geo, same_mat, cache=None, stack_match=None):
    """Overlapping shell pairs (_Pairs, gap-shell indices), of the same material only when
    `same_mat`. With `stack_match`, stacks count as one shell (their representative): the
    pairs are the overlaps that aren't stacks."""
    if cache is not None:
        cache.begin("stacks", "pairs")
    parts = _group_pairs(geo, _grouped(geo, same_mat, stack_match, cache), cache)
    if cache is not None:
        cache.trim("stacks", "pairs")
    return _merge_pairs(geo.nshells, [(g, p) for _s, g, p in parts])[0]


def _measure_grouped(geo, W, H, points, shift, radius, selected_only, tiles=False, same_mat=False, cache=None,
                     stack_match=None):
    """Gaps, tile-border distances and overlaps. With `same_mat`, every material is measured
    on its own (shells of other materials use another texture), which is the same as
    filtering by material but skips the work between materials. With `stack_match` (0..1),
    shells covering each other at least that much are a stack - copies sharing texture space
    on purpose - and are measured as one shell: the result's `stacks` tells which.
    `cache`: a _MeasureCache, to measure again only the groups that changed."""
    S = geo.nshells
    if S == 0:
        m = _Measurement()
        return m
    if cache is not None:
        cache.begin(*cache.TABLES)
    grouped = _grouped(geo, same_mat, stack_match, cache)
    stacks = _merge_stacks(geo, grouped) if stack_match is not None else None
    sel = geo.shell_sel
    if stacks is not None and selected_only and sel.size == S:
        # a stack is selected when one of its shells is
        sel = np.bincount(stacks.rep, weights=sel, minlength=S)[stacks.rep] > 0
    parts = []
    computed = 0
    for sig, g, prs in _group_pairs(geo, grouped, cache):
        msig = (sig, int(W), int(H), int(points), float(shift), float(radius), bool(selected_only), bool(tiles),
                sel[g].tobytes() if selected_only else None)
        m = cache.get("meas", msig) if cache is not None else None
        if m is None:
            sub = _sub_geometry(geo, g)
            sub.pairs = prs
            if selected_only:
                sub.shell_sel = sel[g]
            m = _measure(sub, W, H, points, shift, radius, selected_only, tiles, False)
            computed += 1
            if cache is not None:
                cache.put("meas", msig, m)
        parts.append((g, prs, m))
    if cache is not None:
        cache.stats = (computed, len(grouped))
        cache.trim(*cache.TABLES)
    out = _merge_measurements(geo, parts)
    out.stacks = stacks
    if tiles:
        _tile_marks(_with_selection(geo, sel) if selected_only else geo, selected_only, out)
    return out


def _merge_measurements(geo, parts):
    """One measurement from per-group ones (parts [(g, pairs, measurement)]), in the order a
    measurement of all shells at once lists its results."""
    S = geo.nshells
    m = _Measurement()
    m.shells_total = S
    if not parts:
        return m
    ms = [pm for _g, _p, pm in parts]
    m.shells_sampled = sum(pm.shells_sampled for pm in ms)
    m.samples = sum(pm.samples for pm in ms)
    m.overlap_pairs = sum(pm.overlap_pairs for pm in ms)

    def cat(name, shape2=False):
        arrs = [getattr(pm, name) for pm in ms]
        out = np.concatenate(arrs) if arrs else (_EMPTY2 if shape2 else _EMPTY_F)
        return out.reshape(-1, 2) if shape2 else out

    def by_shell(name):
        return np.concatenate([g[getattr(pm, name)] for g, _p, pm in parts]).astype(np.int64)

    # results of one shell stay together and in order; shells in global order
    src = by_shell("src")
    o = np.argsort(src, kind='stable')
    m.src, m.dst = src[o], by_shell("dst")[o]
    m.p_uv, m.q_uv, m.dist = cat("p_uv", True)[o], cat("q_uv", True)[o], cat("dist")[o]
    bsrc = by_shell("bsrc")
    o = np.argsort(bsrc, kind='stable')
    m.bsrc = bsrc[o]
    m.bp_uv, m.bf_uv, m.bdist = cat("bp_uv", True)[o], cat("bf_uv", True)[o], cat("bdist")[o]
    m.baxis = np.concatenate([pm.baxis for pm in ms]).astype(np.int64)[o]
    ov_src = by_shell("ov_src")
    o = np.argsort(ov_src, kind='stable')
    m.ov_src, m.ov_uv = ov_src[o], cat("ov_uv", True)[o]

    prs, gidx = _merge_pairs(S, [(g, p) for g, p, _m in parts])
    pi = np.concatenate([gi[pm.pair_label_pi] for gi, pm in zip(gidx, ms)]).astype(np.int64)
    o = np.argsort(pi, kind='stable')
    m.pair_label_pi, m.pair_label_uv = pi[o], cat("pair_label_uv", True)[o]
    ci = np.concatenate([gi[pm.cross_pi] for gi, pm in zip(gidx, ms)]).astype(np.int64)
    o = np.argsort(ci, kind='stable')
    m.cross_pi, m.cross_uv = ci[o], cat("cross_uv", True)[o]
    codes = []
    for g, _p, pm in parts:
        n = g.size
        c = pm.overlap_codes
        codes.append(g[c // max(n, 1)] * S + g[c % max(n, 1)])
    m.overlap_codes = _unique(np.concatenate(codes).astype(np.int64))
    return m


def _overlapping_shells(geo, pairs, stacks=None):
    """All-shell indices of the shells in an overlapping pair; with `stacks`, every shell of
    the stacks involved."""
    if pairs is None or not len(pairs):
        return _EMPTY_I
    g = _unique(np.concatenate((pairs.s, pairs.c)))
    if stacks is not None:
        g = stacks.members(g)
    return geo.gap_shell[g]


def _overlaps_but_one(geo, pairs, stacks=None):
    """Shells to move away so that no shell lies on another: of every stack, and of every set
    of overlapping shells, one stays. Which: one that isn't flipped where there is one, then
    the one standing for the most shells (a stack before a single shell lying on it), then the
    largest, then the first. (Greedy: shells are kept in that order unless they overlap one
    kept already; a stack goes or stays as the shell standing for it does, its copies always
    go.) All-shell indices."""
    S = geo.nshells
    moved = np.zeros(S, dtype=bool)
    if stacks is not None:
        moved[stacks.copies()] = True
    if pairs is not None and len(pairs):
        s, c = pairs.s, pairs.c
        inv = _unique(np.concatenate((s, c)))
        flipped = geo.flipped[inv] if geo.flipped.size else np.zeros(inv.size, dtype=bool)
        area = np.abs(geo.shell_area[inv]) if geo.shell_area.size else np.zeros(inv.size)
        top = float(area.max()) if area.size else 0.0
        # (copies have the same area up to rounding: sizes compared to six digits)
        size = np.rint(area / top * 1e6) if top > 0.0 else np.zeros(inv.size)
        count = stacks.size[inv] if stacks is not None else np.ones(inv.size, dtype=np.int64)
        order = inv[np.lexsort((inv, -size, -count, flipped))]
        nbr = {}
        for a, b in zip(s.tolist(), c.tolist()):
            nbr.setdefault(a, []).append(b)
            nbr.setdefault(b, []).append(a)
        kept, gone = set(), []
        for g in order.tolist():
            if any(x in kept for x in nbr.get(g, ())):
                gone.append(g)
            else:
                kept.add(g)
        if gone:
            gone = np.asarray(gone, dtype=np.int64)
            moved[stacks.members(gone) if stacks is not None else gone] = True
    return geo.gap_shell[np.flatnonzero(moved)]


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _geometry_signature(objects, sync, want_sel, want_td=False, want_orient=False):
    parts = []
    for obj in objects:
        bm = bmesh.from_edit_mesh(obj.data)
        uvl = bm.loops.layers.uv.active
        parts.append((obj.data.as_pointer(), len(bm.verts), len(bm.faces),
                      uvl.name if uvl is not None else "",
                      (tuple(tuple(r) for r in obj.matrix_world), tuple(obj.scale)) if want_td else None,
                      tuple(_material_keys(obj)), obj.name))
    return (sync, want_sel, want_td, want_orient, tuple(parts))


def _redraw_timer():
    scene = getattr(bpy.context, "scene", None)
    st = getattr(scene, "uv_gap_overlay", None) if scene is not None else None
    _redraw_areas(('IMAGE_EDITOR', 'VIEW_3D') if st is not None and st.td_show_3d else ('IMAGE_EDITOR',))
    return None


def _schedule_redraw(delay):
    try:
        if not bpy.app.timers.is_registered(_redraw_timer):
            bpy.app.timers.register(_redraw_timer, first_interval=max(0.02, delay))
    except Exception:
        pass


def _poll_uv_selection(objects, geo):
    """Read the UV selection of the shown objects again. Without UV Sync Selection, Blender
    announces no UV selection change, so Selected Shells Only looks now and then (see
    _ensure_geometry). True when a shell's selection changed."""
    st = _State
    t0 = time.perf_counter()
    changed = False
    for P, oi in zip(geo.obj_parts, geo.obj_src):  # (the objects something is shown of)
        obj = objects[oi]
        if not obj.data.is_editmode:
            continue
        A = _read_edit_arrays(obj, False)
        if A is None or A.uv_sel.size != P.A.uv_sel.size or A.v_sel.size != P.A.v_sel.size:
            continue  # (the mesh itself changed: that arrives as an update)
        sel = _parts_selection(P, _corner_selection(A, _sel_kind(True, False, A.sync_valid)))
        P.A.uv_sel, P.A.v_sel, P.A.sel_readable = A.uv_sel, A.v_sel, A.sel_readable
        if not np.array_equal(sel, P.gap["sel"]):
            P.gap["sel"] = sel
            changed = True
    st.sel_poll_ms = (time.perf_counter() - t0) * 1000.0
    # reading costs time: look again after eight times as long, five times a second at most
    st.sel_poll = time.perf_counter() + max(0.2, st.sel_poll_ms * 0.008)
    return changed


def _selected_shells_changed(geo):
    """Take over the selected shells of the geometry's objects (kept shells, other selection)."""
    if geo.obj_parts:
        geo.shell_sel = np.concatenate([P.gap["sel"] for P in geo.obj_parts]).astype(bool)
    _State.sel_version += 1


def _ensure_geometry(objects, sync, want_sel, want_td=False, want_orient=False, fresh=False, poll=False):
    """(geometry, stale). Heavy meshes keep the previous result until edits pause.

    Objects keep their shells while they don't change: changes arrive as depsgraph updates,
    counted per object; what Blender doesn't announce (a renamed material, an object moved by
    its parent) is caught by comparing a signature of the objects, at once after any update
    and SIG_CHECK_S apart otherwise. `fresh`: read every object again and never answer with a
    stale result (operators act on it). `poll`: this redraw may look for UV selection changes
    (the view stands still)."""
    st = _State
    now = time.perf_counter()
    flags = (bool(sync), bool(want_sel), bool(want_td), bool(want_td and want_orient))
    dirty = st.geo_dirty or (st.sel_dirty and _need_selection(sync, want_sel) != SEL_NONE)
    quick = (flags, st.epoch, st.data_gen, tuple([o.data.as_pointer() for o in objects]))
    valid = False
    if not fresh and st.geo is not None and not dirty and quick == st.quick_key and now - st.sig_time < SIG_CHECK_S:
        valid = True
    else:
        sig = _geometry_signature(objects, sync, want_sel, want_td, want_orient)
        st.sig_time = now
        if not fresh and st.geo is not None and sig == st.geo_sig:
            if not dirty:
                st.quick_key = quick
                valid = True
            elif st.last_geo_ms > LIVE_MS:
                wait = DEBOUNCE_S - (now - st.last_change)
                if wait > 0.0:
                    _schedule_redraw(wait + 0.01)
                    return st.geo, True
    if valid:
        if flags[1] and not flags[0] and poll:  # Selected Shells Only without UV Sync Selection
            if now < st.sel_poll:
                _schedule_redraw(st.sel_poll - now + 0.01)
            elif _poll_uv_selection(objects, st.geo):
                _selected_shells_changed(st.geo)
        return st.geo, False

    parts, src, sel_changed, meta_changed = _object_parts(objects, sync, want_sel, want_td, not fresh, want_orient,
                                                           sig[4])
    serials = tuple(P.serial for P in parts)
    if st.geo is None or meta_changed or st.geo_flags != flags[2:] or serials != st.geo.src_serials:
        st.geo = _assemble(parts, src, want_td, flags[3])
        st.geo_flags = flags[2:]
        st.geo_version += 1
        st.meas_cache.clear()
        st.labels_cache.clear()
        st.td_values.clear()
        st.td_fill.clear()
        st.frame.clear()
        st.stacks = st.stacks_key = None
    else:
        st.geo.obj_src = [i for i, P in zip(src, parts) if P.gap is not None]
        if sel_changed:
            _selected_shells_changed(st.geo)
    st.last_geo_ms = (time.perf_counter() - now) * 1000.0
    st.geo_sig, st.quick_key = sig, quick
    st.geo_dirty = st.sel_dirty = False
    _prune_objects(objects)
    return st.geo, False


def _uv_geometry(scene, st, objects, fresh=False, poll=False):
    """The UV editor's shells (one cache for the overlay, the panels and the operators): texel
    density always, the selection for Selected Shells Only, orientation once the arrows have
    been shown (from then on, so that switching them off and on again costs nothing)."""
    if st.info_show and st.show_orientation:
        _State.orient_seen = True
    return _ensure_geometry(objects, bool(scene.tool_settings.use_uv_select_sync), bool(st.selected_only), True,
                            _State.orient_seen, fresh, poll)


def _stack_match(s):
    """How much shells must cover each other to be one stack (0..1), or None: stacks are off."""
    return max(0.5, min(1.0, float(s.stack_match) / 100.0)) if s.stacks else None


def _measure_cache():
    if _State.mcache is None:
        _State.mcache = _MeasureCache()
    return _State.mcache


def _ensure_measurement(geo, W, H, s):
    """(measurement, stale). Heavy layouts re-measure once a dragged slider pauses."""
    st = _State
    smatch = _stack_match(s)
    key = (st.geo_version, W, H, int(s.points), round(float(s.shift), 4),
           round(float(s.search_px), 4), bool(s.selected_only), bool(s.tile_border), bool(s.same_material),
           None if smatch is None else round(smatch, 6), st.sel_version if s.selected_only else -1)
    hit = st.meas_cache.pop(key, None)
    if hit is not None:
        st.meas_cache[key] = hit  # most recently used last
        st.meas, st.meas_key = hit, key
        return hit, False
    if st.meas is not None and st.meas_geo_version == st.geo_version and st.last_meas_ms > LIVE_MS:
        now = time.perf_counter()
        if key != st.pending_key:
            st.pending_key, st.pending_since = key, now
        wait = DEBOUNCE_S - (now - st.pending_since)
        if wait > 0.0:
            _schedule_redraw(wait + 0.01)
            return st.meas, True
    t0 = time.perf_counter()
    meas = _measure_grouped(geo, W, H, s.points, s.shift / 100.0, s.search_px, s.selected_only, s.tile_border,
                            s.same_material, _measure_cache(), smatch)
    st.last_meas_ms = (time.perf_counter() - t0) * 1000.0
    st.meas_version += 1
    meas.version = st.meas_version
    st.meas, st.meas_key = meas, key
    st.meas_geo_version = st.geo_version
    st.pending_key = None
    _cache_put(st.meas_cache, key, meas)
    return meas, False


def _current_stacks(geo, s, meas=None):
    """(stacks of the geometry or None, what they were found for): from the measurement when
    it is of these shells and settings, else found here (kept until either changes)."""
    st = _State
    smatch = _stack_match(s)
    if smatch is None or not geo.nshells:
        return None, None
    key = (st.geo_version, bool(s.same_material), round(smatch, 6))
    if meas is not None and meas.stacks is not None and meas.stacks.rep.size == geo.nshells \
            and st.meas_key is not None and (st.meas_key[0], st.meas_key[8], st.meas_key[9]) == key:
        return meas.stacks, key
    if st.stacks is None or st.stacks_key != key or st.stacks.rep.size != geo.nshells:
        st.stacks = _stacks_of(geo, bool(s.same_material), smatch, _measure_cache())
        st.stacks_key = key
    return st.stacks, key


def _home_counts(geo, meas=None, stacks=None):
    """What lies on something else in the 0-1 tile (UDIM 1001), which is where Move All but
    One clears up: (overlapping pairs with both shells there, stacked copies there). The
    panels show them next to the totals once shells lie on each other in other tiles too.
    A count is None when the measurement or the stacks aren't of these shells."""
    st = _State
    S = geo.nshells
    if not S or geo.tile.shape[0] != S:
        return None, None
    if meas is not None and meas.shells_total != S:
        meas = None
    if stacks is not None and stacks.rep.size != S:
        stacks = None
    key = (st.geo_version, meas.version if meas is not None else None,
           None if stacks is None else (st.stacks_key if stacks is st.stacks else 'measured'))
    if st.home is not None and st.home[0] == key:
        return st.home[1]
    home = (geo.tile[:, 0] == 0) & (geo.tile[:, 1] == 0)
    pairs = copies = None
    if meas is not None:
        codes = meas.overlap_codes
        pairs = int(np.count_nonzero(home[codes // S] & home[codes % S])) if codes.size else 0
    if stacks is not None:
        copies = int(np.count_nonzero(home[stacks.copies()]))
    st.home = (key, (pairs, copies))
    return pairs, copies


# ---------------------------------------------------------------------------
# Shell data for the panels, the material list and Low / High auto-fill
# ---------------------------------------------------------------------------
#
# Draw callbacks publish per-shell densities and materials here. Scene properties (the
# material list, auto-filled Low / High) must not be written while drawing, so those
# writes are queued for a timer.

class _Published:
    __slots__ = ("key", "mode", "objects", "td", "k", "mat", "mat_keys", "mat_total", "mat_vis", "stats",
                 "tmin", "tmax")


def _material_label(key):
    return key if key else NO_MATERIAL


def _published(key, mode, objects, td, k, mat, mat_keys, mat_total, mat_vis):
    """`td` in px/m for the density factor `k` (_td_factor) of the publishing view."""
    p = _Published()
    p.key, p.mode, p.objects = key, mode, objects
    p.td = np.asarray(td, dtype=np.float64)
    p.k = float(k)
    p.mat = np.asarray(mat, dtype=np.int64)
    p.mat_keys = list(mat_keys)
    p.mat_total = np.asarray(mat_total, dtype=np.int64)
    p.mat_vis = np.asarray(mat_vis, dtype=np.int64)
    nm = len(p.mat_keys)
    ok = np.isfinite(p.td)
    counts = np.bincount(p.mat, minlength=nm) if p.mat.size else np.zeros(nm, dtype=np.int64)
    lo = np.full(nm, np.inf)
    hi = np.full(nm, -np.inf)
    if ok.any():
        np.minimum.at(lo, p.mat[ok], p.td[ok])
        np.maximum.at(hi, p.mat[ok], p.td[ok])
    # per material: (shells shown, lowest density, highest density, faces, faces shown)
    p.stats = {k: (int(counts[i]), float(lo[i]) if np.isfinite(lo[i]) else None,
                   float(hi[i]) if np.isfinite(hi[i]) else None, int(p.mat_total[i]), int(p.mat_vis[i]))
               for i, k in enumerate(p.mat_keys)}
    p.tmin = float(p.td[ok].min()) if ok.any() else None
    p.tmax = float(p.td[ok].max()) if ok.any() else None
    return p


def _checked_keys(st):
    return tuple(sorted(it.key for it in st.materials if it.checked))


def _td_range_of(pub, checked):
    """(lowest, highest) density of the shells in the checked material sets (all shells when
    none is checked), or None."""
    if pub is None or not pub.td.size:
        return None
    mask = np.isfinite(pub.td)
    if checked:
        ids = [i for i, k in enumerate(pub.mat_keys) if k in set(checked)]
        mask &= np.isin(pub.mat, ids)
    if not mask.any():
        return None
    vals = pub.td[mask]
    return float(vals.min()), float(vals.max())


def _auto_key(scene, st, pub, checked):
    """What an auto-fill depends on, and the density factor it fills for. Densities are
    filled for the texture size used by the 3D view, the same for every editor, so two UV
    editors showing different images can't take turns refilling."""
    Wc, Hc, _src = _resolve_resolution_3d(st)
    kc = _td_factor(Wc, Hc, _unit_scale(scene))
    return (pub.mode, pub.objects, checked, round(kc, 6)), kc


def _auto_range(pub, checked, kc):
    """(low, high) of the checked sets for the density factor kc, or None."""
    rng = _td_range_of(pub, checked)
    if rng is None or not pub.k > 0.0:
        return None
    f = kc / pub.k
    return rng[0] * f, rng[1] * f


def _queue(scene, what, value):
    _State.pending[(scene.name, what)] = value
    try:
        if not bpy.app.timers.is_registered(_apply_pending):
            bpy.app.timers.register(_apply_pending, first_interval=0.0)
    except Exception:
        pass


def _sync_material_items(st, keys):
    """Make the material list show `keys`, keeping the checked state and the active row."""
    items = st.materials
    checked = {it.key for it in items if it.checked}
    active = items[st.material_index].key if 0 <= st.material_index < len(items) else None
    items.clear()
    for k in keys:
        it = items.add()
        it.key = k
        it.name = _material_label(k)
        if k in checked:
            it.checked = True
    st.material_index = keys.index(active) if active in keys else 0


def _apply_pending():
    """Timer: scene property writes queued by draw callbacks."""
    pending, _State.pending = _State.pending, {}
    for (scene_name, what), value in pending.items():
        scene = bpy.data.scenes.get(scene_name)
        st = getattr(scene, "uv_gap_overlay", None) if scene is not None else None
        if st is None:
            continue
        try:
            if what == "materials":
                if [it.key for it in st.materials] != list(value):
                    _sync_material_items(st, list(value))
            elif what == "fill":
                st.td_low_m, st.td_high_m = value
                if st.settings_version < 140:
                    st.settings_version = 140
        except Exception:
            _report("could not update the settings", traceback.format_exc())
    _redraw_areas(('IMAGE_EDITOR', 'VIEW_3D'))
    return None


def _after_publish(scene, st, pub):
    """Keep the material list in step with the shown objects, and auto-fill Low / High when
    the checked material sets, the objects, the texture size or the unit scale changed."""
    shown = [k for k, (_n, _lo, _hi, faces, _fv) in pub.stats.items() if faces > 0]
    shown.sort(key=lambda k: (k == "", _material_label(k).lower()))
    if [it.key for it in st.materials] != shown:
        _queue(scene, "materials", tuple(shown))
    if st.td_auto_range and (st.td_show or st.td_show_3d):  # only while the colors are shown
        checked = _checked_keys(st)
        fkey, kc = _auto_key(scene, st, pub, checked)
        if fkey != _State.auto_key:
            rng = _auto_range(pub, checked, kc)
            if rng is not None:
                _State.auto_key = fkey
                if abs(rng[0] - st.td_low_m) > 1e-6 * max(1.0, rng[0]) or \
                        abs(rng[1] - st.td_high_m) > 1e-6 * max(1.0, rng[1]):
                    _queue(scene, "fill", rng)


def _publish_edit(scene, st, geo, W, H, objects):
    """Publish the UV editor's shells (Edit Mode)."""
    scale = _unit_scale(scene)
    key = ('EDIT', _State.geo_version, W, H, scale)
    pub = _State.pub
    if pub is None or pub.key != key:
        td = _geo_td(geo, W, H, scale) if geo.td_nshells else np.full(geo.sh_n, np.nan)
        pub = _State.pub = _published(key, 'EDIT', tuple(sorted(o.name_full for o in objects)), td,
                                      _td_factor(W, H, scale), geo.sh_mat, geo.mat_keys, geo.mat_total,
                                      geo.mat_vis)
    _after_publish(scene, st, pub)
    return pub


def _panel_pub(context):
    """Published shells for a panel. In Edit Mode they are computed right here if no UV editor
    has done it yet (reading is safe from any draw callback); in Object Mode they come from
    the 3D view's texel density overlay."""
    scene = context.scene
    st = scene.uv_gap_overlay
    edit = context.mode == 'EDIT_MESH'
    if edit:
        objects = _edit_mesh_objects(context)
        if objects:
            try:
                geo, _stale = _uv_geometry(scene, st, objects)
                space = context.space_data
                if space is not None and space.type == 'IMAGE_EDITOR':
                    W, H, _src = _resolve_resolution(st, space)
                else:
                    W, H, _src = _resolve_resolution_3d(st)
                _publish_edit(scene, st, geo, W, H, objects)
            except Exception:
                _report("panel data", traceback.format_exc())
    pub = _State.pub
    return pub if pub is not None and pub.mode == ('EDIT' if edit else 'OBJECT') else None


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

_SHADER_CHOICES = {
    'line': ('POLYLINE_SMOOTH_COLOR', '3D_POLYLINE_SMOOTH_COLOR', 'SMOOTH_COLOR', '2D_SMOOTH_COLOR'),
    'fill': ('SMOOTH_COLOR', '2D_SMOOTH_COLOR', '3D_SMOOTH_COLOR'),
}


def _builtin(kind):
    known = _State.shader_names.get(kind)
    for name in ((known,) if known else _SHADER_CHOICES[kind]):
        try:
            shader = gpu.shader.from_builtin(name)
        except Exception:
            continue
        _State.shader_names[kind] = name
        return name, shader
    return "", None


def _batch_attrs(shader, prim, attrs):
    """Batch from float32 arrays (fast buffer path), falling back to Python lists."""
    if _State.numpy_buffers:
        try:
            return batch_for_shader(shader, prim, {k: np.ascontiguousarray(v, dtype=np.float32)
                                                   for k, v in attrs.items()})
        except Exception:
            _State.numpy_buffers = False
    return batch_for_shader(shader, prim, {k: np.asarray(v).tolist() for k, v in attrs.items()})


def _batch(shader, prim, pos, col):
    return _batch_attrs(shader, prim, {"pos": pos, "color": col})


# Texel density colors computed on the GPU from a per-vertex density and the thresholds as
# uniforms: changing Low / Needed / High, the texture size, the unit scale or the opacity
# rebuilds nothing. Same ramp as _td_rgb (red - yellow - green - cyan - blue, gray: no area).
_TD_VERT = """
void main()
{
  gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
  vec3 c = vec3(0.5);
  if (tdens >= 0.0) {
    float low = u_thr.x;
    float needed = u_thr.y;
    float high = u_thr.z;
    float td = tdens * u_thr.w;
    float t = 1.0;
    if (td < needed) {
      t = (needed > low) ? clamp((td - low) / (needed - low), 0.0, 1.0) : 0.0;
    }
    else if (td > needed) {
      t = (high > needed) ? 1.0 + clamp((td - needed) / (high - needed), 0.0, 1.0) : 2.0;
    }
    float h = 2.0 * t;
    c = clamp(vec3(abs(h - 3.0) - 1.0, 2.0 - abs(h - 2.0), 2.0 - abs(h - 4.0)), 0.0, 1.0);
  }
  v_color = vec4(c, u_alpha);
}
"""
_TD_FRAG = """
void main()
{
  frag_color = v_color;
}
"""


def _td_shader():
    """The texel density shader, or None where custom shaders are unavailable (per-vertex
    colors are used then). Created on first use, inside a draw callback."""
    sh = _State.td_shader
    if sh is None:
        sh = False
        try:
            iface = gpu.types.GPUStageInterfaceInfo("uvgap_td_iface")
            iface.smooth('VEC4', "v_color")
            info = gpu.types.GPUShaderCreateInfo()
            info.push_constant('MAT4', "ModelViewProjectionMatrix")
            info.push_constant('VEC4', "u_thr")  # low, needed, high (px/m), px/m per unit of `tdens`
            info.push_constant('FLOAT', "u_alpha")
            info.vertex_in(0, 'VEC3', "pos")
            info.vertex_in(1, 'FLOAT', "tdens")
            info.vertex_out(iface)
            info.fragment_out(0, 'VEC4', "frag_color")
            info.vertex_source(_TD_VERT)
            info.fragment_source(_TD_FRAG)
            sh = gpu.shader.create_from_info(info)
        except Exception:
            _report("texel density shader unavailable, using per-vertex colors", traceback.format_exc())
            sh = False
        _State.td_shader = sh
    return sh or None


def _td_shader_setup(sh, thr, k, alpha):
    sh.bind()
    sh.uniform_float("u_thr", (float(thr[0]), float(thr[1]), float(thr[2]), float(k)))
    sh.uniform_float("u_alpha", float(alpha))


def _set_mvp(sh):
    sh.uniform_float("ModelViewProjectionMatrix",
                     gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix())


def _segment_verts(P, Q):
    v = np.zeros((2 * P.shape[0], 3), dtype=np.float32)
    v[0::2, :2] = P
    v[1::2, :2] = Q
    return v


def _rect_verts(R):
    v = np.zeros((R.shape[0], 6, 3), dtype=np.float32)
    x0, y0, x1, y1 = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
    v[:, 0, 0], v[:, 0, 1] = x0, y0
    v[:, 1, 0], v[:, 1, 1] = x1, y0
    v[:, 2, 0], v[:, 2, 1] = x1, y1
    v[:, 3, 0], v[:, 3, 1] = x0, y0
    v[:, 4, 0], v[:, 4, 1] = x1, y1
    v[:, 5, 0], v[:, 5, 1] = x0, y1
    return v.reshape(-1, 3)


def _square_verts(C, h):
    return _rect_verts(np.column_stack((C[:, 0] - h, C[:, 1] - h, C[:, 0] + h, C[:, 1] + h)))


def _cross_verts(C, h):
    v = np.zeros((C.shape[0], 4, 3), dtype=np.float32)
    cx, cy = C[:, 0], C[:, 1]
    v[:, 0, 0], v[:, 0, 1] = cx - h, cy - h
    v[:, 1, 0], v[:, 1, 1] = cx + h, cy + h
    v[:, 2, 0], v[:, 2, 1] = cx - h, cy + h
    v[:, 3, 0], v[:, 3, 1] = cx + h, cy - h
    return v.reshape(-1, 3)


def _tick_verts(C, axis, h):
    """Short ticks along the tile edge at the foot points (the end cap of border lines)."""
    v = np.zeros((C.shape[0], 2, 3), dtype=np.float32)
    vert = axis == 0
    v[:, 0, 0] = np.where(vert, C[:, 0], C[:, 0] - h)
    v[:, 1, 0] = np.where(vert, C[:, 0], C[:, 0] + h)
    v[:, 0, 1] = np.where(vert, C[:, 1] - h, C[:, 1])
    v[:, 1, 1] = np.where(vert, C[:, 1] + h, C[:, 1])
    return v.reshape(-1, 3)


def _gradient(d, min_px, needed_px, c_zero, c_min, c_needed):
    """0 px -> c_zero, Minimal -> c_min, Needed and above -> c_needed."""
    lo = max(0.0, float(min_px))
    hi = max(lo + 1e-6, float(needed_px))
    t1 = np.clip(d / lo, 0.0, 1.0) if lo > 0.0 else np.ones_like(d)
    t2 = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    below = c_zero + (c_min - c_zero) * t1[:, None]
    above = c_min + (c_needed - c_min) * t2[:, None]
    return np.where((d < lo)[:, None], below, above)


def _fmt_px(d):
    return ("%.0f px" % d) if d >= 99.95 else ("%.1f px" % d)


def _view_mapping(region):
    """UV -> region pixels:  x = (u - u0) * sx,  y = (v - v0) * sy."""
    v2d = region.view2d
    u0, v0 = v2d.region_to_view(0.0, 0.0)
    u1, v1 = v2d.region_to_view(1000.0, 1000.0)
    du, dv = u1 - u0, v1 - v0
    if abs(du) < 1e-12 or abs(dv) < 1e-12:
        return None
    return u0, v0, 1000.0 / du, 1000.0 / dv


def _ui_scale(context):
    try:
        s = float(context.preferences.system.ui_scale)
    except Exception:
        s = 1.0
    return s if s > 0.0 else 1.0


def _in_region(C, rw, rh, margin):
    return ((C[:, 0] >= -margin) & (C[:, 0] <= rw + margin) &
            (C[:, 1] >= -margin) & (C[:, 1] <= rh + margin))


def _visible_segments(P, Q, rw, rh, margin):
    return ((np.maximum(P[:, 0], Q[:, 0]) >= -margin) & (np.minimum(P[:, 0], Q[:, 0]) <= rw + margin) &
            (np.maximum(P[:, 1], Q[:, 1]) >= -margin) & (np.minimum(P[:, 1], Q[:, 1]) <= rh + margin))


class _Labels:
    """Label boxes in priority order - one or more text lines, optionally an arrow (per-shell
    info blocks) - plus line colors; rebuilt only when the measurement or the style changes.

    Per label: anchor (UV), lifted, nlines, width (content, px), icon (UV direction, NaN: none),
    icon_rgb, icw (arrow column width: as tall as the block, at least iw). Per text line: text,
    rgb, line_of (label), row. th: text height, lh: line pitch."""
    __slots__ = ("anchor", "lifted", "nlines", "width", "icon", "icon_rgb", "icw",
                 "text", "rgb", "line_of", "row", "th", "lh", "iw", "gap_rgb", "border_rgb")


SCALE_RGB = (1.0, 0.62, 0.25)      # object scale note
STACK_RGB = (0.78, 0.78, 0.82)     # the "x N" line of a stack's info block
AXIS_RGB = {1: (0.35, 0.62, 1.0),  # orientation arrow: +Z (blue, as Blender's Z axis)
            2: (0.45, 0.88, 0.35)}  # +Y on flat-lying shells (green)


def _fmt_scale(s):
    """'2' for a uniform scale, '1 x 2 x 1' otherwise; None when the scale is 1."""
    if all(abs(c - 1.0) <= 1e-5 for c in s):
        return None
    parts = [("%.3f" % c).rstrip("0").rstrip(".") for c in s]
    parts = ["0" if p in ("-0", "") else p for p in parts]
    return parts[0] if parts[0] == parts[1] == parts[2] else " × ".join(parts)


# ---------------------------------------------------------------------------
# Texel density
# ---------------------------------------------------------------------------

def _td_values(uv_area, world_area, W, H, scale):
    """Texel density in px/m per shell: sqrt(texture pixels covered / square meters).
    NaN where a shell has no 3D area."""
    wa = np.asarray(world_area, dtype=np.float64) * (scale * scale)
    with np.errstate(divide='ignore', invalid='ignore'):
        td = np.sqrt(np.asarray(uv_area, dtype=np.float64) * (float(W) * float(H)) / wa)
    td[~np.isfinite(td)] = np.nan
    return td


def _td_base(uv_area, world_area):
    """sqrt(UV area / world area) per shell, the texture-independent part of the texel
    density: px/m = this * sqrt(W * H) / unit scale. -1 where a shell has no 3D area."""
    with np.errstate(divide='ignore', invalid='ignore'):
        b = np.sqrt(np.asarray(uv_area, dtype=np.float64) / np.asarray(world_area, dtype=np.float64))
    b[~np.isfinite(b)] = -1.0
    return b


def _td_factor(W, H, scale):
    return float(np.sqrt(float(W) * float(H))) / float(scale)


def _td_thresholds(st):
    """(low, needed, high) in px/m."""
    return float(st.td_low_m), float(st.td_needed_m), float(st.td_high_m)


def _td_range_bounds(td, f0, f1):
    """(lo, hi) in px/m: fractions f0..f1 of the lowest..highest finite density; None if none."""
    ok = td[np.isfinite(td)]
    if not ok.size:
        return None
    tmin, tmax = float(ok.min()), float(ok.max())
    span = tmax - tmin
    eps = max(span, abs(tmax), 1e-12) * 1e-9  # the shells at the ends are always in range
    return tmin + span * float(f0) - eps, tmin + span * float(f1) + eps


def _td_rgb(td, low, needed, high):
    """Red at Low and below, green at Needed, blue at High and above; the hue runs through
    yellow and cyan in between. Shells without a density are gray."""
    td = np.asarray(td, dtype=np.float64)
    t = np.ones(td.shape)
    with np.errstate(invalid='ignore', divide='ignore'):
        below = td < needed
        above = td > needed
        if needed > low:
            t[below] = np.clip((td[below] - low) / (needed - low), 0.0, 1.0)
        else:
            t[below] = 0.0
        if high > needed:
            t[above] = 1.0 + np.clip((td[above] - needed) / (high - needed), 0.0, 1.0)
        else:
            t[above] = 2.0
    h = t * 2.0  # hue in 60 degree steps: 0 red, 1 yellow, 2 green, 3 cyan, 4 blue
    rgb = np.column_stack((np.clip(np.abs(h - 3.0) - 1.0, 0.0, 1.0),
                           np.clip(2.0 - np.abs(h - 2.0), 0.0, 1.0),
                           np.clip(2.0 - np.abs(h - 4.0), 0.0, 1.0)))
    rgb[~np.isfinite(td)] = 0.5
    return rgb


def _fmt_td(v, unit):
    if v >= 99.95:
        return "%.0f %s" % (v, unit)
    if v >= 9.995:
        return "%.1f %s" % (v, unit)
    return "%.2f %s" % (v, unit)


def _td_stats(td, thr):
    """(shells, min, median, max, below low, above high) in px/m, for the panels."""
    ok = td[np.isfinite(td)]
    if not ok.size:
        return (int(td.size), None, None, None, 0, 0)
    low, _needed, high = thr
    return (int(td.size), float(ok.min()), float(np.median(ok)), float(ok.max()),
            int(np.count_nonzero(ok <= low)), int(np.count_nonzero(ok >= high)))


def _geo_td(geo, W, H, scale):
    key = (_State.geo_version, W, H, scale)
    td = _State.td_values.get(key)
    if td is None:
        td = _td_values(geo.td_uv_area, geo.td_world_area, W, H, scale)
        _cache_put(_State.td_values, key, td)
    return td


def _td_fill_batch(geo, td, thr, alpha):
    """Texel density fill of the UV editor, built in UV space once per change and drawn
    through the view matrix. Returns (batch, shader, uses_td_shader): with the texel density
    shader the batch holds densities and survives threshold / texture / opacity changes."""
    sh = _td_shader()
    if sh is not None:
        key = ('shader', _State.geo_version)
        hit = _State.td_fill.get(key)
        if hit is None:
            batch = None
            if geo.td_tri_uv.size:
                T = geo.td_tri_uv.reshape(-1, 2)
                pos = np.zeros((T.shape[0], 3), dtype=np.float32)
                pos[:, :2] = T
                dens = np.repeat(_td_base(geo.td_uv_area, geo.td_world_area)[geo.td_tri_shell], 3)
                batch = _batch_attrs(sh, 'TRIS', {"pos": pos, "tdens": dens})
            hit = (batch, sh, None)
            _cache_put(_State.td_fill, key, hit, size=2)
        return hit[0], hit[1], True
    key = (_State.geo_version, id(td), thr, round(alpha, 4))
    hit = _State.td_fill.get(key)
    if hit is None:
        _name, sh = _builtin('fill')
        batch = None
        if sh is not None and geo.td_tri_uv.size:
            T = geo.td_tri_uv.reshape(-1, 2)
            pos = np.zeros((T.shape[0], 3), dtype=np.float32)
            pos[:, :2] = T
            col = np.empty((T.shape[0], 4), dtype=np.float32)
            col[:, :3] = np.repeat(_td_rgb(td, *thr)[geo.td_tri_shell], 3, axis=0)
            col[:, 3] = alpha
            batch = _batch(sh, 'TRIS', pos, col)
        hit = (batch, sh, td)  # keeps td alive, so id(td) stays unique while cached
        _cache_put(_State.td_fill, key, hit, size=2)
    return hit[0], hit[1], False


def _build_labels(geo, meas, st, fsize, td_info, info=None, stacks=None, stack_lines=True):
    """`info`: (show object scale, show orientation arrows, show Flipped) for the per-shell info
    blocks; by default Flipped follows the gap overlay and the other two are off. `stacks`
    (_Stacks): a stack gets one block, on the shell standing for it - with `stack_lines`,
    telling the number of shells in it. What differs between a stack's shells shows in that
    block: the span of their densities ("256 - 512 px/m": copies of different size in the
    scene), "Scale varies"."""
    lab = _Labels()
    blf.size(FONT_ID, fsize)
    lab.th = th = float(blf.dimensions(FONT_ID, "0123456789")[1])
    lab.lh = th * 1.45
    lab.iw = th * 1.7
    c0 = np.array(st.color_zero[:], dtype=np.float64)
    c1 = np.array(st.color_min[:], dtype=np.float64)
    c2 = np.array(st.color_needed[:], dtype=np.float64)
    c_ov = np.array(st.color_overlap[:], dtype=np.float64)
    c_fl = tuple(float(c) for c in st.color_flipped[:])
    if meas is not None:
        lab.gap_rgb = _gradient(meas.dist, st.min_px, st.needed_px, c0, c1, c2)
        lab.border_rgb = _gradient(meas.bdist, st.border_min_px, st.border_needed_px, c0, c1, c2)
    else:
        lab.gap_rgb = lab.border_rgb = np.empty((0, 3))
    if info is None:
        info = (False, False, meas is not None and bool(st.show_flipped))
    show_scale, show_orient, show_flip = info

    # problems first (tile crossings, overlaps), then the shells' info blocks (those with a
    # warning first), then distances by severity
    anchors, lifted, icons, icon_rgb = [], [], [], []
    counts, texts, rgbs = [], [], []  # lines per label; text and color of every line, label by label

    def add(pts, text, rgb, lift):
        k = len(pts)
        if k:
            anchors.append(np.asarray(pts, dtype=np.float64).reshape(-1, 2))
            lifted.append(np.full(k, 1.0 if lift else 0.0))
            icons.append(np.full((k, 2), np.nan))
            icon_rgb.append(np.zeros((k, 3)))
            counts.append(np.ones(k, dtype=np.int64))
            texts.extend([text] * k if isinstance(text, str) else text)
            rgbs.append(np.tile(np.asarray(rgb, dtype=np.float64), (k, 1)) if np.ndim(rgb) == 1
                        else np.asarray(rgb, dtype=np.float64).reshape(-1, 3))

    if meas is not None:
        add(meas.tile_label_uv, "Crosses tile", c_ov, True)
        add(meas.pair_label_uv, "Overlap", c_ov, True)
        add(meas.ov_uv, "Overlap", c_ov, True)

    n = geo.sh_n
    if n:
        has = np.zeros(n, dtype=bool)
        warn = np.zeros(n, dtype=bool)
        td_ok = np.zeros(n, dtype=bool)
        td_txt = td_col = None
        if td_info is not None and len(td_info[0]) == n:
            td, unit, factor, thr = td_info
            td_ok = np.isfinite(td)
            ok = np.flatnonzero(td_ok)
            td_txt = dict(zip(ok.tolist(), [_fmt_td(v, unit) for v in (td[ok] * factor).tolist()]))
            td_col = dict(zip(ok.tolist(), map(tuple, (_td_rgb(td[ok], *thr) * 0.55 + 0.45).tolist())))
            has |= td_ok  # lighter colors: readable on the dark box
        scale_on = np.zeros(n, dtype=bool)
        otxt = []
        if show_scale and geo.obj_scale:
            otxt = [_fmt_scale(s) for s in geo.obj_scale]
            scale_on = np.asarray([t is not None for t in otxt], dtype=bool)[geo.sh_obj]
            has |= scale_on
            warn |= scale_on
        flip = np.zeros(n, dtype=bool)
        if show_flip and geo.flipped.size and geo.gap_shell.size == geo.flipped.size:
            flip[geo.gap_shell] = geo.flipped
            has |= flip
            warn |= flip
        arrow = (geo.sh_axis > 0) if show_orient and geo.sh_axis.size == n else np.zeros(n, dtype=bool)
        has |= arrow
        stack_txt = {}
        scale_varies = np.zeros(n, dtype=bool)
        if stacks is not None and stacks.rep.size == geo.nshells and geo.gap_shell.size == geo.nshells:
            copies = geo.gap_shell[stacks.copies()]
            mem = geo.gap_shell                    # every gap shell, and the shell standing for it
            rep_of = geo.gap_shell[stacks.rep]
            stacked = stacks.size[stacks.rep] > 1
            if td_txt is not None and stacked.any():
                # the densities of a stack's shells: one number when they agree, else their span
                lo, hi = np.full(n, np.inf), np.full(n, -np.inf)
                m = stacked & td_ok[mem]
                np.minimum.at(lo, rep_of[m], td[mem[m]])
                np.maximum.at(hi, rep_of[m], td[mem[m]])
                reps = np.flatnonzero(np.isfinite(lo))
                cols = (_td_rgb(lo[reps], *thr) * 0.55 + 0.45).tolist()
                for a, col, l, h in zip(reps.tolist(), cols, lo[reps].tolist(), hi[reps].tolist()):
                    if h > l * 1.005:
                        td_txt[a] = "%s – %s" % (_fmt_td(l * factor, "").strip(), _fmt_td(h * factor, unit))
                        warn[a] = True
                    else:
                        td_txt[a] = _fmt_td(l * factor, unit)
                    td_col[a] = tuple(col)
                    td_ok[a] = has[a] = True
            if show_scale and otxt and stacked.any():
                codes = {}
                code = np.asarray([codes.setdefault(t, len(codes)) for t in otxt], dtype=np.int64)[geo.sh_obj]
                lo_c, hi_c = np.full(n, len(codes), dtype=np.int64), np.full(n, -1, dtype=np.int64)
                np.minimum.at(lo_c, rep_of[stacked], code[mem[stacked]])
                np.maximum.at(hi_c, rep_of[stacked], code[mem[stacked]])
                scale_varies = hi_c > lo_c
                scale_varies[copies] = False
                has |= scale_varies
                warn |= scale_varies
            has[copies] = False  # (they lie under the shell standing for their stack)
            warn[copies] = False
            u = stacks.unique()
            u = u[stacks.size[u] > 1] if stack_lines else u[:0]
            for g, k, nf in zip(u.tolist(), stacks.size[u].tolist(), stacks.nflip[u].tolist()):
                a = int(geo.gap_shell[g])
                # (a flipped stack says Flipped itself: all its shells are)
                stack_txt[a] = "× %d" % k if (nf == 0 or nf == k) else "× %d (%d flipped)" % (k, nf)
            if stack_txt:
                has[np.fromiter(stack_txt, dtype=np.int64, count=len(stack_txt))] = True
        order = np.concatenate((np.flatnonzero(has & warn), np.flatnonzero(has & ~warn)))
        if order.size:
            obj = geo.sh_obj.tolist()
            block, cols = [], []
            for i in order.tolist():
                n0 = len(texts)
                if td_ok[i]:
                    texts.append(td_txt[i])
                    cols.append(td_col[i])
                if i in stack_txt:
                    texts.append(stack_txt[i])
                    cols.append(STACK_RGB)
                if scale_varies[i]:
                    texts.append("Scale varies")
                    cols.append(SCALE_RGB)
                elif scale_on[i]:
                    texts.append("Scale " + otxt[obj[i]])
                    cols.append(SCALE_RGB)
                if flip[i]:
                    texts.append("Flipped")
                    cols.append(c_fl)
                block.append(len(texts) - n0)
            counts.append(np.asarray(block, dtype=np.int64))
            rgbs.append(np.asarray(cols, dtype=np.float64).reshape(-1, 3))
            k = order.size
            anchors.append(geo.sh_anchor[order])
            lifted.append(np.zeros(k))
            ic = np.full((k, 2), np.nan)
            icr = np.zeros((k, 3))
            a = arrow[order]
            if a.any():
                ic[a] = geo.sh_orient[order][a]
                icr[a] = np.asarray([AXIS_RGB[c] for c in geo.sh_axis[order][a].tolist()])
            icons.append(ic)
            icon_rgb.append(icr)

    if meas is not None:
        d_all = np.concatenate((meas.dist, meas.bdist))
        if d_all.size:
            severity = np.concatenate((meas.dist / max(float(st.needed_px), 1e-6),
                                       meas.bdist / max(float(st.border_needed_px), 1e-6)))
            order = np.argsort(severity, kind='stable')
            mids = np.concatenate(((meas.p_uv + meas.q_uv) * 0.5, (meas.bp_uv + meas.bf_uv) * 0.5))
            add(mids[order], [_fmt_px(x) for x in d_all[order].tolist()],
                np.concatenate((lab.gap_rgb, lab.border_rgb))[order], False)

    counts = np.concatenate(counts) if counts else _EMPTY_I
    nlab = int(counts.size)
    lab.anchor = np.concatenate(anchors) if anchors else _EMPTY2
    lab.lifted = np.concatenate(lifted) if lifted else _EMPTY_F
    lab.icon = np.concatenate(icons) if icons else _EMPTY2
    lab.icon_rgb = np.concatenate(icon_rgb) if icon_rgb else np.empty((0, 3))
    lab.nlines = counts
    lab.text = texts
    lab.rgb = np.concatenate(rgbs) if rgbs else np.empty((0, 3))
    first = np.cumsum(counts) - counts  # the first line of each label
    lab.line_of = np.repeat(np.arange(nlab, dtype=np.int64), counts)
    lab.row = np.arange(len(texts), dtype=np.int64) - np.repeat(first, counts)
    widths = {t: float(blf.dimensions(FONT_ID, t)[0]) for t in set(texts)}
    maxw = np.zeros(nlab)
    if texts:
        some = counts > 0  # (a block can be an arrow alone)
        maxw[some] = np.maximum.reduceat(np.asarray([widths[t] for t in texts], dtype=np.float64), first[some])
    has_icon = np.isfinite(lab.icon[:, 0]) if nlab else np.zeros(0, dtype=bool)
    tall = th + (np.maximum(counts, 1) - 1) * lab.lh
    lab.icw = np.where(has_icon, np.maximum(lab.iw, tall), 0.0)
    lab.width = maxw + lab.icw + np.where(has_icon & (counts > 0), th * 0.35, 0.0)
    return lab


def _clashes(grid, gx0, gx1, gy0, gy1, x0, y0, x1, y1):
    for gx in range(gx0, gx1 + 1):
        for gy in range(gy0, gy1 + 1):
            for r in grid.get((gx, gy), ()):
                if x0 < r[2] and x1 > r[0] and y0 < r[3] and y1 > r[1]:
                    return True
    return False


COARSE_OVER = 1500  # with more labels in view, only the first one of each label-sized cell is tried


def _place_labels(lab, anchor_px, lift, pad, rw, rh, declutter, coarse_over=COARSE_OVER):
    """Labels to draw (indices, priority order) and, per label, its box, the left end of its
    text, its first baseline and its center. Decluttering keeps a label only when it doesn't
    cover one placed before it; with very many labels in view (`coarse_over`) a first pass
    keeps one label per label-sized screen cell, which bounds the work."""
    ax = anchor_px[:, 0]
    ay = anchor_px[:, 1] + lab.lifted * lift
    hw = lab.width * 0.5
    th = lab.th
    tall = th + (np.maximum(lab.nlines, 1) - 1) * lab.lh  # first line's top to last line's baseline
    x0, x1 = ax - hw - pad, ax + hw + pad
    y0, y1 = ay - tall * 0.5 - th * 0.3 - pad * 0.5, ay + tall * 0.5 + pad
    idx = np.flatnonzero((x1 >= 0.0) & (x0 <= rw) & (y1 >= 0.0) & (y0 <= rh))
    if declutter and idx.size > 1:
        cw = max(1.0, float(np.median(x1[idx] - x0[idx])))
        ch = max(1.0, float(np.median(y1[idx] - y0[idx])))
        if idx.size > coarse_over:
            # coarse pass: the first label (by priority) in each label-sized screen cell
            cell_key = ((np.floor(ax[idx] / cw).astype(np.int64) + _KEY_OFF) * _KEY_MUL +
                        np.floor(ay[idx] / ch).astype(np.int64) + _KEY_OFF)
            _cells, first = np.unique(cell_key, return_index=True)
            idx = idx[np.sort(first)]
        # exact pass (over the survivors)
        cell = 2.0 * max(cw, ch)
        grid = {}
        keep = []
        X0, Y0, X1, Y1 = x0[idx].tolist(), y0[idx].tolist(), x1[idx].tolist(), y1[idx].tolist()
        for k, i in enumerate(idx.tolist()):
            a0, b0, a1, b1 = X0[k], Y0[k], X1[k], Y1[k]
            gx0, gx1, gy0, gy1 = int(a0 // cell), int(a1 // cell), int(b0 // cell), int(b1 // cell)
            if _clashes(grid, gx0, gx1, gy0, gy1, a0, b0, a1, b1):
                continue
            rect = (a0, b0, a1, b1)
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    grid.setdefault((gx, gy), []).append(rect)
            keep.append(i)
        idx = np.asarray(keep, dtype=np.int64)
    tx = ax - hw + np.where(np.isfinite(lab.icon[:, 0]), lab.icw + th * 0.35, 0.0)
    ty = ay + tall * 0.5 - th
    return idx, np.column_stack((x0, y0, x1, y1)), tx, ty, ax, ay


def _draw_texts(lab, idx, tx, ty, fsize, alpha, shadow):
    placed = np.zeros(lab.nlines.size, dtype=bool)
    placed[idx] = True
    li = np.flatnonzero(placed[lab.line_of]) if lab.line_of.size else _EMPTY_I
    if not li.size:
        return
    blf.size(FONT_ID, fsize)
    if shadow:
        blf.enable(FONT_ID, blf.SHADOW)
        blf.shadow(FONT_ID, 3, 0.0, 0.0, 0.0, min(1.0, alpha))
        blf.shadow_offset(FONT_ID, 1, -1)
    try:
        owner = lab.line_of[li]
        xs = tx[owner].tolist()
        ys = (ty[owner] - lab.row[li] * lab.lh).tolist()
        cols = lab.rgb[li].tolist()
        text = lab.text
        for k, i in enumerate(li.tolist()):
            r, g, b = cols[k]
            blf.color(FONT_ID, r, g, b, alpha)
            blf.position(FONT_ID, xs[k], ys[k], 0.0)
            blf.draw(FONT_ID, text[i])
    finally:
        if shadow:
            blf.disable(FONT_ID, blf.SHADOW)


def _arrow_verts(C, D, size):
    """Triangles (shaft + head) of arrows centered on C, pointing along the unit vectors D,
    `size` long (one value or one per arrow)."""
    size = np.broadcast_to(np.asarray(size, dtype=np.float64), (C.shape[0],))[:, None]
    N = np.column_stack((-D[:, 1], D[:, 0]))
    tip = C + D * (size * 0.5)
    neck = tip - D * (size * 0.45)
    tail = C - D * (size * 0.5)
    sw = N * np.maximum(0.8, size * 0.08)
    hw = N * (size * 0.3)
    pts = np.stack((tail + sw, tail - sw, neck - sw, tail + sw, neck - sw, neck + sw,
                    neck + hw, neck - hw, tip), axis=1)
    v = np.zeros((pts.shape[0] * 9, 3), dtype=np.float32)
    v[:, :2] = pts.reshape(-1, 2)
    return v


class _Frame:
    """Everything drawn in region pixels for one view of the data: reused as long as the
    view (pan / zoom / size), the data and the style stay the same."""
    __slots__ = ("hl", "under", "lines", "fills", "boxes", "icons", "idx", "tx", "ty")


def _frame_data(st, geo, meas, lab, mapping, rw, rh, alpha, dot, xh, lift, margin, pad, hl=None, stacks=None):
    u0, v0, sx, sy = mapping

    def to_region(uv):
        return np.column_stack(((uv[:, 0] - u0) * sx, (uv[:, 1] - v0) * sy))

    hl_p = []
    under_p, under_c = [], []
    lines_p, lines_c, fills_p, fills_c = [], [], [], []

    # shells inside the texel density range: an outline under everything else
    if hl is not None and hl.any() and geo.seg_shell.size:
        seg = hl[geo.seg_shell]
        P, Q = to_region(geo.seg_a[seg]), to_region(geo.seg_b[seg])
        vis = _visible_segments(P, Q, rw, rh, margin)
        if vis.any():
            hl_p.append(_segment_verts(P[vis], Q[vis]))

    if meas is not None:
        ov_rgba = np.array(tuple(st.color_overlap) + (alpha,), dtype=np.float32)
        # flipped shells: tinted fill underneath the lines, outline on top
        # (a mirrored copy lying in a stack with an unflipped shell isn't shown as flipped)
        flipped = geo.flipped
        if stacks is not None and stacks.rep.size == geo.nshells:
            flipped = flipped & (stacks.rep == np.arange(geo.nshells))
        if st.show_flipped and flipped.any():
            fl_rgb = tuple(st.color_flipped)
            tris = geo.flip_tris[flipped[geo.flip_tri_shell]] if geo.flip_tris.size else geo.flip_tris
            if tris.size:
                T = to_region(tris.reshape(-1, 2)).reshape(-1, 3, 2)
                vis = ((T[:, :, 0].max(axis=1) >= 0.0) & (T[:, :, 0].min(axis=1) <= rw) &
                       (T[:, :, 1].max(axis=1) >= 0.0) & (T[:, :, 1].min(axis=1) <= rh))
                if vis.any():
                    T = T[vis].reshape(-1, 2)
                    v = np.zeros((T.shape[0], 3), dtype=np.float32)
                    v[:, :2] = T
                    under_p.append(v)
                    under_c.append(np.tile(np.array(fl_rgb + (0.22 * alpha,), dtype=np.float32), (T.shape[0], 1)))
            seg = flipped[geo.seg_shell]
            P, Q = to_region(geo.seg_a[seg]), to_region(geo.seg_b[seg])
            vis = _visible_segments(P, Q, rw, rh, margin)
            if vis.any():
                lines_p.append(_segment_verts(P[vis], Q[vis]))
                lines_c.append(np.tile(np.array(fl_rgb + (alpha,), dtype=np.float32), (2 * int(vis.sum()), 1)))

        # distances: gradient line from the border point + a marker on the border;
        # tile-border lines end in a tick along the tile edge
        for p_uv, q_uv, rgb, axis in ((meas.p_uv, meas.q_uv, lab.gap_rgb, None),
                                      (meas.bp_uv, meas.bf_uv, lab.border_rgb, meas.baxis)):
            if not p_uv.size:
                continue
            P, Q = to_region(p_uv), to_region(q_uv)
            vis = _visible_segments(P, Q, rw, rh, margin)
            if not vis.any():
                continue
            P, Q = P[vis], Q[vis]
            rgba = np.empty((P.shape[0], 4), dtype=np.float32)
            rgba[:, :3] = rgb[vis]
            rgba[:, 3] = alpha
            lines_p.append(_segment_verts(P, Q))
            lines_c.append(np.repeat(rgba, 2, axis=0))
            fills_p.append(_square_verts(P, dot))
            fills_c.append(np.repeat(rgba, 6, axis=0))
            if axis is not None:
                lines_p.append(_tick_verts(Q, axis[vis], xh))
                lines_c.append(np.repeat(rgba, 2, axis=0))

        # problems: crosses on points inside another shell, where overlapping borders
        # cross, and where shells cross a tile line
        for uv in (meas.ov_uv, meas.cross_uv, meas.tile_mark_uv):
            if uv.size:
                X = to_region(uv)
                X = X[_in_region(X, rw, rh, margin)]
                if X.size:
                    lines_p.append(_cross_verts(X, xh))
                    lines_c.append(np.tile(ov_rgba, (4 * X.shape[0], 1)))

    fr = _Frame()
    fr.idx, fr.tx, fr.ty = _EMPTY_I, _EMPTY_F, _EMPTY_F
    boxes = ax = ay = None
    if lab.nlines.size:
        fr.idx, boxes, fr.tx, fr.ty, ax, ay = _place_labels(lab, to_region(lab.anchor), lift, pad, rw, rh,
                                                            st.declutter)
    _lname, lsh = _builtin('line')
    _fname, fsh = _builtin('fill')
    fr.hl = None
    if hl_p and lsh is not None:
        HL = np.concatenate(hl_p)
        fr.hl = _batch(lsh, 'LINES', HL, np.tile(np.array((1.0, 1.0, 1.0, 0.9 * alpha), dtype=np.float32),
                                                 (HL.shape[0], 1)))
    fr.under = (_batch(fsh, 'TRIS', np.concatenate(under_p), np.concatenate(under_c))
                if under_p and fsh is not None else None)
    fr.lines = (_batch(lsh, 'LINES', np.concatenate(lines_p), np.concatenate(lines_c))
                if lines_p and lsh is not None else None)
    fr.fills = (_batch(fsh, 'TRIS', np.concatenate(fills_p), np.concatenate(fills_c))
                if fills_p and fsh is not None else None)
    fr.boxes = fr.icons = None
    if fr.idx.size and st.label_background and fsh is not None:
        bg = np.array((0.0, 0.0, 0.0, 0.85 * alpha), dtype=np.float32)
        fr.boxes = _batch(fsh, 'TRIS', _rect_verts(boxes[fr.idx]), np.tile(bg, (6 * fr.idx.size, 1)))
    if fr.idx.size and fsh is not None:
        sel = fr.idx[np.isfinite(lab.icon[fr.idx, 0])]
        if sel.size:
            D = np.column_stack((lab.icon[sel, 0] * sx, lab.icon[sel, 1] * sy))
            dn = np.hypot(D[:, 0], D[:, 1])
            good = dn > 0.0
            sel, D, dn = sel[good], D[good], dn[good]
            if sel.size:
                C = np.column_stack((ax[sel] - (lab.width[sel] - lab.icw[sel]) * 0.5, ay[sel]))
                rgba = np.empty((sel.size, 4), dtype=np.float32)
                rgba[:, :3] = lab.icon_rgb[sel]
                rgba[:, 3] = alpha
                fr.icons = _batch(fsh, 'TRIS', _arrow_verts(C, D / dn[:, None], lab.icw[sel] * 0.9),
                                  np.repeat(rgba, 9, axis=0))
    return fr


def _draw_batch(kind, batch, width=None, rw=None, rh=None):
    name, sh = _builtin(kind)
    if sh is None or batch is None:
        return
    sh.bind()
    polyline = 'POLYLINE' in name
    if width is not None:
        if polyline:
            sh.uniform_float("viewportSize", (rw, rh))
            sh.uniform_float("lineWidth", width)
        else:
            gpu.state.line_width_set(width)
    batch.draw(sh)
    if width is not None and not polyline:
        gpu.state.line_width_set(1.0)


def _draw_overlay(context, region, st, geo, meas, W, H, stale, stacks=None, stacks_key=None):
    mapping = _view_mapping(region)
    if mapping is None:
        return
    u0, v0, sx, sy = mapping
    rw, rh = float(region.width), float(region.height)
    ui = _ui_scale(context)
    alpha = max(0.0, min(1.0, float(st.opacity))) * (0.35 if stale else 1.0)
    if alpha <= 0.0:
        return
    lw = max(1.0, float(st.line_width) * ui)
    dot = lw * 0.5 + 1.5 * ui
    xh = 4.0 * ui + lw * 0.5
    fsize = max(4.0, float(st.font_size) * ui)
    lift = xh + fsize * 0.9
    margin = 64.0 * ui
    pad = max(2.0, round(3.0 * ui))

    scale = _unit_scale(context.scene)
    td = _geo_td(geo, W, H, scale) if geo.td_nshells else None
    td_info = td_key = None
    if st.td_show and td is not None:
        thr = _td_thresholds(st)
        td_info = (td, _TD_LABEL.get(st.td_unit, "px/m"), _TD_FACTOR.get(st.td_unit, 1.0), thr)
        td_key = (W, H, scale, st.td_unit, thr)
        _State.td_uv_stats = _td_stats(td, thr)
    info = (bool(st.info_show and st.show_scale), bool(st.info_show and st.show_orientation),
            bool(meas is not None and st.show_flipped))

    hl = hl_key = None
    if (st.td_range_highlight and td is not None and geo.gap_shell.size
            and (st.td_sel_from > 0.0 or st.td_sel_to < 100.0)):
        bounds = _td_range_bounds(td, st.td_sel_from / 100.0, st.td_sel_to / 100.0)
        if bounds is not None:
            with np.errstate(invalid='ignore'):
                inside = (td >= bounds[0]) & (td <= bounds[1])
            hl = inside[geo.gap_shell]
            hl_key = (bounds, W, H, scale)

    key = (meas.version if meas is not None else None, _State.geo_version, round(fsize, 3),
           float(st.min_px), float(st.needed_px),
           float(st.border_min_px), float(st.border_needed_px),
           tuple(st.color_zero), tuple(st.color_min), tuple(st.color_needed),
           tuple(st.color_overlap), tuple(st.color_flipped), td_key, info, stacks_key, bool(st.info_show))
    lab = _State.labels_cache.get(key)
    if lab is None:
        lab = _build_labels(geo, meas, st, fsize, td_info, info, stacks, bool(st.info_show))
        _cache_put(_State.labels_cache, key, lab)

    fkey = (mapping, rw, rh, key, alpha, lw, dot, xh, lift, margin, pad,
            bool(st.declutter), bool(st.label_background), bool(st.show_flipped), hl_key)
    rkey = region.as_pointer() if hasattr(region, "as_pointer") else id(region)
    hit = _State.frame.get(rkey)
    if hit is not None and hit[0] == fkey:
        fr = hit[1]
    else:
        fr = _frame_data(st, geo, meas, lab, mapping, rw, rh, alpha, dot, xh, lift, margin, pad, hl, stacks)
        _cache_put(_State.frame, rkey, (fkey, fr), size=8)

    gpu.state.blend_set('ALPHA')
    try:
        if td_info is not None:
            fill_alpha = max(0.0, min(1.0, float(st.td_fill_opacity))) * (0.35 if stale else 1.0)
            batch, sh, shader_td = _td_fill_batch(geo, td_info[0], td_info[3], fill_alpha)
            if batch is not None:
                with gpu.matrix.push_pop():  # UV -> region pixels
                    gpu.matrix.multiply_matrix(Matrix(((sx, 0.0, 0.0, -u0 * sx), (0.0, sy, 0.0, -v0 * sy),
                                                       (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))))
                    if shader_td:
                        _td_shader_setup(sh, td_info[3], _td_factor(W, H, scale), fill_alpha)
                        _set_mvp(sh)
                    else:
                        sh.bind()
                    batch.draw(sh)
        _draw_batch('line', fr.hl, lw + 2.0 * ui, rw, rh)
        _draw_batch('fill', fr.under)
        _draw_batch('line', fr.lines, lw, rw, rh)
        _draw_batch('fill', fr.fills)
        if fr.idx.size:
            _draw_batch('fill', fr.boxes)
            _draw_batch('fill', fr.icons)
            _draw_texts(lab, fr.idx, fr.tx, fr.ty, fsize, alpha, shadow=not st.label_background)
    finally:
        gpu.state.blend_set('NONE')


def _uv_features_on(st):
    """Anything to draw in the UV editor. Scale notes only join blocks shown for the gap or
    texel density overlays; arrows can be shown on their own."""
    return bool(st.show or st.td_show or (st.info_show and st.show_orientation))


def _draw_main(context):
    scene = getattr(context, "scene", None)
    st = getattr(scene, "uv_gap_overlay", None) if scene is not None else None
    if st is None or not _uv_features_on(st):
        return
    space = context.space_data
    if space is None or space.type != 'IMAGE_EDITOR' or getattr(space, "mode", None) != 'UV':
        return
    overlay = getattr(space, "overlay", None)
    if overlay is not None and not getattr(overlay, "show_overlays", True):
        return
    region = context.region
    if region is None or region.type != 'WINDOW':
        return
    objects = _edit_mesh_objects(context)
    if not objects:
        return
    # a redraw with the view as it was last time isn't panning or zooming: time to look for
    # what Blender doesn't announce (see _ensure_geometry)
    rkey = region.as_pointer() if hasattr(region, "as_pointer") else id(region)
    view = (_view_mapping(region), region.width, region.height)
    still = _State.view_key.get(rkey) == view
    _cache_put(_State.view_key, rkey, view, size=8)
    geo, stale_geo = _uv_geometry(scene, st, objects, poll=still)
    W, H, _src = _resolve_resolution(st, space)
    _publish_edit(scene, st, geo, W, H, objects)
    if geo.nshells == 0 and geo.sh_n == 0:
        return
    meas, stale_meas = None, False
    if st.show and geo.nshells:
        meas, stale_meas = _ensure_measurement(geo, W, H, st)
    stacks, stacks_key = _current_stacks(geo, st, meas)
    _draw_overlay(context, region, st, geo, meas, W, H, stale_geo or stale_meas, stacks, stacks_key)


def _draw_callback():
    try:
        _draw_main(bpy.context)
    except Exception:
        err = traceback.format_exc()
        if err != _State.last_error:  # report each distinct error once, not every redraw
            _State.last_error = err
            print("[UV Shell Gap Overlay] draw error:\n" + err)


# ---------------------------------------------------------------------------
# Texel density in the 3D viewport
# ---------------------------------------------------------------------------

TD3D_DEPTH_BIAS = 2.0e-5  # NDC depth offset toward the viewer, so the colors don't z-fight the mesh


class _TD3D:
    """Texel density data of one object for the 3D viewport (object space)."""
    __slots__ = ("gen", "build_ms", "last_used", "nshell", "uv_area", "face_shell", "face_nvec",
                 "tri_co", "tri_shell", "mat_keys", "shell_mat", "mat_faces",
                 "m3", "stretch", "base", "version", "color_key", "batch", "td", "td_k")


def _td3d_build(obj, depsgraph, edit):
    if edit:  # the edit cage, as shared with the UV editor; hidden faces are not drawn
        A = _object_arrays(obj, True, need_sel=SEL_NONE)
        if A is None:
            return None
        vis = ~A.f_hide
        keys, lut = _material_sets(obj)
    else:  # the evaluated mesh, as drawn (modifiers included)
        ob_eval = obj.evaluated_get(depsgraph)
        me = ob_eval.data
        uvl = me.uv_layers.active if me is not None else None
        if uvl is None:
            return None
        A = _arrays_from_mesh(me, uvl.name, True)
        vis = np.ones(A.f_total.size, dtype=bool)
        keys, lut = _material_sets(ob_eval)
    S = _shells(A, vis)
    e = _TD3D()
    e.nshell = S.nshell
    e.face_shell = S.face_shell
    uv_abs = np.abs(S.area2) * 0.5
    e.uv_area = np.bincount(S.face_shell, weights=uv_abs, minlength=S.nshell)
    e.face_nvec = _face_area_vectors(S, A)
    cmap = np.full(A.f_total.size, -1, dtype=np.int64)
    cmap[S.fidx] = np.arange(S.fidx.size)
    tc = cmap[A.tri_face]
    keep = tc >= 0
    e.tri_co = A.co[A.loop_vert[A.tri_loops[keep]]].reshape(-1, 3).astype(np.float32)
    e.tri_shell = S.face_shell[tc[keep]]
    fm = lut[np.clip(A.f_mat[S.fidx], 0, lut.size - 1)]
    e.mat_keys = keys
    e.shell_mat = _dominant(S.face_shell, fm, uv_abs, S.nshell)
    e.mat_faces = np.bincount(fm, minlength=len(keys))
    e.m3 = e.stretch = e.base = e.color_key = e.batch = e.td = e.td_k = None
    e.version = 0
    return e


def _td3d_entry(obj, depsgraph, edit, may_build=True):
    """(entry, stale, built, deferred) for one object. Heavy rebuilds wait until edits pause;
    `may_build` False (frame budget used up) defers a needed build to the next frame."""
    key = obj.as_pointer()
    mptr = obj.data.as_pointer()
    if edit:  # (the colors don't depend on what is selected)
        gen = ('E', mptr, _State.epoch) + _obj_gens(obj)[0]
    else:
        gen = ('O', mptr, obj.name_full, _State.id_gen.get(key, 0), _State.id_gen.get(mptr, 0),
               tuple(_material_keys(obj)))  # a renamed material sends no depsgraph update
    e = _State.td3d.get(key)
    if e is not None and e.gen == gen:
        return e, False, False, False
    if e is not None and e.build_ms > LIVE_MS:
        wait = DEBOUNCE_S - (time.perf_counter() - _State.last_change)
        if wait > 0.0:
            _schedule_redraw(wait + 0.01)
            return e, True, False, False
    if not may_build:
        return e, e is not None, False, True
    t0 = time.perf_counter()
    e = _td3d_build(obj, depsgraph, edit)
    if e is None:
        _State.td3d.pop(key, None)
        return None, False, True, False
    e.gen = gen
    e.build_ms = (time.perf_counter() - t0) * 1000.0
    e.last_used = time.perf_counter()
    _State.td3d[key] = e
    return e, False, True, False


def _td3d_update(e, M3):
    """Per-shell density base (_td_base) for the object's current stretch: only its scale and
    shear matter (M^T M), so moving or rotating the object recomputes nothing."""
    G = M3.T @ M3
    s = float(np.abs(G).max())
    # to 1e-5: object matrices are single precision, a rotation alone changes G by ~1e-7
    stretch = (np.rint(G * (1e5 / s)).astype(np.int64).tobytes(), float("%.5g" % s)) if s > 0.0 else None
    if e.stretch != stretch or e.base is None:
        world = _world_face_areas(e.face_nvec, M3)
        e.base = _td_base(e.uv_area, np.bincount(e.face_shell, weights=world, minlength=e.nshell))
        e.stretch = stretch
        e.version += 1
        e.batch = e.color_key = e.td = None


def _td3d_values(e, k):
    """Texel density in px/m per shell (NaN: no 3D area)."""
    if e.td is None or e.td_k != k:
        e.td = np.where(e.base >= 0.0, e.base * k, np.nan)
        e.td_k = k
    return e.td


def _td3d_batch(e, sh, shader_td, W, H, scale, thr, alpha):
    """The object's batch: densities for the texel density shader (built once per geometry /
    stretch), or per-vertex colors (rebuilt when the settings change)."""
    if shader_td:
        if e.batch is None or e.color_key != 'shader':
            e.batch = None
            if e.tri_shell.size:
                e.batch = _batch_attrs(sh, 'TRIS', {"pos": e.tri_co, "tdens": np.repeat(e.base[e.tri_shell], 3)})
            e.color_key = 'shader'
        return e.batch
    ckey = (e.version, W, H, scale, thr, round(alpha, 4))
    if e.color_key != ckey:
        td = _td3d_values(e, _td_factor(W, H, scale))
        e.batch = None
        if sh is not None and e.tri_shell.size:
            col = np.empty((e.tri_shell.size * 3, 4), dtype=np.float32)
            col[:, :3] = np.repeat(_td_rgb(td, *thr)[e.tri_shell], 3, axis=0)
            col[:, 3] = alpha
            e.batch = _batch(sh, 'TRIS', e.tri_co, col)
        e.color_key = ckey
    return e.batch


def _visible_in(obj, space):
    try:
        return obj.visible_get(viewport=space)
    except Exception:
        return obj.visible_get()


def _td3d_objects(context, space):
    if context.mode == 'EDIT_MESH':
        return _edit_mesh_objects(context), True
    objs = [o for o in (getattr(context, "selected_objects", None) or ())
            if o.type == 'MESH' and o.data is not None and _visible_in(o, space)]
    return objs, False


def _publish_3d(scene, st, drawn, k):
    """Publish the 3D view's shells (Object Mode) for the panels and Low / High auto-fill."""
    key = ('OBJECT', tuple((e.gen, e.version) for _o, e in drawn), k)
    pub = _State.pub
    if pub is None or pub.key != key:
        ids, total, tds, mats = {}, [], [], []
        for _obj, e in drawn:
            lut = []
            for mk in e.mat_keys:
                if mk not in ids:
                    ids[mk] = len(total)
                    total.append(0)
                lut.append(ids[mk])
            lut = np.asarray(lut, dtype=np.int64)
            for i, c in zip(lut.tolist(), e.mat_faces.tolist()):
                total[i] += c
            tds.append(_td3d_values(e, k))
            mats.append(lut[e.shell_mat])
        pub = _State.pub = _published(key, 'OBJECT', tuple(sorted(o.name_full for o, _e in drawn)),
                                      np.concatenate(tds), k, np.concatenate(mats), list(ids), total, total)
    _after_publish(scene, st, pub)


def _draw3d_main(context):
    scene = getattr(context, "scene", None)
    st = getattr(scene, "uv_gap_overlay", None) if scene is not None else None
    if st is None or not st.td_show_3d:
        return
    space = context.space_data
    if space is None or space.type != 'VIEW_3D':
        return
    overlay = getattr(space, "overlay", None)
    if overlay is not None and not getattr(overlay, "show_overlays", True):
        return
    objects, edit = _td3d_objects(context, space)
    if not objects:
        _State.td_3d_stats = None
        return
    depsgraph = None if edit else context.evaluated_depsgraph_get()
    drawn, stale, deferred = [], False, False
    now = time.perf_counter()
    budget_end = now + TD3D_BUDGET_MS / 1000.0
    built_any = False
    for obj in objects:
        may_build = not built_any or time.perf_counter() < budget_end  # always make progress
        e, s, built, d = _td3d_entry(obj, depsgraph, edit, may_build)
        built_any |= built
        deferred |= d
        if e is not None:
            e.last_used = now
            drawn.append((obj, e))
            stale |= s
    if deferred:  # the rest next frame, so a big selection doesn't freeze the viewport
        _schedule_redraw(0.0)
    for k in [k for k, e in _State.td3d.items() if now - e.last_used > 5.0]:
        del _State.td3d[k]  # objects no longer shown anywhere
    if not drawn:
        _State.td_3d_stats = None
        return
    W, H, _src = _resolve_resolution_3d(st)
    scale = _unit_scale(scene)
    thr = _td_thresholds(st)
    k = _td_factor(W, H, scale)
    alpha = max(0.0, min(1.0, float(st.td_fill_opacity))) * (0.35 if stale else 1.0)
    items = []
    for obj, e in drawn:
        M = obj.matrix_world
        m3 = M.to_3x3()
        if e.m3 is None or e.m3 != m3:  # cheap check first: most frames nothing moved
            _td3d_update(e, np.array(m3, dtype=np.float64))
            e.m3 = m3
        items.append((M.copy(), e))
    fkey = (tuple((e.gen, e.version) for _o, e in drawn), k, thr)
    if _State.td3d_frame is None or _State.td3d_frame[0] != fkey:
        tds = [_td3d_values(e, k) for _o, e in drawn]
        _State.td3d_frame = (fkey, _td_stats(np.concatenate(tds) if tds else _EMPTY_F, thr))
    _State.td_3d_stats = _State.td3d_frame[1]
    if not edit:
        _publish_3d(scene, st, drawn, k)
    if alpha <= 0.0:
        return
    sh = _td_shader()
    shader_td = sh is not None
    if not shader_td:
        _name, sh = _builtin('fill')
        if sh is None:
            return
    batches = [(M, _td3d_batch(e, sh, shader_td, W, H, scale, thr, alpha)) for M, e in items]
    try:
        prev_test, prev_mask = gpu.state.depth_test_get(), gpu.state.depth_mask_get()
    except Exception:
        prev_test, prev_mask = 'NONE', False
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)
    try:
        P = gpu.matrix.get_projection_matrix()
        biased = P.copy()
        biased[2] = P[2] - TD3D_DEPTH_BIAS * P[3]  # NDC z - bias: nearer by a constant depth step
        with gpu.matrix.push_pop_projection():
            gpu.matrix.load_projection_matrix(biased)
            if shader_td:
                _td_shader_setup(sh, thr, k, alpha)
            else:
                sh.bind()
            for M, batch in batches:
                if batch is None:
                    continue
                with gpu.matrix.push_pop():
                    gpu.matrix.multiply_matrix(M)
                    if shader_td:
                        _set_mvp(sh)
                    batch.draw(sh)
    finally:
        gpu.state.depth_mask_set(prev_mask)
        gpu.state.depth_test_set(prev_test)
        gpu.state.blend_set('NONE')


def _draw3d_callback():
    try:
        _draw3d_main(bpy.context)
    except Exception:
        err = traceback.format_exc()
        if err != _State.last_error:
            _State.last_error = err
            print("[UV Shell Gap Overlay] 3D view draw error:\n" + err)


# ---------------------------------------------------------------------------
# Change tracking
# ---------------------------------------------------------------------------

def _mark_changed(geometry=True):
    """Something changed: a mesh or an object, or (`geometry` False) only what is selected."""
    if geometry:
        _State.geo_dirty = True
    else:
        _State.sel_dirty = True
    _State.data_gen += 1
    _State.last_change = time.perf_counter()


def _note_change(ids, geometry=True):
    """Count a change of these objects / meshes (as _on_depsgraph_update does for Blender's
    update notifications): what was read from them is read again."""
    for idb in ids:
        ptr = idb.as_pointer()
        _State.touch[ptr] = _State.touch.get(ptr, 0) + 1
        if geometry:
            _State.id_gen[ptr] = _State.id_gen.get(ptr, 0) + 1
    _mark_changed(geometry)


@persistent
def _on_depsgraph_update(*args):
    depsgraph = args[1] if len(args) > 1 else None
    if depsgraph is None:
        _State.epoch += 1
        _mark_changed()
        return
    changed = geometry = False
    touch, id_gen = _State.touch, _State.id_gen
    own = _State.own_sel
    for upd in depsgraph.updates:
        idb = upd.id
        if isinstance(idb, bpy.types.Mesh):
            if idb.name == _SCRATCH_MESH:
                continue
        elif not (isinstance(idb, bpy.types.Object) and idb.type == 'MESH'):
            continue
        changed = True
        # change counts per object and mesh: only what changed is read again. A selection
        # change (no geometry update) matters only where the selection is looked at.
        orig = getattr(idb, "original", None) or idb
        ptr = orig.as_pointer()
        touch[ptr] = touch.get(ptr, 0) + 1
        # (a selection written by one of the add-on's own buttons is announced as a geometry
        # update, because the edit mesh was written to: see _announce_selection)
        if getattr(upd, "is_updated_geometry", True) and not (own is not None and ptr in own):
            geometry = True
            id_gen[ptr] = id_gen.get(ptr, 0) + 1
    if changed:
        _mark_changed(geometry)


def _announce_selection(context, ids):
    """Have Blender announce, here and now, the selection one of the add-on's buttons has just
    written to these objects and meshes. Blender words it as a geometry update, since the edit
    mesh was written to; counted as one, it would make the next redraw read all these objects
    again. But the button read them just before (_shown_geometry), and between that reading
    and this call nothing happened but its own selection change: so what Blender reports for
    them now is counted as a selection change, and what was read stays valid. (Updates that
    come later, or for other objects, are counted as always. If Blender announces nothing
    here, the update comes after the button and is counted as a geometry update, as before.)"""
    st = _State
    st.own_sel = {idb.as_pointer() for idb in ids}
    try:
        context.view_layer.update()
    except Exception:
        pass
    finally:
        st.own_sel = None


@persistent
def _on_undo_redo(*_args):
    # undo can replace data-blocks (and their memory addresses): read everything again
    # (objects that come back unchanged keep their shells: they are compared, see _object_parts)
    _State.epoch += 1
    _State.arrays.clear()
    _State.td3d.clear()
    _State.auto_key = None  # undo may have reverted auto-filled Low / High
    _mark_changed()


def _migrate_settings():
    """Files saved before 1.4 have no Auto Low / High setting: if their Low / High were set
    by hand, turn Auto off for them so those values aren't replaced. Also a timer after
    register (data is not accessible while add-ons register at startup)."""
    try:
        for scene in bpy.data.scenes:
            st = getattr(scene, "uv_gap_overlay", None)
            if st is None or st.settings_version >= 140 or st.is_property_set("td_auto_range"):
                continue
            if st.is_property_set("td_low_m") or st.is_property_set("td_high_m"):
                st.td_auto_range = False
                st.settings_version = 140
    except Exception:
        _report("could not update settings from an earlier version", traceback.format_exc())
    return None


@persistent
def _on_load(*_args):
    _State.reset()
    _migrate_settings()


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class UVGAP_OT_refresh(bpy.types.Operator):
    bl_idname = "uv.gap_overlay_refresh"
    bl_label = "Refresh"
    bl_description = "Recompute UV shell gaps now"

    def execute(self, context):
        _State.invalidate(hard=True)
        _redraw_uv_editors()
        return {'FINISHED'}


def _select_uv_faces(bm, uvl, sync, shown, picked, extend):
    """Select the UVs of the `picked` faces (with UV Sync Selection: the mesh faces), and
    deselect the other `shown` faces unless extending. Whole shells only: shells share no
    UV vertex, so setting one shell's corners never touches another's."""
    if sync:
        if extend and _UV_SELECT_ON_LOOP and bm.uv_select_sync_valid:
            # Blender 5.0+: add to the synced UV selection, then carry it over to the mesh
            bm.uv_select_foreach_set_from_mesh(True, faces=picked)
            bm.uv_select_sync_to_mesh()
        else:
            if not extend:
                for f in shown:
                    f.select_set(False)
            for f in picked:
                f.select_set(True)
            bm.select_flush_mode()
            if _UV_SELECT_ON_LOOP:
                bm.uv_select_sync_valid = False  # the UV selection follows the new mesh selection
    elif _UV_SELECT_ON_LOOP:
        # Blender 5.0+: per-face setters also clear/set the face corners (flushed down)
        if not extend:
            for f in shown:
                f.uv_select_set(False)
        for f in picked:
            f.uv_select_set(True)
        bm.uv_select_flush_mode()
    else:
        on = set(picked)
        for f in (picked if extend else shown):
            s = f in on
            for loop in f.loops:
                luv = loop[uvl]
                luv.select = s
                luv.select_edge = s


def _selection_keeper(bm, drop):
    """Hiding or deselecting faces also deselects their vertices and edges, shared ones too:
    remember what else is selected and return a function that selects it again."""
    drop_v = {v for f in drop for v in f.verts}
    drop_e = {e for f in drop for e in f.edges}
    faces = [f for f in bm.faces if f.select and f not in drop]
    verts = [v for v in bm.verts if v.select and v not in drop_v]
    edges = [e for e in bm.edges if e.select and e not in drop_e]

    def restore():
        for elems in (faces, edges, verts):
            for x in elems:
                if not x.hide:
                    x.select_set(True)
    return restore


def _hide_reveal(bm, uvl, sync, faces, reveal):
    """Hide or reveal faces in the UV editor, as Blender's UV Hide / Reveal do: with UV Sync
    Selection they are hidden in the mesh (revealed ones get selected); without it the UV
    editor shows the selected faces, so they are deselected (selected, with their UVs).
    Returns how many faces changed."""
    if reveal:
        if sync:
            todo = [f for f in faces if f.hide]
            for f in todo:
                f.hide_set(False)
            for f in todo:
                f.select_set(True)
            bm.select_flush_mode()
            if _UV_SELECT_ON_LOOP and todo:
                bm.uv_select_sync_valid = False
        else:
            todo = [f for f in faces if not f.hide and not f.select]
            for f in todo:
                f.select_set(True)
            if todo:
                _select_uv_faces(bm, uvl, False, (), todo, True)
        return len(todo)
    drop = [f for f in faces if not f.hide and (sync or f.select)]
    if not drop:
        return 0
    restore = _selection_keeper(bm, set(drop))
    for f in drop:
        if sync:
            f.hide_set(True)
        else:
            f.select_set(False)
    restore()
    if sync:
        bm.select_flush_mode()
        if _UV_SELECT_ON_LOOP:
            bm.uv_select_sync_valid = False
    return len(drop)


def _shown_geometry(context):
    """(objects in Edit Mode, their shells as the UV editor shows them): every object is read
    again, so that an operator acts on the meshes as they are now. What turns out unchanged
    keeps its shells and measurements. The geometry is None when no object is in Edit Mode."""
    scene = context.scene
    objects = _edit_mesh_objects(context)
    geo = _uv_geometry(scene, scene.uv_gap_overlay, objects, fresh=True)[0] if objects else None
    return objects, geo


def _material_shells(geo, keys):
    """All-shell indices of the shown shells whose material - the one covering most of the
    shell, as in the overlay and the list - is in `keys`."""
    ids = [i for i, k in enumerate(geo.mat_keys) if k in keys]
    if not geo.sh_n or not ids:
        return _EMPTY_I
    return np.flatnonzero(np.isin(geo.sh_mat, ids))


def _hidden_material_faces(objects, keys):
    """[(object, mesh face indices)] of the shells with a face not shown in the UV editor whose
    material is in `keys`: whole shells, hidden faces included, decide the material. Uses the
    arrays just read for _shown_geometry."""
    out = []
    for obj in objects:
        P = _State.parts.get(obj.as_pointer())
        if P is None or P.vis.all():
            continue
        fidx, _fsh = _material_shell_faces(obj, P.A, _shells(P.A, np.ones(P.vis.size, dtype=bool)), keys)
        if fidx.size:
            out.append((obj, fidx))
    return out


def _material_shell_faces(obj, A, S, keys):
    """(mesh face indices, shell of each) of the shells of `S` whose material - the one
    covering most of the shell, as in the overlay and the list - is in `keys`."""
    uniq, lut = _material_sets(obj)
    fm = lut[np.clip(A.f_mat[S.fidx], 0, lut.size - 1)]
    dom = _dominant(S.face_shell, fm, np.abs(S.area2) * 0.5, S.nshell)
    use = np.asarray([k in keys for k in uniq], dtype=bool)[dom]
    f = np.flatnonzero(use[S.face_shell])
    return S.fidx[f], S.face_shell[f]


def _target_keys(st, index):
    """Material sets an operator acts on: the given row, else the checked sets, else the active row."""
    items = st.materials
    if 0 <= index < len(items):
        return {items[index].key}
    keys = {it.key for it in items if it.checked}
    if not keys and 0 <= st.material_index < len(items):
        keys = {items[st.material_index].key}
    return keys


def _set_names(keys):
    names = sorted(_material_label(k) for k in keys)
    return names[0] if len(names) == 1 else "%d material sets" % len(names)


class UVGAP_OT_select_flipped(bpy.types.Operator):
    bl_idname = "uv.gap_overlay_select_flipped"
    bl_label = "Select Flipped Shells"
    bl_description = ("Select UV shells whose UVs are mirrored (clockwise winding in UV space). "
                      "With UV Sync Selection the mesh faces are selected, otherwise the UVs")
    bl_options = {'REGISTER', 'UNDO'}

    extend: BoolProperty(
        name="Extend", description="Add to the current selection instead of replacing it", default=False)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def execute(self, context):
        objects, geo = _shown_geometry(context)
        count = 0
        if geo is not None:
            # the shells the overlay counts as flipped (mirrored copies lying in stacks too)
            shells = geo.gap_shell[np.flatnonzero(geo.flipped)] if geo.nshells else _EMPTY_I
            count = int(shells.size)
            _select_shells(context, objects, geo, shells, bool(context.scene.tool_settings.use_uv_select_sync),
                           self.extend)
        _redraw_areas(('IMAGE_EDITOR', 'VIEW_3D'))
        self.report({'INFO'}, "%d flipped shell%s selected" % (count, "" if count == 1 else "s"))
        return {'FINISHED'}


class UVGAP_OT_material_select(bpy.types.Operator):
    bl_idname = "uv.gap_overlay_material_select"
    bl_label = "Select Material Shells"
    bl_description = ("Select the UV shells of a material set (with UV Sync Selection the mesh faces): "
                      "the row's set, or the checked sets (the active row when none is checked). "
                      "Shift: add to the selection")
    bl_options = {'REGISTER', 'UNDO'}

    index: IntProperty(name="Row", default=-1, options={'HIDDEN', 'SKIP_SAVE'})
    extend: BoolProperty(
        name="Extend", description="Add to the current selection instead of replacing it",
        default=False, options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def invoke(self, context, event):
        self.extend = self.extend or event.shift
        return self.execute(context)

    def execute(self, context):
        st = context.scene.uv_gap_overlay
        keys = _target_keys(st, self.index)
        if not keys:
            self.report({'WARNING'}, "No material set to select")
            return {'CANCELLED'}
        objects, geo = _shown_geometry(context)
        count = 0
        if geo is not None:
            shells = _material_shells(geo, keys)  # (of the shells shown)
            count = int(shells.size)
            _select_shells(context, objects, geo, shells, bool(context.scene.tool_settings.use_uv_select_sync),
                           self.extend)
        _redraw_areas(('IMAGE_EDITOR', 'VIEW_3D'))
        self.report({'INFO'}, "%s: %d shell%s selected" % (_set_names(keys), count, "" if count == 1 else "s"))
        return {'FINISHED'}


class UVGAP_OT_material_visibility(bpy.types.Operator):
    bl_idname = "uv.gap_overlay_material_visibility"
    bl_label = "Hide / Reveal Material Shells"
    bl_description = ("Hide or reveal the UV shells of a material set in the UV Editor: the row's set, or "
                      "the checked sets (the active row when none is checked). With UV Sync Selection "
                      "the faces are hidden in the mesh; without it they are deselected, as UV Hide does")
    bl_options = {'REGISTER', 'UNDO'}

    index: IntProperty(name="Row", default=-1, options={'HIDDEN', 'SKIP_SAVE'})
    action: EnumProperty(
        name="Action",
        items=[('HIDE', "Hide", "Hide the shells"),
               ('REVEAL', "Reveal", "Show the shells again"),
               ('TOGGLE', "Toggle", "Hide the shells if any is shown, else reveal them")],
        default='TOGGLE', options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def execute(self, context):
        st = context.scene.uv_gap_overlay
        keys = _target_keys(st, self.index)
        if not keys:
            self.report({'WARNING'}, "No material set to hide or reveal")
            return {'CANCELLED'}
        sync = bool(context.scene.tool_settings.use_uv_select_sync)
        objects, geo = _shown_geometry(context)
        shown = _material_shells(geo, keys) if geo is not None else _EMPTY_I
        action = self.action
        if action == 'TOGGLE':
            action = 'HIDE' if shown.size else 'REVEAL'
        if geo is None:
            todo = []
        elif action == 'REVEAL':
            todo = _hidden_material_faces(objects, keys)
        else:
            todo = [(objects[oi], fidx) for oi, fidx in zip(geo.obj_src, _shell_face_ids(geo, shown))]
        changed = 0
        touched = []
        for obj, fidx in todo:
            if not fidx.size:
                continue
            me = obj.data
            bm = bmesh.from_edit_mesh(me)
            bm.faces.ensure_lookup_table()
            faces = bm.faces
            changed += _hide_reveal(bm, bm.loops.layers.uv.active, sync, [faces[i] for i in fidx.tolist()],
                                    action == 'REVEAL')
            bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
            touched += (obj, me)
        _note_change(touched)
        _redraw_areas(('IMAGE_EDITOR', 'VIEW_3D'))
        self.report({'INFO'}, "%s: %d face%s %s" % (_set_names(keys), changed, "" if changed == 1 else "s",
                                                     "revealed" if action == 'REVEAL' else "hidden"))
        return {'FINISHED'}


class UVGAP_OT_material_check(bpy.types.Operator):
    bl_idname = "uv.gap_overlay_material_check"
    bl_label = "Check Material Sets"
    bl_description = "Check all, none or the opposite material sets"
    bl_options = {'REGISTER', 'UNDO'}

    action: EnumProperty(
        name="Action",
        items=[('ALL', "All", "Check every set"), ('NONE', "None", "Uncheck every set"),
               ('INVERT', "Invert", "Swap checked and unchecked")],
        default='ALL')

    def execute(self, context):
        for it in context.scene.uv_gap_overlay.materials:
            it.checked = True if self.action == 'ALL' else False if self.action == 'NONE' else not it.checked
        _redraw_areas(('IMAGE_EDITOR', 'VIEW_3D'))
        return {'FINISHED'}


class UVGAP_OT_td_fill(bpy.types.Operator):
    bl_idname = "uv.gap_overlay_td_fill"
    bl_label = "Fill Low / High"
    bl_description = ("Set Low and High to the lowest and highest texel density of the checked "
                      "material sets (all shells when none is checked)")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        st = context.scene.uv_gap_overlay
        pub = _panel_pub(context)
        checked = _checked_keys(st)
        rng = None
        if pub is not None:
            fkey, kc = _auto_key(context.scene, st, pub, checked)
            rng = _auto_range(pub, checked, kc)
        if rng is None:
            self.report({'WARNING'}, "No texel density to fill from (enter Edit Mode, or show texel density "
                                     "in the 3D Viewport for selected objects)")
            return {'CANCELLED'}
        st.td_low_m, st.td_high_m = rng
        st.settings_version = max(st.settings_version, 140)
        _State.auto_key = fkey
        _redraw_areas(('IMAGE_EDITOR', 'VIEW_3D'))
        return {'FINISHED'}


class UVGAP_OT_select_td_range(bpy.types.Operator):
    bl_idname = "uv.gap_overlay_select_td_range"
    bl_label = "Select Shells in Range"
    bl_description = ("Select the shown UV shells whose texel density is inside the From / To range "
                      "(with UV Sync Selection the mesh faces). Shift: add to the selection")
    bl_options = {'REGISTER', 'UNDO'}

    extend: BoolProperty(
        name="Extend", description="Add to the current selection instead of replacing it",
        default=False, options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def invoke(self, context, event):
        self.extend = self.extend or event.shift
        return self.execute(context)

    def execute(self, context):
        scene = context.scene
        st = scene.uv_gap_overlay
        sync = bool(scene.tool_settings.use_uv_select_sync)
        space = context.space_data
        if space is not None and space.type == 'IMAGE_EDITOR':
            W, H, _src = _resolve_resolution(st, space)
        else:
            W, H, _src = _resolve_resolution_3d(st)
        scale = _unit_scale(scene)
        objects, geo = _shown_geometry(context)
        bounds = None
        if geo is not None and geo.td_nshells:
            td = _td_values(geo.td_uv_area, geo.td_world_area, W, H, scale)
            bounds = _td_range_bounds(td, st.td_sel_from / 100.0, st.td_sel_to / 100.0)
        if bounds is None:
            self.report({'WARNING'}, "No shells with a texel density")
            return {'CANCELLED'}
        with np.errstate(invalid='ignore'):
            inside = np.flatnonzero((td >= bounds[0]) & (td <= bounds[1]))
        count = int(inside.size)
        _select_shells(context, objects, geo, inside, sync, self.extend)
        _redraw_areas(('IMAGE_EDITOR', 'VIEW_3D'))
        f, u = _TD_FACTOR.get(st.td_unit, 1.0), _TD_LABEL.get(st.td_unit, "px/m")
        self.report({'INFO'}, "%d shell%s from %s to %s selected" % (
            count, "" if count == 1 else "s", _fmt_td(max(bounds[0], 0.0) * f, u), _fmt_td(bounds[1] * f, u)))
        return {'FINISHED'}


def _plural(n, word):
    return "%d %s%s" % (n, word, "" if n == 1 else "s")


def _overlap_data(op, context):
    """(objects, geometry, stacks or None, overlapping pairs) of the shells shown in the UV
    editor as they are now - every object is read again, so the operator acts on the mesh as
    it is -, or None (reported) when no shell is shown. With stacks on, the pairs are the
    overlaps that aren't stacks."""
    st = context.scene.uv_gap_overlay
    objects, geo = _shown_geometry(context)
    if geo is None or not geo.nshells:
        op.report({'WARNING'}, "No UV shells shown")
        return None
    same_mat, smatch = bool(st.same_material), _stack_match(st)
    cache = _measure_cache()
    stacks = _stacks_of(geo, same_mat, smatch, cache) if smatch is not None else None
    return objects, geo, stacks, _overlap_pairs(geo, same_mat, cache, smatch)


def _but_one(geo, pairs, stacks, home_only):
    """All-shell indices of every shell lying on another but one per stack or set of
    overlapping shells: the one that stays isn't flipped (where there is such a shell). With
    `home_only`, only shells of the 0-1 tile: copies moved away earlier are left alone."""
    shells = _overlaps_but_one(geo, pairs, stacks)
    if home_only and shells.size:
        gid = geo.sh_gid[shells]
        tile = geo.tile[np.maximum(gid, 0)]
        shells = shells[(gid >= 0) & (tile[:, 0] == 0) & (tile[:, 1] == 0)]
    return shells


def _shell_face_ids(geo, shells):
    """Per object of the geometry: the mesh face indices of the given shells (all-shell indices)."""
    pick = np.zeros(max(geo.sh_n, 1), dtype=bool)
    pick[np.asarray(shells, dtype=np.int64)] = True
    out = []
    for P, b in zip(geo.obj_parts, geo.sh_base):
        local = pick[b:b + P.sh["nshell"]]
        out.append(P.fidx[local[P.face_shell]] if local.any() else _EMPTY_I32)
    return out


def _select_shells(context, objects, geo, shells, sync, extend):
    """Select the given shells (all-shell indices) of a geometry just read (_shown_geometry):
    their UVs, with UV Sync Selection their mesh faces. Without `extend` the other shown
    shells are deselected."""
    touched = []
    for P, oi, pick in zip(geo.obj_parts, geo.obj_src, _shell_face_ids(geo, shells)):
        if extend and not pick.size:
            continue
        obj = objects[oi]
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()
        faces = bm.faces
        _select_uv_faces(bm, bm.loops.layers.uv.active, sync, [faces[i] for i in P.fidx.tolist()],
                         [faces[i] for i in pick.tolist()], extend)
        bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
        touched += (obj, me)
    if touched:
        _note_change(touched, geometry=False)
        _announce_selection(context, touched)


def _move_shells(objects, geo, shells, du, dv):
    """Move the UVs of the given shells (all-shell indices) by (du, dv)."""
    touched = []
    for oi, pick in zip(geo.obj_src, _shell_face_ids(geo, shells)):
        if not pick.size:
            continue
        obj = objects[oi]
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        uvl = bm.loops.layers.uv.active
        bm.faces.ensure_lookup_table()
        faces = bm.faces
        for i in pick.tolist():
            for loop in faces[i].loops:  # (every corner has its own UV: each is moved once)
                luv = loop[uvl]
                u, v = luv.uv
                luv.uv = (u + du, v + dv)
        bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
        touched += (obj, me)
    _note_change(touched)


_OVERLAP_MODES = [
    ('OVERLAPPING', "Overlapping", "Select the shells that overlap and aren't a stack: copies that don't fit, "
                                   "shells lying partly on others (with all the shells of the stacks involved)"),
    ('STACKED', "Stacked", "Select every shell lying in a stack: shells sharing texture space on purpose"),
    ('BUT_ONE', "All but One", "Select all the shells lying on others but one of every stack and overlap: a "
                               "shell that isn't flipped stays unselected (where there is one)"),
]


class UVGAP_OT_select_overlaps(bpy.types.Operator):
    bl_idname = "uv.gap_overlay_select_overlaps"
    bl_label = "Select Overlapping Shells"
    bl_description = "Select UV shells lying on each other (with UV Sync Selection the mesh faces)"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name="Shells", items=_OVERLAP_MODES, default='OVERLAPPING')
    extend: BoolProperty(
        name="Extend", description="Add to the current selection instead of replacing it",
        default=False, options={'SKIP_SAVE'})
    home_only: BoolProperty(
        name="Only in the 0-1 Tile",
        description="All but One: only shells lying in the 0-1 tile (UDIM 1001)", default=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    @classmethod
    def description(cls, context, properties):
        for key, _name, text in _OVERLAP_MODES:
            if key == properties.mode:
                return text + ". With UV Sync Selection the mesh faces are selected. Shift: add to the selection"
        return cls.bl_description

    def invoke(self, context, event):
        self.extend = self.extend or event.shift
        return self.execute(context)

    def execute(self, context):
        got = _overlap_data(self, context)
        if got is None:
            return {'CANCELLED'}
        objects, geo, stacks, pairs = got
        if self.mode == 'STACKED':
            if stacks is None:
                self.report({'WARNING'}, "Stacks are off: turn on Stacked Shells as One")
                return {'CANCELLED'}
            shells = geo.gap_shell[np.flatnonzero(stacks.size > 1)]
            note = ("%s selected in %s" % (_plural(int(shells.size), "stacked shell"), _plural(stacks.counts()[0], "stack"))
                    if shells.size else "No stacked shells")
        elif self.mode == 'BUT_ONE':
            shells = _but_one(geo, pairs, stacks, self.home_only)
            note = ("%s selected: one shell of every stack and overlap stays" % _plural(int(shells.size), "shell")
                    if shells.size else "No shells lie on each other" + (" in the 0-1 tile" if self.home_only else ""))
        else:
            shells = _overlapping_shells(geo, pairs, stacks)
            note = ("%s selected (%s)" % (_plural(int(shells.size), "overlapping shell"),
                                         _plural(len(pairs), "overlapping pair"))
                    if shells.size else "No overlapping shells" + (" (stacks aside)" if stacks is not None else ""))
        _select_shells(context, objects, geo, shells, bool(context.scene.tool_settings.use_uv_select_sync),
                       self.extend)
        _redraw_areas(('IMAGE_EDITOR', 'VIEW_3D'))
        self.report({'INFO'}, note)
        return {'FINISHED'}


class UVGAP_OT_move_overlaps(bpy.types.Operator):
    bl_idname = "uv.gap_overlay_move_overlaps"
    bl_label = "Move All but One"
    bl_description = ("Move all the shells lying on others to another UDIM tile, leaving one of every stack and "
                      "overlap in place - a shell that isn't flipped (where there is one)")
    bl_options = {'REGISTER', 'UNDO'}

    tiles: IntProperty(
        name="Tiles", description="How many UDIM tiles to move along U (negative: to the left)",
        default=1, min=-99, max=99)
    home_only: BoolProperty(
        name="Only in the 0-1 Tile",
        description="Move only shells lying in the 0-1 tile (UDIM 1001): copies moved to another tile earlier "
                    "stay where they are", default=True)
    select: BoolProperty(
        name="Select Moved Shells", description="Select the shells that were moved", default=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def execute(self, context):
        if not self.tiles:
            self.report({'WARNING'}, "Tiles is 0: nothing to move")
            return {'CANCELLED'}
        got = _overlap_data(self, context)
        if got is None:
            return {'CANCELLED'}
        objects, geo, stacks, pairs = got
        shells = _but_one(geo, pairs, stacks, self.home_only)
        if not shells.size:
            self.report({'INFO'}, "No shells lie on each other%s: nothing to move"
                        % (" in the 0-1 tile" if self.home_only else ""))
            return {'CANCELLED'}
        if self.select:
            _select_shells(context, objects, geo, shells, bool(context.scene.tool_settings.use_uv_select_sync), False)
        _move_shells(objects, geo, shells, float(self.tiles), 0.0)
        _redraw_areas(('IMAGE_EDITOR', 'VIEW_3D'))
        self.report({'INFO'}, "%s moved %d tile%s along U; one shell of every stack and overlap stays" % (
            _plural(int(shells.size), "shell"), self.tiles, "" if abs(self.tiles) == 1 else "s"))
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _draw_resolution(col, st, space, in_3d=False):
    col.prop(st, "resolution")
    W, H, src = _resolve_resolution_3d(st) if in_3d else _resolve_resolution(st, space)
    if st.resolution == 'CUSTOM' or (st.resolution == 'IMAGE' and src is None):
        sub = col.column(align=True)
        sub.prop(st, "res_x")
        sub.prop(st, "res_y")
    if st.resolution == 'IMAGE':
        if src:
            col.label(text="%s  (%d x %d)" % (src, W, H), icon='IMAGE_DATA')
        else:
            col.label(text=("No image in a UV editor" if in_3d else "No image here") + ": using Width / Height",
                      icon='INFO')


def _draw_td_settings(col, st):
    col.prop(st, "td_unit")
    sub = col.column(align=True)
    sub.prop(st, "td_low", text="Low (Red)")
    sub.prop(st, "td_needed", text="Needed (Green)")
    sub.prop(st, "td_high", text="High (Blue)")
    n_checked = sum(1 for it in st.materials if it.checked)
    row = col.row(align=True, heading="Auto Low / High")
    row.prop(st, "td_auto_range", text=("%d Checked Material%s" % (n_checked, "" if n_checked == 1 else "s"))
             if n_checked else "All Shells")
    row.operator(UVGAP_OT_td_fill.bl_idname, text="", icon='FILE_REFRESH')
    col.prop(st, "td_fill_opacity", slider=True)


def _draw_td_summary(layout, st, stats, empty_text):
    col = layout.box().column(align=True)
    if stats is None:
        col.label(text=empty_text, icon='INFO')
        return
    n, lo, med, hi, n_low, n_high = stats
    f, u = _TD_FACTOR.get(st.td_unit, 1.0), _TD_LABEL.get(st.td_unit, "px/m")
    if lo is None:
        col.label(text="%d shells, none with a 3D area" % n, icon='ERROR')
        return
    col.label(text="%d shells, median %s" % (n, _fmt_td(med * f, u)))
    col.label(text="Lowest %s, highest %s" % (_fmt_td(lo * f, u), _fmt_td(hi * f, u)))
    col.label(text="At or below Low: %d, at or above High: %d" % (n_low, n_high),
              icon='ERROR' if (n_low or n_high) else 'CHECKMARK')


def _fmt_short(v):
    return ("%.0f" % v) if v >= 99.95 else ("%.1f" % v) if v >= 9.995 else ("%.2f" % v)


class UVGAP_UL_materials(bpy.types.UIList):
    """Material sets: checkbox, name, shells shown and their texel density range, select, hide."""
    bl_idname = "UVGAP_UL_materials"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index=0, flt_flag=0):
        pub = _State.pub
        if pub is not None and pub.mode != ('EDIT' if context.mode == 'EDIT_MESH' else 'OBJECT'):
            pub = None
        stats = pub.stats.get(item.key) if pub is not None else None
        row = layout.row(align=True)
        row.prop(item, "checked", text="")
        mat = bpy.data.materials.get(item.key) if item.key else None
        icon_value = 0
        if mat is not None:
            try:
                icon_value = layout.icon(mat)
            except Exception:
                icon_value = 0
        name = row.row()
        if icon_value:
            name.label(text=item.name, icon_value=icon_value)
        else:
            name.label(text=item.name, icon='MATERIAL' if item.key else 'CANCEL')
        if stats is not None:
            n, lo, hi, _faces, _faces_shown = stats
            shown = n  # shells of this material shown: what the eye toggle hides
            f = _TD_FACTOR.get(data.td_unit, 1.0)
            txt = "%d" % n
            if lo is not None:
                txt += "  " + (_fmt_short(lo * f) if abs(hi - lo) <= 1e-9 * max(hi, 1.0)
                               else "%s-%s" % (_fmt_short(lo * f), _fmt_short(hi * f)))
            info = row.row()
            info.alignment = 'RIGHT'
            info.label(text=txt)
        else:
            shown = 1
        op = row.operator(UVGAP_OT_material_select.bl_idname, text="", icon='RESTRICT_SELECT_OFF', emboss=False)
        op.index = index
        op = row.operator(UVGAP_OT_material_visibility.bl_idname, text="",
                          icon='HIDE_OFF' if shown else 'HIDE_ON', emboss=False)
        op.index = index
        op.action = 'TOGGLE'


def _draw_materials(layout, context):
    st = context.scene.uv_gap_overlay
    _panel_pub(context)
    if not st.materials:
        layout.label(text="Enter Edit Mode on a mesh" if context.mode != 'EDIT_MESH' else "No faces",
                     icon='INFO')
        return
    layout.template_list("UVGAP_UL_materials", "", st, "materials", st, "material_index",
                         rows=3 if len(st.materials) < 4 else 5)
    n_checked = sum(1 for it in st.materials if it.checked)
    col = layout.column(align=True)
    row = col.row(align=True)
    row.operator(UVGAP_OT_material_select.bl_idname, text="Select", icon='RESTRICT_SELECT_OFF')
    op = row.operator(UVGAP_OT_material_visibility.bl_idname, text="Hide", icon='HIDE_ON')
    op.action = 'HIDE'
    op = row.operator(UVGAP_OT_material_visibility.bl_idname, text="Reveal", icon='HIDE_OFF')
    op.action = 'REVEAL'
    row = col.row(align=True)
    row.operator(UVGAP_OT_material_check.bl_idname, text="All", icon='CHECKBOX_HLT').action = 'ALL'
    row.operator(UVGAP_OT_material_check.bl_idname, text="None", icon='CHECKBOX_DEHLT').action = 'NONE'
    row.operator(UVGAP_OT_material_check.bl_idname, text="Invert", icon='ARROW_LEFTRIGHT').action = 'INVERT'
    layout.label(text=("Buttons act on the %d checked set%s" % (n_checked, "" if n_checked == 1 else "s"))
                 if n_checked else "Buttons act on the active row", icon='INFO')
    layout.prop(st, "same_material")


class _UVGapPanel:
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "UV Gaps"


class UVGAP_PT_main(_UVGapPanel, bpy.types.Panel):
    bl_idname = "UVGAP_PT_main"
    bl_label = "UV Shell Gaps"

    @classmethod
    def poll(cls, context):
        sp = context.space_data
        return sp is not None and sp.type == 'IMAGE_EDITOR' and getattr(sp, "mode", None) == 'UV'

    def draw_header(self, context):
        self.layout.prop(context.scene.uv_gap_overlay, "show", text="")

    def draw(self, context):
        st = context.scene.uv_gap_overlay
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.active = st.show

        _draw_resolution(layout.column(), st, context.space_data)

        layout.separator()
        col = layout.column(align=True)
        col.prop(st, "points", slider=True)
        col.prop(st, "shift", slider=True)

        layout.separator()
        col = layout.column(align=True)
        col.prop(st, "min_px")
        col.prop(st, "needed_px")
        col.prop(st, "search_px")
        col = layout.column()
        col.prop(st, "selected_only")
        col.prop(st, "same_material")

        layout.separator()
        row = layout.row(align=True, heading="Flipped Shells")
        row.prop(st, "show_flipped", text="Show")
        row.operator(UVGAP_OT_select_flipped.bl_idname, text="Select", icon='RESTRICT_SELECT_OFF')

        box = layout.box()
        col = box.column(align=True)
        meas, geo = _State.meas, _State.geo
        if context.mode != 'EDIT_MESH':
            col.label(text="Enter Edit Mode on a mesh", icon='INFO')
        elif meas is None or geo is None:
            col.label(text="Measuring on next UV editor redraw", icon='TIME')
        else:
            d = meas.dist
            crit = int(np.count_nonzero(d < st.min_px))
            warn = int(np.count_nonzero((d >= st.min_px) & (d < st.needed_px)))
            col.label(text="%d shells, %d gaps measured" % (meas.shells_total, d.size))
            stacks = meas.stacks if meas.stacks is not None and meas.stacks.rep.size == geo.nshells else None
            if stacks is not None:
                n_stacks, n_stacked = stacks.counts()
                if n_stacks:
                    col.label(text="%d shells in %d stack%s" % (n_stacked, n_stacks, "" if n_stacks == 1 else "s"))
            # (with shells lying on each other in other tiles too - copies moved there -, the
            # count for the 0-1 tile tells whether that one is clean)
            pairs_home = _home_counts(geo, meas, stacks)[0]
            col.label(text="Overlapping pairs: %d" % meas.overlap_pairs,
                      icon='ERROR' if meas.overlap_pairs else 'CHECKMARK')
            if pairs_home is not None and pairs_home != meas.overlap_pairs:
                col.label(text="%d of them in the 0-1 tile" % pairs_home, icon='ERROR' if pairs_home else 'CHECKMARK')
            col.label(text="Gaps below minimal: %d, below needed: %d" % (crit, warn))
            if st.tile_border:
                bd = meas.bdist
                bcrit = int(np.count_nonzero(bd < st.border_min_px))
                bwarn = int(np.count_nonzero((bd >= st.border_min_px) & (bd < st.border_needed_px)))
                col.label(text="Tile border below minimal: %d, below needed: %d" % (bcrit, bwarn))
                if meas.tile_crossing:
                    col.label(text="Shells crossing a tile border: %d" % meas.tile_crossing, icon='ERROR')
            nflip = int(np.count_nonzero(geo.flipped))
            # mirrored copies lying in a stack with an unflipped shell are there on purpose
            tucked = int(np.count_nonzero(geo.flipped & ~geo.flipped[stacks.rep])) if stacks is not None else 0
            col.label(text="Flipped shells: %d" % nflip + (" (%d in stacks)" % tucked if tucked else ""),
                      icon='ERROR' if nflip > tucked else 'CHECKMARK')
        layout.operator(UVGAP_OT_refresh.bl_idname, icon='FILE_REFRESH')


class UVGAP_PT_overlaps(_UVGapPanel, bpy.types.Panel):
    bl_idname = "UVGAP_PT_overlaps"
    bl_label = "Overlaps and Stacks"
    bl_parent_id = "UVGAP_PT_main"

    def draw(self, context):
        st = context.scene.uv_gap_overlay
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column()
        col.prop(st, "stacks")
        sub = col.column()
        sub.active = st.stacks
        sub.prop(st, "stack_match")

        col = layout.column(align=True)
        col.use_property_split = False
        col.label(text="Select (Shift: Add)")
        row = col.row(align=True)
        for mode, text in (('OVERLAPPING', "Overlapping"), ('STACKED', "Stacked"), ('BUT_ONE', "All but One")):
            op = row.operator(UVGAP_OT_select_overlaps.bl_idname, text=text)
            op.mode = mode
            op.home_only = st.stack_home_only

        col = layout.column(align=True)
        col.use_property_split = False
        row = col.row(align=True)
        row.prop(st, "stack_tiles")
        op = row.operator(UVGAP_OT_move_overlaps.bl_idname)
        op.tiles = st.stack_tiles
        op.home_only = st.stack_home_only
        col.prop(st, "stack_home_only")

        geo, meas = _State.geo, _State.meas
        stacks = None
        if context.mode == 'EDIT_MESH' and geo is not None and st.stacks:
            for found in (meas.stacks if meas is not None else None, _State.stacks):
                if found is not None and found.rep.size == geo.nshells:
                    stacks = found
                    break
        if stacks is not None:
            box = layout.box().column(align=True)
            n_stacks, n_stacked = stacks.counts()
            if n_stacks:
                copies = n_stacked - n_stacks
                copies_home = _home_counts(geo, meas if meas is not None and meas.stacks is stacks else None, stacks)[1]
                box.label(text="%d stack%s of %d shells" % (n_stacks, "" if n_stacks == 1 else "s", n_stacked))
                box.label(text="%d shell%s on another" % (copies, " lies" if copies == 1 else "s lie"))
                if copies_home is not None and copies_home != copies:
                    box.label(text="%d of them in the 0-1 tile" % copies_home)
            else:
                box.label(text="No stacked shells")
            if meas is not None and meas.overlap_pairs >= 100:
                box.label(text="Many overlaps: if copies are placed", icon='INFO')
                box.label(text="less exactly, lower Stack Match")


class UVGAP_PT_tiles(_UVGapPanel, bpy.types.Panel):
    bl_idname = "UVGAP_PT_tiles"
    bl_label = "UDIM Tile Borders"
    bl_parent_id = "UVGAP_PT_main"

    def draw_header(self, context):
        self.layout.prop(context.scene.uv_gap_overlay, "tile_border", text="")

    def draw(self, context):
        st = context.scene.uv_gap_overlay
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.active = st.show and st.tile_border
        col = layout.column(align=True)
        col.prop(st, "border_min_px")
        col.prop(st, "border_needed_px")


class UVGAP_PT_display(_UVGapPanel, bpy.types.Panel):
    bl_idname = "UVGAP_PT_display"
    bl_label = "Display"
    bl_parent_id = "UVGAP_PT_main"

    def draw(self, context):
        st = context.scene.uv_gap_overlay
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column(align=True)
        col.prop(st, "font_size")
        col.prop(st, "opacity", slider=True)
        col.prop(st, "line_width")
        col = layout.column(heading="Labels")
        col.prop(st, "label_background", text="Background")
        col.prop(st, "declutter", text="Hide Overlapping")


class UVGAP_PT_colors(_UVGapPanel, bpy.types.Panel):
    bl_idname = "UVGAP_PT_colors"
    bl_label = "Colors"
    bl_parent_id = "UVGAP_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        st = context.scene.uv_gap_overlay
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.active = st.show
        col = layout.column(align=True)
        col.prop(st, "color_zero")
        col.prop(st, "color_min")
        col.prop(st, "color_needed")
        col = layout.column(align=True)
        col.prop(st, "color_overlap")
        col.prop(st, "color_flipped")


class UVGAP_PT_info(_UVGapPanel, bpy.types.Panel):
    bl_idname = "UVGAP_PT_info"
    bl_label = "Shell Info"

    @classmethod
    def poll(cls, context):
        return UVGAP_PT_main.poll(context)

    def draw_header(self, context):
        self.layout.prop(context.scene.uv_gap_overlay, "info_show", text="")

    def draw(self, context):
        st = context.scene.uv_gap_overlay
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.active = st.info_show
        col = layout.column(heading="Show")
        col.prop(st, "show_orientation", text="Orientation Arrows")
        col.prop(st, "show_scale", text="Object Scale Not 1")
        box = layout.box().column(align=True)
        box.label(text="One block per shell:", icon='INFO')
        box.label(text="texel density (Texel Density panel),")
        box.label(text="object scale, Flipped (with UV Shell Gaps)")
        box.label(text="A stack has one block: × 4 = four shells")
        box.label(text="Blue arrow: the scene's up (+Z) on the shell")
        box.label(text="Green arrow: +Y, on shells lying flat")


class UVGAP_PT_texel(_UVGapPanel, bpy.types.Panel):
    bl_idname = "UVGAP_PT_texel"
    bl_label = "Texel Density"

    @classmethod
    def poll(cls, context):
        return UVGAP_PT_main.poll(context)

    def draw_header(self, context):
        self.layout.prop(context.scene.uv_gap_overlay, "td_show", text="")

    def draw(self, context):
        st = context.scene.uv_gap_overlay
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        _panel_pub(context)
        col = layout.column()
        col.active = st.td_show or st.td_show_3d
        _draw_resolution(col, st, context.space_data)
        col.separator()
        _draw_td_settings(col, st)
        layout.prop(st, "td_show_3d")
        if st.td_show:
            _draw_td_summary(layout, st, _State.td_uv_stats if context.mode == 'EDIT_MESH' else None,
                             "Enter Edit Mode on a mesh")


def _draw_td_range(layout, context):
    st = context.scene.uv_gap_overlay
    pub = _panel_pub(context)
    f, u = _TD_FACTOR.get(st.td_unit, 1.0), _TD_LABEL.get(st.td_unit, "px/m")
    col = layout.column(align=True)
    row = col.row(align=True)
    row.prop(st, "td_sel_from", text="", slider=True)
    row.prop(st, "td_sel_to", text="", slider=True)
    if pub is None or pub.tmin is None:
        layout.label(text="Enter Edit Mode on a mesh" if context.mode != 'EDIT_MESH' else "No shells with a 3D area",
                     icon='INFO')
    else:
        lo, hi = _td_range_bounds(pub.td, st.td_sel_from / 100.0, st.td_sel_to / 100.0)
        row = col.row(align=True)
        row.alignment = 'EXPAND'
        row.label(text=_fmt_td(max(lo, 0.0) * f, u))
        row.label(text=_fmt_td(hi * f, u))
        with np.errstate(invalid='ignore'):
            n_in = int(np.count_nonzero((pub.td >= lo) & (pub.td <= hi)))
        box = layout.box().column(align=True)
        box.label(text="Shells: %s to %s" % (_fmt_td(pub.tmin * f, u), _fmt_td(pub.tmax * f, u)))
        box.label(text="In range: %d of %d" % (n_in, pub.td.size))
    layout.prop(st, "td_range_highlight")
    layout.operator(UVGAP_OT_select_td_range.bl_idname, icon='RESTRICT_SELECT_OFF')


class UVGAP_PT_texel_range(_UVGapPanel, bpy.types.Panel):
    bl_idname = "UVGAP_PT_texel_range"
    bl_label = "Select by Range"
    bl_parent_id = "UVGAP_PT_texel"

    def draw(self, context):
        _draw_td_range(self.layout, context)


class UVGAP_PT_materials(_UVGapPanel, bpy.types.Panel):
    bl_idname = "UVGAP_PT_materials"
    bl_label = "Materials"

    @classmethod
    def poll(cls, context):
        return UVGAP_PT_main.poll(context)

    def draw(self, context):
        _draw_materials(self.layout, context)


class _UVGap3DPanel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "UV Gaps"


class UVGAP_PT_texel_3d(_UVGap3DPanel, bpy.types.Panel):
    bl_idname = "UVGAP_PT_texel_3d"
    bl_label = "Texel Density"

    def draw_header(self, context):
        self.layout.prop(context.scene.uv_gap_overlay, "td_show_3d", text="")

    def draw(self, context):
        st = context.scene.uv_gap_overlay
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        _panel_pub(context)
        col = layout.column()
        col.active = st.td_show_3d or st.td_show
        _draw_resolution(col, st, None, in_3d=True)
        col.separator()
        _draw_td_settings(col, st)
        layout.prop(st, "td_show", text="Show in UV Editor")
        if st.td_show_3d:
            edit = context.mode == 'EDIT_MESH'
            _draw_td_summary(layout, st, _State.td_3d_stats,
                             "No object in Edit Mode" if edit else "Select mesh objects with UVs")
            layout.label(text="Objects in Edit Mode" if edit else "Selected objects, modifiers included",
                         icon='EDITMODE_HLT' if edit else 'OBJECT_DATAMODE')


class UVGAP_PT_materials_3d(_UVGap3DPanel, bpy.types.Panel):
    bl_idname = "UVGAP_PT_materials_3d"
    bl_label = "Materials"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        _draw_materials(self.layout, context)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = (
    UVGAP_MaterialItem,
    UVGAP_Settings,
    UVGAP_OT_refresh,
    UVGAP_OT_select_flipped,
    UVGAP_OT_select_overlaps,
    UVGAP_OT_move_overlaps,
    UVGAP_OT_material_select,
    UVGAP_OT_material_visibility,
    UVGAP_OT_material_check,
    UVGAP_OT_td_fill,
    UVGAP_OT_select_td_range,
    UVGAP_UL_materials,
    UVGAP_PT_main,
    UVGAP_PT_overlaps,
    UVGAP_PT_tiles,
    UVGAP_PT_display,
    UVGAP_PT_colors,
    UVGAP_PT_materials,
    UVGAP_PT_texel,
    UVGAP_PT_texel_range,
    UVGAP_PT_info,
    UVGAP_PT_texel_3d,
    UVGAP_PT_materials_3d,
)

_handler_lists = (
    ("depsgraph_update_post", _on_depsgraph_update),
    ("undo_post", _on_undo_redo),
    ("redo_post", _on_undo_redo),
    ("load_post", _on_load),
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.uv_gap_overlay = PointerProperty(type=UVGAP_Settings)
    _State.reset()
    _State.shader_names = {}
    _State.td_shader = None
    _State.numpy_buffers = True
    _State.draw_handle = bpy.types.SpaceImageEditor.draw_handler_add(
        _draw_callback, (), 'WINDOW', 'POST_PIXEL')
    _State.draw3d_handle = bpy.types.SpaceView3D.draw_handler_add(
        _draw3d_callback, (), 'WINDOW', 'POST_VIEW')
    for name, fn in _handler_lists:
        handlers = getattr(bpy.app.handlers, name)
        if fn not in handlers:
            handlers.append(fn)
    if not bpy.app.timers.is_registered(_create_scratch_mesh):
        bpy.app.timers.register(_create_scratch_mesh, first_interval=0.0)
    if not bpy.app.timers.is_registered(_migrate_settings):
        bpy.app.timers.register(_migrate_settings, first_interval=0.0)


def unregister():
    if _State.draw_handle is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_State.draw_handle, 'WINDOW')
        _State.draw_handle = None
    if _State.draw3d_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_State.draw3d_handle, 'WINDOW')
        _State.draw3d_handle = None
    for name, fn in _handler_lists:
        handlers = getattr(bpy.app.handlers, name)
        if fn in handlers:
            handlers.remove(fn)
    for timer in (_redraw_timer, _create_scratch_mesh, _apply_pending, _migrate_settings):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
    del bpy.types.Scene.uv_gap_overlay
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
    _State.reset()
    _State.td_shader = None
    try:  # the scratch mesh has no users and is never saved; remove it right away anyway
        me = bpy.data.meshes.get(_SCRATCH_MESH)
        if me is not None and not me.users:
            bpy.data.meshes.remove(me)
    except Exception:
        pass
    _redraw_areas(('IMAGE_EDITOR', 'VIEW_3D'))


if __name__ == "__main__":
    register()
