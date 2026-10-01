# SPDX-License-Identifier: GPL-3.0-or-later
#
# UV Shell Gap Overlay  -  Blender add-on for 3.6 LTS to 5.2 LTS
#
# Draws, along the shell borders in the UV Editor, the pixel distance between
# neighboring UV shells and from shells to their UDIM tile border, graded
# against minimal / needed padding; marks overlapping, tile-crossing and
# flipped (mirrored) shells. Colors and labels every shell by its texel
# density, in the UV Editor and on the mesh in the 3D Viewport. One info
# block per shell collects its texel density, object scale, flip state and
# an arrow for the scene's up direction. Material sets: select, hide and
# reveal the shells of chosen materials, measure gaps only within a material.
#
# Authors: Iurii Kruglov & Claude (Anthropic)

bl_info = {
    "name": "UV Shell Gap Overlay",
    "author": "Iurii Kruglov, Claude (Anthropic)",
    "version": (1, 4, 0),
    "blender": (3, 6, 0),
    "location": "UV Editor / 3D Viewport > Sidebar (N) > UV Gaps",
    "description": "Pixel gaps between UV shells and to UDIM tile borders, texel density per shell "
                   "(UV Editor and 3D Viewport), per-shell info blocks, material sets; overlap, "
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

# Blender 5.0 moved UV selection from BMLoopUV.select / select_edge to
# BMLoop.uv_select_vert / uv_select_edge and BMFace.uv_select (shared by all UV maps).
_UV_SELECT_ON_LOOP = hasattr(bmesh.types.BMLoop, "uv_select_vert")

_EMPTY2 = np.empty((0, 2))
_EMPTY_F = np.empty(0)
_EMPTY_I = np.empty(0, dtype=np.int64)
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
    geo_dirty = True
    geo_version = 0
    data_gen = 0         # bumped on every mesh / object change (edit-mode array cache)
    id_gen = {}          # original ID pointer -> geometry change count (3D view cache)
    last_change = 0.0
    last_geo_ms = 0.0
    meas = None
    meas_key = None
    meas_version = 0
    meas_geo_version = -1
    last_meas_ms = 0.0
    pending_key = None
    pending_since = 0.0
    meas_cache = {}      # small per-key caches, so several UV editors don't evict each other
    labels_cache = {}
    arrays = {}          # edit-mode object -> (data_gen, _Arrays)
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
    def invalidate(cls):
        cls.geo_dirty = True
        cls.geo_sig = None
        cls.meas_key = None
        cls.data_gen += 1
        cls.meas_cache.clear()
        cls.labels_cache.clear()
        cls.arrays.clear()
        cls.td3d.clear()
        cls.td_values.clear()
        cls.td_fill.clear()
        cls.frame.clear()
        cls.td3d_frame = None

    @classmethod
    def reset(cls):
        cls.invalidate()
        cls.geo = None
        cls.meas = None
        cls.last_geo_ms = 0.0
        cls.last_meas_ms = 0.0
        cls.pending_key = None
        cls.id_gen.clear()
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


class _Islands:
    """Faces shown in the UV editor, grouped into UV shells."""
    __slots__ = ("vx", "vy", "face_ids", "face_area2", "face_sel", "face_shell", "nshell",
                 "edge_owner", "faces", "sel_readable")


def _uv_islands(bm, uvl, sync, want_sel=False, keep_faces=False, sync_valid=None):
    """UV vertices (a mesh vertex + its UV), per-face UV vertex ids, twice the signed UV
    area of each face (negative = clockwise = mirrored), and shells: faces sharing a UV
    vertex belong to one shell. Border edges are the UV edges used by a single face.

    `sync_valid`: BMesh.uv_select_sync_valid of the edit mesh when `bm` is a copy of it."""
    if sync_valid is None:
        sync_valid = bool(getattr(bm, "uv_select_sync_valid", False))
    q = UV_QUANT
    vid_of = {}
    vid_get = vid_of.get
    vx, vy, vid_face = [], [], []
    parent, face_ids, face_area2, face_sel = [], [], [], []
    faces = [] if keep_faces else None
    edge_owner = {}
    sel_readable = True
    # where "selected" comes from (Selected Shells Only)
    if not want_sel:
        sel_kind = 0
    elif _UV_SELECT_ON_LOOP and (not sync or sync_valid):
        sel_kind = 1  # UV selection stored on the face corner (Blender 5.0+)
    elif sync:
        sel_kind = 2  # mesh selection
    else:
        sel_kind = 3  # UV selection stored on the UV layer (Blender 3.x / 4.x)

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    fi = 0
    for f in bm.faces:
        if f.hide or not (sync or f.select):
            continue  # not displayed in the UV editor
        ids = []
        fsel = False
        for loop in f.loops:
            luv = loop[uvl]
            u, v = luv.uv
            key = (loop.vert, round(u * q), round(v * q))
            vid = vid_get(key)
            if vid is None:
                vid = len(vx)
                vid_of[key] = vid
                vx.append(u)
                vy.append(v)
                vid_face.append(fi)
            ids.append(vid)
            if sel_kind and not fsel and sel_readable:
                try:
                    if sel_kind == 1:
                        fsel = loop.uv_select_vert
                    elif sel_kind == 2:
                        fsel = loop.vert.select
                    else:
                        fsel = luv.select
                except AttributeError:
                    sel_readable = False
        parent.append(fi)
        face_sel.append(fsel)
        face_ids.append(ids)
        if keep_faces:
            faces.append(f)

        n = len(ids)
        area2 = 0.0
        for k in range(n):
            a = ids[k]
            b = ids[k + 1] if k + 1 < n else ids[0]
            area2 += vx[a] * vy[b] - vx[b] * vy[a]
            ekey = (a, b) if a < b else (b, a)
            if ekey in edge_owner:
                edge_owner[ekey] = None  # shared by two faces: interior UV edge
            else:
                edge_owner[ekey] = (fi, a, b)
        face_area2.append(area2)

        for vid in ids:
            other = vid_face[vid]
            if other != fi:
                ra, rb = find(fi), find(other)
                if ra != rb:
                    parent[ra] = rb
        fi += 1

    shell_of_root = {}
    face_shell = [0] * fi
    for i in range(fi):
        r = find(i)
        s = shell_of_root.get(r)
        if s is None:
            s = len(shell_of_root)
            shell_of_root[r] = s
        face_shell[i] = s

    isl = _Islands()
    isl.vx, isl.vy = vx, vy
    isl.face_ids, isl.face_area2, isl.face_sel = face_ids, face_area2, face_sel
    isl.face_shell, isl.nshell = face_shell, len(shell_of_root)
    isl.edge_owner, isl.faces, isl.sel_readable = edge_owner, faces, sel_readable
    return isl


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


def _object_arrays(obj, want_3d, reuse=True):
    """Arrays of an edit-mode object, shared by the UV editor and 3D views until the next
    change (changes arrive as depsgraph updates; `reuse=False` always reads again)."""
    key = obj.as_pointer()
    hit = _State.arrays.get(key)
    if reuse and hit is not None and hit[0] == _State.data_gen and (hit[1].co is not None or not want_3d):
        return hit[1]
    A = _read_edit_arrays(obj, want_3d)
    _State.arrays.pop(key, None)
    if A is not None:
        _cache_put(_State.arrays, key, (_State.data_gen, A), size=8)
    return A


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


def _gap_pieces(S, A, sel_kind):
    """Ordered shell borders, per-shell selection and area, fan triangles of mirrored
    shells: the input of the gap measurement."""
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
    if sel_kind == 0 or not A.sel_readable:
        shell_sel = np.ones(S.nshell, dtype=bool)
    else:
        ls = A.v_sel[A.loop_vert[S.lid]] if sel_kind == 2 else A.uv_sel[S.lid]
        face_sel = np.logical_or.reduceat(ls, S.off) if S.off.size else np.zeros(0, dtype=bool)
        shell_sel = np.bincount(S.face_shell, weights=face_sel, minlength=S.nshell) > 0

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


def _face_area_vectors(S, A):
    """Newell area vector of each face, in object space."""
    P = A.co[A.loop_vert[S.lid]].astype(np.float64)
    if not S.off.size:
        return np.empty((0, 3))
    return np.add.reduceat(np.cross(P, P[S.nxt]), S.off, axis=0) * 0.5


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


def _shell_orientation(S, A, M3, cx, cy, face_area):
    """Where the scene's up axis (+Z) runs across each shell, as a unit direction in UV space.

    Each face's UV -> 3D mapping is fitted to its corners (least squares); the axis, projected
    onto the face, is taken back to UV space and weighted by how much of it lies along the face
    and by the face's area. Shells lying flat (+Z mostly along their normal) use +Y instead.
    Returns directions (NaN: no arrow) and the axis (0 none, 1 +Z, 2 +Y) per shell."""
    ns, fsh, lf = S.nshell, S.face_shell, S.lface
    du = S.vx[S.vid] - cx[lf]
    dv = S.vy[S.vid] - cy[lf]
    P = A.co[A.loop_vert[S.lid]].astype(np.float64) @ M3.T  # world orientation and scale
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


def _shell_pieces(S, A, matrix, fmat, want_3d, want_orient=False):
    """Per shell: label point, UV area and material; with 3D data also the world area (texel
    density), the triangles for the colored fill and, if wanted, the up direction in UV space."""
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
        M3 = np.array(matrix.to_3x3(), dtype=np.float64)
        world = np.linalg.norm(_face_area_vectors(S, A) @ _cofactor(M3).T, axis=1)
        out["world_area"] = np.bincount(fsh, weights=world, minlength=ns)
        if want_orient:
            out["orient"], out["axis"] = _shell_orientation(S, A, M3, cx, cy, world)
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
                 "gap_shell", "shell_mat", "mat_keys", "mat_total", "mat_vis", "obj_names", "obj_scale")

    def __init__(self):
        self.seg_a = _EMPTY2
        self.seg_b = _EMPTY2
        self.seg_sign = _EMPTY_F
        self.seg_shell = _EMPTY_I
        self.shell_off = np.zeros(1, dtype=np.int64)
        self.shell_sel = np.empty(0, dtype=bool)
        self.bmin = _EMPTY2
        self.bmax = _EMPTY2
        self.pairs = {}
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


def _extract(objects, sync, want_sel, want_td=False, reuse=False, want_orient=False):
    geo = _Geometry()
    gaps, shs = [], []
    mat_id, mat_total, mat_vis = {}, [], []

    def material_ids(keys):
        out = []
        for key in keys:
            i = mat_id.get(key)
            if i is None:
                i = mat_id[key] = len(mat_total)
                mat_total.append(0)
                mat_vis.append(0)
            out.append(i)
        return np.asarray(out, dtype=np.int64)

    for obj in objects:
        me = obj.data
        if not me.is_editmode:
            continue
        A = _object_arrays(obj, want_td, reuse)
        if A is None:
            continue
        fm = _face_materials(A, material_ids(_material_keys(obj)))
        vis = ~A.f_hide if sync else (~A.f_hide & A.f_sel)  # faces shown in the UV editor
        for counts, faces in ((mat_total, fm), (mat_vis, fm[vis])):
            for i, c in enumerate(np.bincount(faces, minlength=len(counts)).tolist()):
                counts[i] += c
        S = _shells(A, vis)
        if not S.fidx.size:
            continue
        geo.nfaces += int(S.fidx.size)
        gaps.append(_gap_pieces(S, A, _sel_kind(want_sel, sync, A.sync_valid)))
        p = _shell_pieces(S, A, obj.matrix_world, fm[S.fidx], want_td, want_td and want_orient)
        p["obj"] = len(geo.obj_names)
        shs.append(p)
        geo.obj_names.append(obj.name)
        geo.obj_scale.append(tuple(obj.scale))

    geo.mat_keys = list(mat_id)
    geo.mat_total = np.asarray(mat_total, dtype=np.int64)
    geo.mat_vis = np.asarray(mat_vis, dtype=np.int64)
    if not shs:
        return geo
    base = np.cumsum([0] + [p["nshell"] for p in shs])
    n = geo.sh_n = int(base[-1])
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
    geo.pairs = _find_overlapping_pairs(geo)
    return geo


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


def _probe_hits(pts, own, sgrid, bmin, bmax, A, B, slope, off):
    """Every (point index, shell) where the point lies inside a shell other than own[point].

    Candidates come from shell bounding boxes; the even-odd rule over the
    candidate shell's whole border decides (holes count as outside).
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
    nseg = off[c + 1] - off[c]
    csum = np.cumsum(nseg)
    hp, hs = [], []
    i0 = 0
    while i0 < q.size:
        base = csum[i0 - 1] if i0 else 0
        i1 = min(q.size, max(i0 + 1, int(np.searchsorted(csum, base + CHUNK, side='right'))))
        qq, cc, ns = q[i0:i1], c[i0:i1], nseg[i0:i1]
        seg = _ranges(off[cc], off[cc + 1])
        row = np.repeat(qq, ns)
        ppx, ppy = pts[row, 0], pts[row, 1]
        ay, by = A[seg, 1], B[seg, 1]
        cross = ((ay > ppy) != (by > ppy)) & (ppx < A[seg, 0] + (ppy - ay) * slope[seg])
        odd = (np.add.reduceat(cross, np.cumsum(ns) - ns, dtype=np.int32) & 1).astype(bool)
        hp.append(qq[odd])
        hs.append(cc[odd])
        i0 = i1
    return np.concatenate(hp), np.concatenate(hs)


# ---------------------------------------------------------------------------
# Overlapping shell pairs (UV space, computed once per geometry change)
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
    code = np.unique(np.concatenate(codes))
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


def _find_overlapping_pairs(geo):
    """{(s, c): (label_uv, crossing_points_uv)} for every overlapping shell pair (s < c).

    Borders that cross -> overlap, labeled at the crossing nearest their center.
    No crossing, but a point just inside one shell lies inside the other -> the
    shell is contained in or stacked on the other, labeled at that point.
    Shells that only touch are not overlaps.
    """
    S = geo.nshells
    pairs = {}
    if S < 2:
        return pairs
    A, B, off, seg_shell = geo.seg_a, geo.seg_b, geo.shell_off, geo.seg_shell
    L = np.hypot(B[:, 0] - A[:, 0], B[:, 1] - A[:, 1])

    i, j = _candidate_segment_pairs(A, B, L, seg_shell)
    if i.size:
        k, pts = _pair_crossings(A, B, i, j)
        if k.size:
            code = seg_shell[i[k]] * S + seg_shell[j[k]]
            order = np.argsort(code, kind='stable')
            code, pts = code[order], pts[order]
            starts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
            ends = np.r_[starts[1:], code.size]
            for a0, a1 in zip(starts.tolist(), ends.tolist()):
                grp = pts[a0:a1]
                ctr = grp.mean(axis=0)
                label = grp[int(np.argmin(((grp - ctr) ** 2).sum(axis=1)))].copy()
                if len(grp) > MAX_CROSS_MARKS:
                    grp = grp[np.linspace(0, len(grp) - 1, MAX_CROSS_MARKS).astype(np.int64)]
                c = int(code[a0])
                pairs[(c // S, c % S)] = (label, grp.copy())

    probes, owner = _shell_probes(A, B, geo.seg_sign, off, UV_EPS)
    if len(probes):
        hp, hs = _probe_hits(probes, owner, _shell_grid(geo.bmin, geo.bmax),
                             geo.bmin, geo.bmax, A, B, _slopes(A, B), off)
        for p, c in zip(hp.tolist(), hs.tolist()):
            s = int(owner[p])
            key = (s, c) if s < c else (c, s)
            if key not in pairs:
                pairs[key] = (probes[p].copy(), _EMPTY2)
    return pairs


# ---------------------------------------------------------------------------
# Measurement (texture-pixel space; results stored back in UV space)
# ---------------------------------------------------------------------------

class _Measurement:
    __slots__ = ("p_uv", "q_uv", "dist", "src", "dst",
                 "bp_uv", "bf_uv", "bdist", "baxis", "bsrc",
                 "ov_uv", "pair_label_uv", "cross_uv",
                 "tile_mark_uv", "tile_label_uv", "tile_crossing",
                 "shells_total", "shells_sampled", "samples", "overlap_pairs", "version")

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
        self.pair_label_uv = _EMPTY2
        self.cross_uv = _EMPTY2
        self.tile_mark_uv = _EMPTY2  # where shells cross tile lines
        self.tile_label_uv = _EMPTY2
        self.tile_crossing = 0
        self.shells_total = 0
        self.shells_sampled = 0
        self.samples = 0
        self.overlap_pairs = 0
        self.version = 0


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
    code = np.unique(np.concatenate(codes))
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
    covered = set()
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
            codes = np.unique(np.minimum(a, hs) * S + np.maximum(a, hs))
            covered = set(zip((codes // S).tolist(), (codes % S).tolist()))
            m.ov_uv = P[ov] / scale

        # 2) gap: nearest border point of another shell within the radius, in front of this border
        P3 = P.reshape(-1, n, 2)
        N3 = (-inward[g]).reshape(-1, n, 2)
        ck, cj = _shell_candidates(sh, bmin[sh] - R, bmax[sh] + R, A, B, L, seg_shell, group)
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
            near = np.flatnonzero((room <= R) & ~geo.tile_cross[sh])
            if near.size:
                M = len(A)
                remap = np.full(sh.size, -1, dtype=np.int64)
                remap[near] = np.arange(near.size)
                keep = remap[ck] >= 0
                own = sh[near]
                own_len = off[own + 1] - off[own]
                code = np.unique(np.concatenate((
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

    labels, marks = [], []
    overlapping = set(covered)
    for (s, c), (label_uv, cross_uv) in geo.pairs.items():
        if selected_only and not (geo.shell_sel[s] or geo.shell_sel[c]):
            continue
        if mat is not None and mat[s] != mat[c]:
            continue
        overlapping.add((s, c))
        if len(cross_uv):
            marks.append(cross_uv)
        if (s, c) not in covered:  # no measurement point caught this overlap: label it anyway
            labels.append(label_uv)
    m.overlap_pairs = len(overlapping)
    if labels:
        m.pair_label_uv = np.array(labels, dtype=np.float64).reshape(-1, 2)
    if marks:
        m.cross_uv = np.concatenate(marks)

    if tiles and geo.tile_cross_pts:
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
    return m


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
                      tuple(tuple(r) for r in obj.matrix_world) if want_td else None,
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


def _ensure_geometry(objects, sync, want_sel, want_td=False, want_orient=False):
    """(geometry, stale). Heavy meshes keep the previous result until edits pause."""
    st = _State
    sig = _geometry_signature(objects, sync, want_sel, want_td, want_orient)
    if st.geo is not None and sig == st.geo_sig:
        if not st.geo_dirty:
            return st.geo, False
        if st.last_geo_ms > LIVE_MS:
            wait = DEBOUNCE_S - (time.perf_counter() - st.last_change)
            if wait > 0.0:
                _schedule_redraw(wait + 0.01)
                return st.geo, True
    t0 = time.perf_counter()
    st.geo = _extract(objects, sync, want_sel, want_td, reuse=True, want_orient=want_orient)
    st.last_geo_ms = (time.perf_counter() - t0) * 1000.0
    st.geo_sig = sig
    st.geo_dirty = False
    st.geo_version += 1
    st.meas_cache.clear()
    st.labels_cache.clear()
    st.td_values.clear()
    st.td_fill.clear()
    st.frame.clear()
    return st.geo, False


def _uv_geometry(scene, st, objects):
    """The UV editor's shells (one cache for the overlay and the panels): selection and
    texel density always, orientation while the arrows are shown."""
    return _ensure_geometry(objects, bool(scene.tool_settings.use_uv_select_sync), True, True,
                            bool(st.info_show and st.show_orientation))


def _ensure_measurement(geo, W, H, s):
    """(measurement, stale). Heavy layouts re-measure once a dragged slider pauses."""
    st = _State
    key = (st.geo_version, W, H, int(s.points), round(float(s.shift), 4),
           round(float(s.search_px), 4), bool(s.selected_only), bool(s.tile_border), bool(s.same_material))
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
    meas = _measure(geo, W, H, s.points, s.shift / 100.0, s.search_px, s.selected_only, s.tile_border,
                    s.same_material)
    st.last_meas_ms = (time.perf_counter() - t0) * 1000.0
    st.meas_version += 1
    meas.version = st.meas_version
    st.meas, st.meas_key = meas, key
    st.meas_geo_version = st.geo_version
    st.pending_key = None
    _cache_put(st.meas_cache, key, meas)
    return meas, False


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


def _build_labels(geo, meas, st, fsize, td_info, info=None):
    """`info`: (show object scale, show orientation arrows, show Flipped) for the per-shell info
    blocks; by default Flipped follows the gap overlay and the other two are off."""
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
    anchors, lifted, icons, icon_rgb, lines = [], [], [], [], []

    def add(pts, text, rgb, lift):
        k = len(pts)
        if k:
            anchors.append(np.asarray(pts, dtype=np.float64).reshape(-1, 2))
            lifted.append(np.full(k, 1.0 if lift else 0.0))
            icons.append(np.full((k, 2), np.nan))
            icon_rgb.append(np.zeros((k, 3)))
            rgbs = (np.tile(rgb, (k, 1)) if np.ndim(rgb) == 1 else np.asarray(rgb)).tolist()
            txts = [text] * k if isinstance(text, str) else text
            lines.extend([[(t, tuple(c))] for t, c in zip(txts, rgbs)])

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
        order = np.concatenate((np.flatnonzero(has & warn), np.flatnonzero(has & ~warn)))
        if order.size:
            obj = geo.sh_obj.tolist()
            for i in order.tolist():
                ln = []
                if td_ok[i]:
                    ln.append((td_txt[i], td_col[i]))
                if scale_on[i]:
                    ln.append(("Scale " + otxt[obj[i]], SCALE_RGB))
                if flip[i]:
                    ln.append(("Flipped", c_fl))
                lines.append(ln)
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

    nlab = len(lines)
    lab.anchor = np.concatenate(anchors) if anchors else _EMPTY2
    lab.lifted = np.concatenate(lifted) if lifted else _EMPTY_F
    lab.icon = np.concatenate(icons) if icons else _EMPTY2
    lab.icon_rgb = np.concatenate(icon_rgb) if icon_rgb else np.empty((0, 3))
    counts = np.asarray([len(x) for x in lines], dtype=np.int64)
    lab.nlines = counts
    flat = [t for x in lines for t in x]
    lab.text = [t for t, _c in flat]
    lab.rgb = np.asarray([c for _t, c in flat], dtype=np.float64).reshape(-1, 3)
    lab.line_of = np.repeat(np.arange(nlab, dtype=np.int64), counts)
    lab.row = np.arange(len(flat), dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    widths = {t: float(blf.dimensions(FONT_ID, t)[0]) for t in set(lab.text)}
    maxw = np.zeros(nlab)
    if flat:
        np.maximum.at(maxw, lab.line_of, np.asarray([widths[t] for t in lab.text], dtype=np.float64))
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


def _place_labels(lab, anchor_px, lift, pad, rw, rh, declutter):
    """Labels to draw (indices, priority order) and, per label, its box, the left end of its
    text, its first baseline and its center."""
    ax = anchor_px[:, 0]
    ay = anchor_px[:, 1] + lab.lifted * lift
    hw = lab.width * 0.5
    th = lab.th
    tall = th + (np.maximum(lab.nlines, 1) - 1) * lab.lh  # first line's top to last line's baseline
    x0, x1 = ax - hw - pad, ax + hw + pad
    y0, y1 = ay - tall * 0.5 - th * 0.3 - pad * 0.5, ay + tall * 0.5 + pad
    idx = np.flatnonzero((x1 >= 0.0) & (x0 <= rw) & (y1 >= 0.0) & (y0 <= rh))
    if declutter and idx.size > 1:
        # coarse pass: the first label (by priority) in each label-sized screen cell
        cw = max(1.0, float(np.median(x1[idx] - x0[idx])))
        ch = max(1.0, float(np.median(y1[idx] - y0[idx])))
        cell_key = ((np.floor(ax[idx] / cw).astype(np.int64) + _KEY_OFF) * _KEY_MUL +
                    np.floor(ay[idx] / ch).astype(np.int64) + _KEY_OFF)
        _unique, first = np.unique(cell_key, return_index=True)
        idx = idx[np.sort(first)]
        # exact pass over the survivors
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


def _frame_data(st, geo, meas, lab, mapping, rw, rh, alpha, dot, xh, lift, margin, pad, hl=None):
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
        if st.show_flipped and geo.flipped.any():
            fl_rgb = tuple(st.color_flipped)
            if geo.flip_tris.size:
                T = to_region(geo.flip_tris.reshape(-1, 2)).reshape(-1, 3, 2)
                vis = ((T[:, :, 0].max(axis=1) >= 0.0) & (T[:, :, 0].min(axis=1) <= rw) &
                       (T[:, :, 1].max(axis=1) >= 0.0) & (T[:, :, 1].min(axis=1) <= rh))
                if vis.any():
                    T = T[vis].reshape(-1, 2)
                    v = np.zeros((T.shape[0], 3), dtype=np.float32)
                    v[:, :2] = T
                    under_p.append(v)
                    under_c.append(np.tile(np.array(fl_rgb + (0.22 * alpha,), dtype=np.float32), (T.shape[0], 1)))
            seg = geo.flipped[geo.seg_shell]
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


def _draw_overlay(context, region, st, geo, meas, W, H, stale):
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
           tuple(st.color_overlap), tuple(st.color_flipped), td_key, info)
    lab = _State.labels_cache.get(key)
    if lab is None:
        lab = _build_labels(geo, meas, st, fsize, td_info, info)
        _cache_put(_State.labels_cache, key, lab)

    fkey = (mapping, rw, rh, key, alpha, lw, dot, xh, lift, margin, pad,
            bool(st.declutter), bool(st.label_background), bool(st.show_flipped), hl_key)
    rkey = region.as_pointer() if hasattr(region, "as_pointer") else id(region)
    hit = _State.frame.get(rkey)
    if hit is not None and hit[0] == fkey:
        fr = hit[1]
    else:
        fr = _frame_data(st, geo, meas, lab, mapping, rw, rh, alpha, dot, xh, lift, margin, pad, hl)
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
    geo, stale_geo = _uv_geometry(scene, st, objects)
    W, H, _src = _resolve_resolution(st, space)
    _publish_edit(scene, st, geo, W, H, objects)
    if geo.nshells == 0 and geo.sh_n == 0:
        return
    meas, stale_meas = None, False
    if st.show and geo.nshells:
        meas, stale_meas = _ensure_measurement(geo, W, H, st)
    _draw_overlay(context, region, st, geo, meas, W, H, stale_geo or stale_meas)


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
        A = _object_arrays(obj, True)
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
    if edit:
        gen = ('E', mptr, _State.data_gen)
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
        world = np.linalg.norm(e.face_nvec @ _cofactor(M3).T, axis=1)
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

def _mark_changed():
    _State.geo_dirty = True
    _State.data_gen += 1
    _State.last_change = time.perf_counter()


@persistent
def _on_depsgraph_update(*args):
    depsgraph = args[1] if len(args) > 1 else None
    if depsgraph is None:
        _mark_changed()
        return
    changed = False
    for upd in depsgraph.updates:
        idb = upd.id
        if isinstance(idb, bpy.types.Mesh):
            if idb.name == _SCRATCH_MESH:
                continue
        elif not (isinstance(idb, bpy.types.Object) and idb.type == 'MESH'):
            continue
        changed = True
        if getattr(upd, "is_updated_geometry", True):  # per-object change count (3D view cache)
            orig = getattr(idb, "original", None) or idb
            ptr = orig.as_pointer()
            _State.id_gen[ptr] = _State.id_gen.get(ptr, 0) + 1
    if changed:
        _mark_changed()


@persistent
def _on_undo_redo(*_args):
    # undo can replace data-blocks (and their memory addresses): drop everything read
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
        _State.invalidate()
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


def _fresh_arrays(obj, sync, want_3d=False):
    """(arrays, faces shown in the UV editor) of an edit-mode object, read now (operators)."""
    A = _read_edit_arrays(obj, want_3d)
    if A is None:
        return None
    return A, (~A.f_hide if sync else (~A.f_hide & A.f_sel))


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
        sync = bool(context.scene.tool_settings.use_uv_select_sync)
        count = 0
        for obj in _edit_mesh_objects(context):
            me = obj.data
            bm = bmesh.from_edit_mesh(me)
            uvl = bm.loops.layers.uv.active
            if uvl is None:
                continue
            isl = _uv_islands(bm, uvl, sync, keep_faces=True)
            area = [0.0] * isl.nshell
            for i, s in enumerate(isl.face_shell):
                area[s] += isl.face_area2[i]
            flipped = [a < 0.0 for a in area]
            count += sum(flipped)
            picked = [f for f, s in zip(isl.faces, isl.face_shell) if flipped[s]]
            _select_uv_faces(bm, uvl, sync, isl.faces, picked, self.extend)
            bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
        _State.invalidate()
        _redraw_uv_editors()
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
        sync = bool(context.scene.tool_settings.use_uv_select_sync)
        count = 0
        for obj in _edit_mesh_objects(context):
            got = _fresh_arrays(obj, sync)
            if got is None:
                continue
            A, vis = got
            fidx, fsh = _material_shell_faces(obj, A, _shells(A, vis), keys)  # the shells shown
            count += int(np.unique(fsh).size)
            me = obj.data
            bm = bmesh.from_edit_mesh(me)
            bm.faces.ensure_lookup_table()
            faces = bm.faces
            _select_uv_faces(bm, bm.loops.layers.uv.active, sync,
                             [faces[i] for i in np.flatnonzero(vis).tolist()],
                             [faces[i] for i in fidx.tolist()], self.extend)
            bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
        _State.invalidate()
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
        todo = []
        for obj in _edit_mesh_objects(context):
            got = _fresh_arrays(obj, sync)
            if got is None:
                continue
            A, vis = got
            shown, _fsh = _material_shell_faces(obj, A, _shells(A, vis), keys)
            todo.append((obj, A, shown))
        action = self.action
        if action == 'TOGGLE':
            action = 'HIDE' if any(shown.size for _o, _a, shown in todo) else 'REVEAL'
        changed = 0
        for obj, A, shown in todo:
            if action == 'REVEAL':  # whole shells, hidden faces included, decide the material
                fidx, _fsh = _material_shell_faces(obj, A, _shells(A, np.ones(A.f_total.size, dtype=bool)), keys)
            else:
                fidx = shown
            if not fidx.size:
                continue
            me = obj.data
            bm = bmesh.from_edit_mesh(me)
            bm.faces.ensure_lookup_table()
            faces = bm.faces
            changed += _hide_reveal(bm, bm.loops.layers.uv.active, sync, [faces[i] for i in fidx.tolist()],
                                    action == 'REVEAL')
            bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
        _State.invalidate()
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
        per = []
        for obj in _edit_mesh_objects(context):
            got = _fresh_arrays(obj, sync, want_3d=True)
            if got is None:
                continue
            A, vis = got
            S = _shells(A, vis)
            if not S.fidx.size:
                continue
            uv = np.bincount(S.face_shell, weights=np.abs(S.area2) * 0.5, minlength=S.nshell)
            M3 = np.array(obj.matrix_world.to_3x3(), dtype=np.float64)
            world = np.linalg.norm(_face_area_vectors(S, A) @ _cofactor(M3).T, axis=1)
            td = _td_values(uv, np.bincount(S.face_shell, weights=world, minlength=S.nshell), W, H, scale)
            per.append((obj, S, vis, td))
        bounds = _td_range_bounds(np.concatenate([p[3] for p in per]), st.td_sel_from / 100.0,
                                  st.td_sel_to / 100.0) if per else None
        if bounds is None:
            self.report({'WARNING'}, "No shells with a texel density")
            return {'CANCELLED'}
        count = 0
        for obj, S, vis, td in per:
            with np.errstate(invalid='ignore'):
                inside = (td >= bounds[0]) & (td <= bounds[1])
            count += int(np.count_nonzero(inside))
            me = obj.data
            bm = bmesh.from_edit_mesh(me)
            bm.faces.ensure_lookup_table()
            faces = bm.faces
            _select_uv_faces(bm, bm.loops.layers.uv.active, sync,
                             [faces[i] for i in np.flatnonzero(vis).tolist()],
                             [faces[i] for i in S.fidx[inside[S.face_shell]].tolist()], self.extend)
            bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
        _State.invalidate()
        _redraw_areas(('IMAGE_EDITOR', 'VIEW_3D'))
        f, u = _TD_FACTOR.get(st.td_unit, 1.0), _TD_LABEL.get(st.td_unit, "px/m")
        self.report({'INFO'}, "%d shell%s from %s to %s selected" % (
            count, "" if count == 1 else "s", _fmt_td(max(bounds[0], 0.0) * f, u), _fmt_td(bounds[1] * f, u)))
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
            col.label(text="Overlapping pairs: %d" % meas.overlap_pairs,
                      icon='ERROR' if meas.overlap_pairs else 'CHECKMARK')
            col.label(text="Gaps below minimal: %d, below needed: %d" % (crit, warn))
            if st.tile_border:
                bd = meas.bdist
                bcrit = int(np.count_nonzero(bd < st.border_min_px))
                bwarn = int(np.count_nonzero((bd >= st.border_min_px) & (bd < st.border_needed_px)))
                col.label(text="Tile border below minimal: %d, below needed: %d" % (bcrit, bwarn))
                if meas.tile_crossing:
                    col.label(text="Shells crossing a tile border: %d" % meas.tile_crossing, icon='ERROR')
            nflip = int(np.count_nonzero(geo.flipped))
            col.label(text="Flipped shells: %d" % nflip, icon='ERROR' if nflip else 'CHECKMARK')
        layout.operator(UVGAP_OT_refresh.bl_idname, icon='FILE_REFRESH')


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
    UVGAP_OT_material_select,
    UVGAP_OT_material_visibility,
    UVGAP_OT_material_check,
    UVGAP_OT_td_fill,
    UVGAP_OT_select_td_range,
    UVGAP_UL_materials,
    UVGAP_PT_main,
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
