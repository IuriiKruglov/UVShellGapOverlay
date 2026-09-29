# SPDX-License-Identifier: GPL-3.0-or-later
#
# UV Shell Gap Overlay  -  Blender add-on for 3.6 LTS to 5.2 LTS
#
# Draws, along the shell borders in the UV Editor, the pixel distance between
# neighboring UV shells and from shells to their UDIM tile border, graded
# against minimal / needed padding; marks overlapping, tile-crossing and
# flipped (mirrored) shells.
#
# Authors: Iurii Kruglov & Claude (Anthropic)

bl_info = {
    "name": "UV Shell Gap Overlay",
    "author": "Iurii Kruglov, Claude (Anthropic)",
    "version": (1, 2, 0),
    "blender": (3, 6, 0),
    "location": "UV Editor > Sidebar (N) > UV Gaps",
    "description": "Pixel distances between neighboring UV shells and to UDIM tile borders, "
                   "graded against minimal / needed padding; overlap, tile-crossing and flipped-shell checks",
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
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
)
from gpu_extras.batch import batch_for_shader


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
FONT_ID = 0

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
    geo = None
    geo_sig = None
    geo_dirty = True
    geo_version = 0
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
    shader_names = {}
    numpy_buffers = True
    last_error = None

    @classmethod
    def invalidate(cls):
        cls.geo_dirty = True
        cls.geo_sig = None
        cls.meas_key = None
        cls.meas_cache.clear()
        cls.labels_cache.clear()

    @classmethod
    def reset(cls):
        cls.invalidate()
        cls.geo = None
        cls.meas = None
        cls.last_geo_ms = 0.0
        cls.last_meas_ms = 0.0
        cls.pending_key = None


def _cache_put(cache, key, value, size=4):
    cache[key] = value
    while len(cache) > size:
        cache.pop(next(iter(cache)))


def _redraw_uv_editors(*_args):
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for win in wm.windows:
        screen = win.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()


def _on_setting_changed(self, context):
    _redraw_uv_editors()


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


def _uv_islands(bm, uvl, sync, want_sel=False, keep_faces=False):
    """UV vertices (a mesh vertex + its UV), per-face UV vertex ids, twice the signed UV
    area of each face (negative = clockwise = mirrored), and shells: faces sharing a UV
    vertex belong to one shell. Border edges are the UV edges used by a single face."""
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
    elif _UV_SELECT_ON_LOOP and (not sync or getattr(bm, "uv_select_sync_valid", False)):
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


def _sign(x):
    return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)


def _extract_bmesh(bm, uvl, sync, want_sel, acc):
    """Append the shells of one edit-mesh to the accumulator; returns visible face count."""
    isl = _uv_islands(bm, uvl, sync, want_sel)
    nf = len(isl.face_shell)
    if nf == 0:
        return 0
    vx, vy = isl.vx, isl.vy
    face_shell, face_area2 = isl.face_shell, isl.face_area2
    nshell = isl.nshell

    edges_by_shell = [[] for _ in range(nshell)]
    for rec in isl.edge_owner.values():
        if rec is not None:
            f_idx, a, b = rec
            edges_by_shell[face_shell[f_idx]].append((a, b, _sign(face_area2[f_idx])))
    shell_area2 = [0.0] * nshell
    for i in range(nf):
        shell_area2[face_shell[i]] += face_area2[i]
    all_selected = (not want_sel) or (not isl.sel_readable)
    shell_sel = [all_selected] * nshell
    if not all_selected:
        for i in range(nf):
            if isl.face_sel[i]:
                shell_sel[face_shell[i]] = True

    ax, ay, bx, by, sg, off, sel, area, tris, tri_shell = acc
    flipped_gid = {}
    for s in range(nshell):
        edges = edges_by_shell[s]
        if not edges:
            continue  # closed UV surface with no border: nothing to measure
        gid = len(off) - 1
        for a, b, sgn in _chain_border(edges):
            ax.append(vx[a])
            ay.append(vy[a])
            bx.append(vx[b])
            by.append(vy[b])
            sg.append(sgn)
        off.append(len(ax))
        sel.append(shell_sel[s])
        area.append(shell_area2[s])
        if shell_area2[s] < 0.0:
            flipped_gid[s] = gid

    if flipped_gid:  # fan triangles of mirrored shells, for the highlight fill
        face_ids = isl.face_ids
        for i in range(nf):
            gid = flipped_gid.get(face_shell[i])
            if gid is None:
                continue
            ids = face_ids[i]
            a0 = ids[0]
            for k in range(1, len(ids) - 1):
                b1, c1 = ids[k], ids[k + 1]
                tris.append((vx[a0], vy[a0], vx[b1], vy[b1], vx[c1], vy[c1]))
                tri_shell.append(gid)
    return nf


class _Geometry:
    __slots__ = ("seg_a", "seg_b", "seg_sign", "seg_shell", "shell_off", "shell_sel",
                 "bmin", "bmax", "pairs", "nshells", "nfaces",
                 "shell_area", "flipped", "flip_tris", "flip_tri_shell",
                 "tile", "tile_cross", "tile_cross_pts")

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


def _extract(objects, sync, want_sel):
    acc = ([], [], [], [], [], [0], [], [], [], [])
    nfaces = 0
    for obj in objects:
        me = obj.data
        if not me.is_editmode:
            continue
        bm = bmesh.from_edit_mesh(me)
        uvl = bm.loops.layers.uv.active
        if uvl is None:
            continue
        nfaces += _extract_bmesh(bm, uvl, sync, want_sel, acc)

    ax, ay, bx, by, sg, off, sel, area, tris, tri_shell = acc
    geo = _Geometry()
    geo.nfaces = nfaces
    S = len(off) - 1
    if S == 0:
        return geo
    geo.seg_a = np.column_stack((np.asarray(ax, dtype=np.float64), np.asarray(ay, dtype=np.float64)))
    geo.seg_b = np.column_stack((np.asarray(bx, dtype=np.float64), np.asarray(by, dtype=np.float64)))
    geo.seg_sign = np.asarray(sg, dtype=np.float64)
    geo.shell_off = np.asarray(off, dtype=np.int64)
    geo.shell_sel = np.asarray(sel, dtype=bool)
    geo.seg_shell = np.repeat(np.arange(S, dtype=np.int64), np.diff(geo.shell_off))
    lo = np.minimum(geo.seg_a, geo.seg_b)
    hi = np.maximum(geo.seg_a, geo.seg_b)
    geo.bmin = np.minimum.reduceat(lo, geo.shell_off[:-1], axis=0)
    geo.bmax = np.maximum.reduceat(hi, geo.shell_off[:-1], axis=0)
    geo.nshells = S
    geo.shell_area = np.asarray(area, dtype=np.float64) * 0.5
    geo.flipped = geo.shell_area < 0.0
    geo.flip_tris = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 2)
    geo.flip_tri_shell = np.asarray(tri_shell, dtype=np.int64)
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
    (in the same tile when tile_key is given) whose bounding box meets [lo[k], hi[k]]."""
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


def _measure(geo, W, H, points, shift, radius, selected_only, tiles=False):
    m = _Measurement()
    S = geo.nshells
    m.shells_total = S
    if S == 0:
        return m
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
        ck, cj = _shell_candidates(sh, bmin[sh] - R, bmax[sh] + R, A, B, L, seg_shell, tile_key)
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

def _geometry_signature(objects, sync, want_sel):
    parts = []
    for obj in objects:
        bm = bmesh.from_edit_mesh(obj.data)
        uvl = bm.loops.layers.uv.active
        parts.append((obj.data.as_pointer(), len(bm.verts), len(bm.faces),
                      uvl.name if uvl is not None else ""))
    return (sync, want_sel, tuple(parts))


def _redraw_timer():
    _redraw_uv_editors()
    return None


def _schedule_redraw(delay):
    try:
        if not bpy.app.timers.is_registered(_redraw_timer):
            bpy.app.timers.register(_redraw_timer, first_interval=max(0.02, delay))
    except Exception:
        pass


def _ensure_geometry(objects, sync, want_sel):
    """(geometry, stale). Heavy meshes keep the previous result until edits pause."""
    st = _State
    sig = _geometry_signature(objects, sync, want_sel)
    if st.geo is not None and sig == st.geo_sig:
        if not st.geo_dirty:
            return st.geo, False
        if st.last_geo_ms > LIVE_MS:
            wait = DEBOUNCE_S - (time.perf_counter() - st.last_change)
            if wait > 0.0:
                _schedule_redraw(wait + 0.01)
                return st.geo, True
    t0 = time.perf_counter()
    st.geo = _extract(objects, sync, want_sel)
    st.last_geo_ms = (time.perf_counter() - t0) * 1000.0
    st.geo_sig = sig
    st.geo_dirty = False
    st.geo_version += 1
    st.meas_cache.clear()
    st.labels_cache.clear()
    return st.geo, False


def _ensure_measurement(geo, W, H, s):
    """(measurement, stale). Heavy layouts re-measure once a dragged slider pauses."""
    st = _State
    key = (st.geo_version, W, H, int(s.points), round(float(s.shift), 4),
           round(float(s.search_px), 4), bool(s.selected_only), bool(s.tile_border))
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
    meas = _measure(geo, W, H, s.points, s.shift / 100.0, s.search_px, s.selected_only, s.tile_border)
    st.last_meas_ms = (time.perf_counter() - t0) * 1000.0
    st.meas_version += 1
    meas.version = st.meas_version
    st.meas, st.meas_key = meas, key
    st.meas_geo_version = st.geo_version
    st.pending_key = None
    _cache_put(st.meas_cache, key, meas)
    return meas, False


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


def _batch(shader, prim, pos, col):
    """Batch from float32 arrays (fast buffer path), falling back to Python lists."""
    if _State.numpy_buffers:
        try:
            return batch_for_shader(shader, prim, {
                "pos": np.ascontiguousarray(pos, dtype=np.float32),
                "color": np.ascontiguousarray(col, dtype=np.float32)})
        except Exception:
            _State.numpy_buffers = False
    return batch_for_shader(shader, prim, {"pos": pos.tolist(), "color": col.tolist()})


def _draw_lines(pos, col, width, rw, rh):
    name, sh = _builtin('line')
    if sh is None or not len(pos):
        return
    batch = _batch(sh, 'LINES', pos, col)
    sh.bind()
    polyline = 'POLYLINE' in name
    if polyline:
        sh.uniform_float("viewportSize", (rw, rh))
        sh.uniform_float("lineWidth", width)
    else:
        gpu.state.line_width_set(width)
    batch.draw(sh)
    if not polyline:
        gpu.state.line_width_set(1.0)


def _draw_tris(pos, col):
    _name, sh = _builtin('fill')
    if sh is None or not len(pos):
        return
    batch = _batch(sh, 'TRIS', pos, col)
    sh.bind()
    batch.draw(sh)


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
    """Label text, color and size in priority order, plus line colors; rebuilt only when
    the measurement or the style changes."""
    __slots__ = ("anchor", "lifted", "text", "rgb", "width", "th", "gap_rgb", "border_rgb")


def _build_labels(geo, meas, st, fsize):
    lab = _Labels()
    blf.size(FONT_ID, fsize)
    lab.th = float(blf.dimensions(FONT_ID, "0123456789")[1])
    c0 = np.array(st.color_zero[:], dtype=np.float64)
    c1 = np.array(st.color_min[:], dtype=np.float64)
    c2 = np.array(st.color_needed[:], dtype=np.float64)
    c_ov = np.array(st.color_overlap[:], dtype=np.float64)
    c_fl = np.array(st.color_flipped[:], dtype=np.float64)
    lab.gap_rgb = _gradient(meas.dist, st.min_px, st.needed_px, c0, c1, c2)
    lab.border_rgb = _gradient(meas.bdist, st.border_min_px, st.border_needed_px, c0, c1, c2)

    # problems first (tile crossings, overlaps, flipped shells), then distances by severity
    anchors, lifted, texts, colors = [], [], [], []

    def add(pts, text, rgb, lift):
        if len(pts):
            anchors.append(pts)
            lifted.append(np.full(len(pts), 1.0 if lift else 0.0))
            texts.extend([text] * len(pts))
            colors.append(np.tile(rgb, (len(pts), 1)))

    add(meas.tile_label_uv, "Crosses tile", c_ov, True)
    add(meas.pair_label_uv, "Overlap", c_ov, True)
    add(meas.ov_uv, "Overlap", c_ov, True)
    if st.show_flipped and geo.flipped.any():
        fl = np.flatnonzero(geo.flipped)
        add((geo.bmin[fl] + geo.bmax[fl]) * 0.5, "Flipped", c_fl, False)

    d_all = np.concatenate((meas.dist, meas.bdist))
    if d_all.size:
        severity = np.concatenate((meas.dist / max(float(st.needed_px), 1e-6),
                                   meas.bdist / max(float(st.border_needed_px), 1e-6)))
        order = np.argsort(severity, kind='stable')
        mids = np.concatenate(((meas.p_uv + meas.q_uv) * 0.5, (meas.bp_uv + meas.bf_uv) * 0.5))
        anchors.append(mids[order])
        lifted.append(np.zeros(d_all.size))
        texts.extend(_fmt_px(x) for x in d_all[order].tolist())
        colors.append(np.concatenate((lab.gap_rgb, lab.border_rgb))[order])

    lab.anchor = np.concatenate(anchors) if anchors else _EMPTY2
    lab.lifted = np.concatenate(lifted) if lifted else _EMPTY_F
    lab.text = texts
    lab.rgb = np.concatenate(colors) if colors else np.empty((0, 3))
    widths = {t: float(blf.dimensions(FONT_ID, t)[0]) for t in set(texts)}
    lab.width = np.array([widths[t] for t in texts], dtype=np.float64)
    return lab


def _clashes(grid, gx0, gx1, gy0, gy1, x0, y0, x1, y1):
    for gx in range(gx0, gx1 + 1):
        for gy in range(gy0, gy1 + 1):
            for r in grid.get((gx, gy), ()):
                if x0 < r[2] and x1 > r[0] and y0 < r[3] and y1 > r[1]:
                    return True
    return False


def _place_labels(lab, anchor_px, lift, pad, rw, rh, declutter):
    """Indices of labels to draw (priority order) plus their anchors and boxes."""
    ax = anchor_px[:, 0]
    ay = anchor_px[:, 1] + lab.lifted * lift
    hw = lab.width * 0.5
    th = lab.th
    x0, x1 = ax - hw - pad, ax + hw + pad
    y0, y1 = ay - th * 0.8 - pad * 0.5, ay + th * 0.5 + pad
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
    return idx, ax, ay, np.column_stack((x0, y0, x1, y1))


def _draw_texts(lab, idx, ax, ay, fsize, alpha, shadow):
    blf.size(FONT_ID, fsize)
    if shadow:
        blf.enable(FONT_ID, blf.SHADOW)
        blf.shadow(FONT_ID, 3, 0.0, 0.0, 0.0, min(1.0, alpha))
        blf.shadow_offset(FONT_ID, 1, -1)
    try:
        xs = (ax[idx] - lab.width[idx] * 0.5).tolist()
        ys = (ay[idx] - lab.th * 0.5).tolist()
        cols = lab.rgb[idx].tolist()
        text = lab.text
        for k, i in enumerate(idx.tolist()):
            r, g, b = cols[k]
            blf.color(FONT_ID, r, g, b, alpha)
            blf.position(FONT_ID, xs[k], ys[k], 0.0)
            blf.draw(FONT_ID, text[i])
    finally:
        if shadow:
            blf.disable(FONT_ID, blf.SHADOW)


def _draw_overlay(context, region, st, geo, meas, stale):
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

    key = (meas.version, _State.geo_version, round(fsize, 3), bool(st.show_flipped),
           float(st.min_px), float(st.needed_px), float(st.border_min_px), float(st.border_needed_px),
           tuple(st.color_zero), tuple(st.color_min), tuple(st.color_needed),
           tuple(st.color_overlap), tuple(st.color_flipped))
    lab = _State.labels_cache.get(key)
    if lab is None:
        lab = _build_labels(geo, meas, st, fsize)
        _cache_put(_State.labels_cache, key, lab)

    def to_region(uv):
        return np.column_stack(((uv[:, 0] - u0) * sx, (uv[:, 1] - v0) * sy))

    under_p, under_c = [], []
    lines_p, lines_c, fills_p, fills_c = [], [], [], []
    ov_rgba = np.array(tuple(st.color_overlap) + (alpha,), dtype=np.float32)

    # flipped shells: tinted fill underneath everything, outline on top
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

    # problems: crosses on points inside another shell, where overlapping borders cross,
    # and where shells cross a tile line
    for uv in (meas.ov_uv, meas.cross_uv, meas.tile_mark_uv):
        if uv.size:
            X = to_region(uv)
            X = X[_in_region(X, rw, rh, margin)]
            if X.size:
                lines_p.append(_cross_verts(X, xh))
                lines_c.append(np.tile(ov_rgba, (4 * X.shape[0], 1)))

    idx = _EMPTY_I
    if lab.text:
        idx, ax, ay, boxes = _place_labels(lab, to_region(lab.anchor), lift, pad, rw, rh, st.declutter)

    gpu.state.blend_set('ALPHA')
    try:
        if under_p:
            _draw_tris(np.concatenate(under_p), np.concatenate(under_c))
        if lines_p:
            _draw_lines(np.concatenate(lines_p), np.concatenate(lines_c), lw, rw, rh)
        if fills_p:
            _draw_tris(np.concatenate(fills_p), np.concatenate(fills_c))
        if idx.size:
            if st.label_background:
                bg = np.array((0.0, 0.0, 0.0, 0.85 * alpha), dtype=np.float32)
                _draw_tris(_rect_verts(boxes[idx]), np.tile(bg, (6 * idx.size, 1)))
            _draw_texts(lab, idx, ax, ay, fsize, alpha, shadow=not st.label_background)
    finally:
        gpu.state.blend_set('NONE')


def _draw_main(context):
    scene = getattr(context, "scene", None)
    st = getattr(scene, "uv_gap_overlay", None) if scene is not None else None
    if st is None or not st.show:
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
    sync = bool(scene.tool_settings.use_uv_select_sync)
    geo, stale_geo = _ensure_geometry(objects, sync, bool(st.selected_only))
    if geo.nshells == 0:
        return
    W, H, _src = _resolve_resolution(st, space)
    meas, stale_meas = _ensure_measurement(geo, W, H, st)
    _draw_overlay(context, region, st, geo, meas, stale_geo or stale_meas)


def _draw_callback():
    try:
        _draw_main(bpy.context)
    except Exception:
        err = traceback.format_exc()
        if err != _State.last_error:  # report each distinct error once, not every redraw
            _State.last_error = err
            print("[UV Shell Gap Overlay] draw error:\n" + err)


# ---------------------------------------------------------------------------
# Change tracking
# ---------------------------------------------------------------------------

@persistent
def _on_depsgraph_update(*args):
    if _State.geo is None:
        return
    depsgraph = args[1] if len(args) > 1 else None
    if depsgraph is None:
        _State.geo_dirty = True
        _State.last_change = time.perf_counter()
        return
    for upd in depsgraph.updates:
        idb = upd.id
        if isinstance(idb, bpy.types.Mesh) or (isinstance(idb, bpy.types.Object) and idb.type == 'MESH'):
            _State.geo_dirty = True
            _State.last_change = time.perf_counter()
            return


@persistent
def _on_undo_redo(*_args):
    _State.geo_dirty = True
    _State.last_change = time.perf_counter()


@persistent
def _on_load(*_args):
    _State.reset()


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
            if sync:
                if self.extend and _UV_SELECT_ON_LOOP and bm.uv_select_sync_valid:
                    # Blender 5.0+: add to the synced UV selection, then carry it over to the mesh
                    bm.uv_select_foreach_set_from_mesh(True, faces=picked)
                    bm.uv_select_sync_to_mesh()
                else:
                    if not self.extend:
                        for f in isl.faces:
                            f.select_set(False)
                    for f in picked:
                        f.select_set(True)
                    bm.select_flush_mode()
                    if _UV_SELECT_ON_LOOP:
                        bm.uv_select_sync_valid = False  # the UV selection follows the new mesh selection
            elif _UV_SELECT_ON_LOOP:
                # Blender 5.0+: per-face setters also clear/set the face corners (flushed down)
                for f, s in zip(isl.faces, isl.face_shell):
                    if flipped[s] or not self.extend:
                        f.uv_select_set(flipped[s])
                bm.uv_select_flush_mode()
            else:
                for f, s in zip(isl.faces, isl.face_shell):
                    on = flipped[s]
                    if on or not self.extend:
                        for loop in f.loops:
                            luv = loop[uvl]
                            luv.select = on
                            luv.select_edge = on
            bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
        _State.invalidate()
        _redraw_uv_editors()
        self.report({'INFO'}, "%d flipped shell%s selected" % (count, "" if count == 1 else "s"))
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

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

        col = layout.column()
        col.prop(st, "resolution")
        W, H, src = _resolve_resolution(st, context.space_data)
        if st.resolution == 'CUSTOM' or (st.resolution == 'IMAGE' and src is None):
            sub = col.column(align=True)
            sub.prop(st, "res_x")
            sub.prop(st, "res_y")
        if st.resolution == 'IMAGE':
            if src:
                col.label(text="%s  (%d x %d)" % (src, W, H), icon='IMAGE_DATA')
            else:
                col.label(text="No image here: using Width / Height", icon='INFO')

        layout.separator()
        col = layout.column(align=True)
        col.prop(st, "points", slider=True)
        col.prop(st, "shift", slider=True)

        layout.separator()
        col = layout.column(align=True)
        col.prop(st, "min_px")
        col.prop(st, "needed_px")
        col.prop(st, "search_px")
        layout.prop(st, "selected_only")

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
        layout.active = st.show
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


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = (
    UVGAP_Settings,
    UVGAP_OT_refresh,
    UVGAP_OT_select_flipped,
    UVGAP_PT_main,
    UVGAP_PT_tiles,
    UVGAP_PT_display,
    UVGAP_PT_colors,
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
    _State.numpy_buffers = True
    _State.draw_handle = bpy.types.SpaceImageEditor.draw_handler_add(
        _draw_callback, (), 'WINDOW', 'POST_PIXEL')
    for name, fn in _handler_lists:
        handlers = getattr(bpy.app.handlers, name)
        if fn not in handlers:
            handlers.append(fn)


def unregister():
    if _State.draw_handle is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_State.draw_handle, 'WINDOW')
        _State.draw_handle = None
    for name, fn in _handler_lists:
        handlers = getattr(bpy.app.handlers, name)
        if fn in handlers:
            handlers.remove(fn)
    if bpy.app.timers.is_registered(_redraw_timer):
        bpy.app.timers.unregister(_redraw_timer)
    del bpy.types.Scene.uv_gap_overlay
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
    _State.reset()
    _redraw_uv_editors()


if __name__ == "__main__":
    register()
