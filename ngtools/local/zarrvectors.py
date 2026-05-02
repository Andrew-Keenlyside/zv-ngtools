"""
ngtools/local/zarrvectors.py

Serves zarr-vectors data as Neuroglancer precomputed resources.

Contains:
  - Pure-logic functions: URL parsers, encoders, info/chunk/by_id/mesh
    serving functions. Framework-agnostic — take an array, return bytes/dicts.
  - HTTP handler classes (ZarrVectorsStreamlineHandler, ZarrVectorsPointHandler,
    ZarrVectorsMeshHandler) — exposed over the fileserver via route
    registration in ngtools/local/viewer.py.
  - Byte-sized LRU caches with reference-counted eviction wired to
    Scene.load / Scene.unload.
  - Reload command hooks: reload_zv_array / reload_all_zv_caches.

RESOURCE TYPES
--------------
- Annotations (streamlines, points) in `neuroglancer_annotations_v1` format:
  spatial chunks, by_id, multi-level with prominence filter.
- Legacy mesh + optional `segment_properties/info` subresource.

ENTRY POINTS
------------
Pure logic (used internally by the HTTP handlers, or callable directly):
  parse_url(path) -> (protocol, store_path, array_name, resource)
  build_info(array, annotation_type, store_path, array_name)
  build_mesh_info(array)
  build_segment_properties_info(array)
  serve_spatial_chunk(array, level, chunk_key, ..., store_path, array_name)
  serve_annotation_by_id(array, segment_id, ..., store_path, array_name)
  serve_mesh_manifest(array, segment_id)
  serve_mesh_fragment(array, segment_id, ..., store_path, array_name)

Client-side URL fan-out (called by Scene.load):
  parse_zarrvectors_url(url)
  expand_zarrvectors_url(url, fileserver_base, *, include_mesh=True)

Cache lifecycle (called by Scene.load/unload):
  register_array_usage(store_path, array_name)
  unregister_array_usage(store_path, array_name)
  configure_caches(info_max_bytes=..., geometry_max_bytes=..., ...)

Reload command hooks:
  reload_zv_array(store_path, array_name)
  reload_all_zv_caches()

CACHES
------
Three module-level LRU caches with byte-based size caps:
  _info_cache        (default 4 MiB)
  _geometry_cache    (default 2 GiB)
  _by_id_index_cache (default 256 MiB)

All caches key by `(store_path, array_name, ...)`. Reference counts per
`(store_path, array_name)` are tracked in `_array_refcount`; when a key
reaches zero, its cache entries are evicted.

If `array.mtime()` is available, each cached-path request runs a staleness
check — if mtime has advanced, caches for the array are evicted. Disable
with `configure_caches(enable_mtime_check=False)` for remote-store-heavy
deployments.

When PR #38 lands its `_SizedRefreshCache` relocation to `ngtools/utils`,
swap the local `_SizedLruCache` below for that shared class — all public
interfaces are unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import struct
import threading
from collections import OrderedDict
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np

LOG = logging.getLogger(__name__)


# =============================================================================
# Byte-sized LRU cache (local; swap for ngtools.utils._SizedRefreshCache later)
# =============================================================================

class _SizedLruCache:
    """
    LRU cache with a byte-based size cap.

    - `get(key)` returns the value and marks it as most-recently-used, or
      returns the sentinel `MISS` if absent.
    - `put(key, value, size_bytes)` inserts/updates. Evicts LRU entries
      until total size <= max_size_bytes.
    - `invalidate(key)` removes one entry if present.
    - `invalidate_matching(pred)` removes all entries where `pred(key)` is True.
    - `clear()` removes everything.

    Thread-safe via a single lock — not optimized for contention; fine for a
    single-process Tornado fileserver.

    `size_bytes` is the caller-declared size. For simple numpy buffers or
    bytes, use `len(body)`. For complex structures (dicts, the by_id index),
    estimate with `_estimate_size()` or pick a conservative fixed cost.
    """

    MISS = object()

    def __init__(self, max_size_bytes: int, name: str = "cache"):
        self._data: OrderedDict = OrderedDict()  # key -> (value, size_bytes)
        self._total: int = 0
        self._max: int = int(max_size_bytes)
        self._lock = threading.Lock()
        self._name = name

    def __len__(self):
        with self._lock:
            return len(self._data)

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return self.MISS
            value, size = self._data.pop(key)
            self._data[key] = (value, size)  # move to end (most recent)
            return value

    def put(self, key, value, size_bytes: int):
        with self._lock:
            if key in self._data:
                _, old_size = self._data.pop(key)
                self._total -= old_size
            self._data[key] = (value, int(size_bytes))
            self._total += int(size_bytes)
            # Evict LRU until within budget.
            while self._total > self._max and self._data:
                evicted_key, (_, evicted_size) = self._data.popitem(last=False)
                self._total -= evicted_size
                LOG.debug("[%s] evicted LRU key %r (%d bytes)",
                          self._name, evicted_key, evicted_size)

    def invalidate(self, key) -> bool:
        """Remove one entry. Returns True if it was present."""
        with self._lock:
            if key not in self._data:
                return False
            _, size = self._data.pop(key)
            self._total -= size
            return True

    def invalidate_matching(self, pred: Callable[[Any], bool]) -> int:
        """Remove all entries whose key matches pred. Returns count removed."""
        with self._lock:
            to_remove = [k for k in self._data if pred(k)]
            for k in to_remove:
                _, size = self._data.pop(k)
                self._total -= size
            return len(to_remove)

    def clear(self):
        with self._lock:
            self._data.clear()
            self._total = 0

    def resize(self, max_size_bytes: int):
        """
        Change the byte cap in place. Evicts LRU entries if the new cap is
        smaller than current usage. Does NOT rebind the cache — callers
        holding a reference continue to see the same object.
        """
        with self._lock:
            self._max = int(max_size_bytes)
            while self._total > self._max and self._data:
                evicted_key, (_, evicted_size) = self._data.popitem(last=False)
                self._total -= evicted_size
                LOG.debug("[%s] evicted LRU key %r (%d bytes) on resize",
                          self._name, evicted_key, evicted_size)


# =============================================================================
# Cache instances and reference-count tracker
# =============================================================================
# Cache keys are all tuples starting with (store_path, array_name) so the
# refcount-based eviction in unregister_array_usage can invalidate cleanly.

# Info JSON objects: small; a few MiB is more than enough for hundreds of arrays.
_info_cache = _SizedLruCache(max_size_bytes=4 * 1024 * 1024, name="info")

# Spatial chunks (bytes): the main memory budget. Default 2 GiB; override
# via `configure_caches()` in production.
_geometry_cache = _SizedLruCache(max_size_bytes=2 * 1024 * 1024 * 1024,
                                  name="geometry")

# by_id index dicts. Up to 256 MiB; large-id-space arrays may exceed this and
# rebuild on demand.
_by_id_index_cache = _SizedLruCache(max_size_bytes=256 * 1024 * 1024,
                                     name="by_id_index")

# Per-array reference count. Keys are (store_path, array_name) tuples.
# Incremented by Scene.load (via register_array_usage), decremented by
# Scene.unload (via unregister_array_usage). When a key drops to zero,
# all caches for that key are invalidated.
_array_refcount: dict = {}
_refcount_lock = threading.Lock()

# Session 9: mtime-on-read invalidation
# Per-(store_path, array_name) cached mtime for the array. On each cache hit,
# we re-stat the array and compare. If newer, invalidate all caches for the
# array before serving from the (now-stale) cache.
_array_mtime_cache: dict = {}
_mtime_lock = threading.Lock()


# Mtime check defaults. Set False for remote-store-heavy deployments where
# stat calls over the network are expensive. Override via
# `configure_caches(enable_mtime_check=False)`.
_enable_mtime_check: bool = True


def configure_caches(
    *,
    info_max_bytes: Optional[int] = None,
    geometry_max_bytes: Optional[int] = None,
    by_id_index_max_bytes: Optional[int] = None,
    enable_mtime_check: Optional[bool] = None,
) -> None:
    """
    Resize the global caches at runtime in-place. None leaves a cache
    unchanged. Cache object identity is preserved, so external code holding
    a reference continues to see the same object.

    `enable_mtime_check`: when True (default), each cache hit re-stats the
    backing array and invalidates if the mtime is newer than when the entry
    was cached. Disable for remote stores where stat calls are expensive,
    and rely on `reload_zv_array()` instead.

    Called by `LocalNeuroglancer.__init__` (or scene setup) if operators
    want to override the defaults — e.g., smaller geometry cap on
    memory-constrained machines.
    """
    global _enable_mtime_check
    if info_max_bytes is not None:
        _info_cache.resize(info_max_bytes)
    if geometry_max_bytes is not None:
        _geometry_cache.resize(geometry_max_bytes)
    if by_id_index_max_bytes is not None:
        _by_id_index_cache.resize(by_id_index_max_bytes)
    if enable_mtime_check is not None:
        _enable_mtime_check = bool(enable_mtime_check)


def register_array_usage(store_path: str, array_name: str) -> int:
    """
    Scene.load calls this when registering a layer that uses this array.
    Returns the new refcount (useful for debugging).

    Multiple layers sharing an array (e.g., mesh + segment_properties
    subresource, or the same store loaded twice under different names)
    each take one reference.
    """
    key = (store_path, array_name)
    with _refcount_lock:
        _array_refcount[key] = _array_refcount.get(key, 0) + 1
        return _array_refcount[key]


def unregister_array_usage(store_path: str, array_name: str) -> int:
    """
    Scene.unload calls this when a layer using this array goes away.
    Returns the new refcount. When the count reaches zero, caches for this
    (store_path, array_name) are evicted.
    """
    key = (store_path, array_name)
    with _refcount_lock:
        if key not in _array_refcount:
            LOG.warning(
                "unregister_array_usage: unknown key %r (already at 0?)", key
            )
            return 0
        _array_refcount[key] -= 1
        count = _array_refcount[key]
        if count <= 0:
            del _array_refcount[key]

    if count <= 0:
        _evict_caches_for(store_path, array_name)
    return max(0, count)


def get_array_refcount(store_path: str, array_name: str) -> int:
    """For debugging/inspection."""
    with _refcount_lock:
        return _array_refcount.get((store_path, array_name), 0)


def _evict_caches_for(store_path: str, array_name: str) -> None:
    """Internal: invalidate all cache entries for one array."""
    prefix = (store_path, array_name)

    def matches(key):
        return isinstance(key, tuple) and len(key) >= 2 and key[:2] == prefix

    n1 = _info_cache.invalidate_matching(matches)
    n2 = _geometry_cache.invalidate_matching(matches)
    n3 = _by_id_index_cache.invalidate_matching(matches)
    LOG.debug(
        "evicted caches for %r: info=%d geometry=%d by_id=%d",
        prefix, n1, n2, n3,
    )


def invalidate_array_caches(store_path: str, array_name: str) -> None:
    """
    Force-evict all caches for one array, regardless of refcount.

    Session 9's `reload` command calls this. After a force-evict, subsequent
    requests rebuild from the live data.
    """
    _evict_caches_for(store_path, array_name)
    # Also clear the tracked mtime so the next read re-stats the array.
    with _mtime_lock:
        _array_mtime_cache.pop((store_path, array_name), None)


# =============================================================================
# Session 9: mtime-on-read invalidation
# =============================================================================

def _read_array_mtime(array) -> Optional[float]:
    """
    Read the current mtime from the array.

    Prefers `array.mtime()` if present. Returns None if the array exposes
    no mtime method — callers should treat None as "always fresh" (i.e.
    skip the staleness check).
    """
    mtime_fn = getattr(array, "mtime", None)
    if not callable(mtime_fn):
        return None
    try:
        m = mtime_fn()
    except Exception as e:
        LOG.warning("array.mtime() raised %r; skipping staleness check", e)
        return None
    if m is None:
        return None
    return float(m)


def _check_and_evict_if_stale(
    array,
    store_path: Optional[str],
    array_name: Optional[str],
) -> None:
    """
    Check whether the array has been modified since its cached entries
    were populated. If so, evict all caches for this array.

    No-op when:
      - mtime checking is disabled (_enable_mtime_check is False)
      - store_path or array_name is None (uncached path)
      - array exposes no mtime() method

    Called at the start of every cached code path, BEFORE cache lookup,
    so a stale hit is impossible.
    """
    if not _enable_mtime_check:
        return
    if store_path is None or array_name is None:
        return

    current = _read_array_mtime(array)
    if current is None:
        # Array doesn't support mtime; rely on reload_zv_array() instead.
        return

    key = (store_path, array_name)
    with _mtime_lock:
        previous = _array_mtime_cache.get(key)
        if previous is None:
            # First time seeing this array; record the mtime and move on.
            _array_mtime_cache[key] = current
            return
        if current > previous:
            # Array has been rewritten since last seen; evict caches.
            _array_mtime_cache[key] = current
            stale_detected = True
        else:
            stale_detected = False

    if stale_detected:
        LOG.info(
            "detected stale cache for %r (mtime %s -> %s); evicting",
            key, previous, current,
        )
        _evict_caches_for(store_path, array_name)


# =============================================================================
# Session 9: public reload hooks for the nglocal shell
# =============================================================================

def reload_zv_array(store_path: str, array_name: str) -> None:
    """
    Force-reload one array: evicts all caches and resets the tracked mtime.

    Called by the ngtools `reload` shell command when the user specifies a
    particular layer. The layer name is translated by Scene.py to
    (store_path, array_name) via the _cache_identity captured at load time.

    Equivalent to `invalidate_array_caches` but exposed as a stable public
    name for the shell to call.
    """
    invalidate_array_caches(store_path, array_name)


def reload_all_zv_caches() -> None:
    """
    Force-reload everything: clear all zv caches and mtime tracking.

    Called by the ngtools `reload` shell command when no layer is specified.
    Does NOT touch refcounts — layers remain registered, they'll just
    rebuild on next request.
    """
    _info_cache.clear()
    _geometry_cache.clear()
    _by_id_index_cache.clear()
    with _mtime_lock:
        _array_mtime_cache.clear()
    LOG.info("reload_all_zv_caches: all zv caches and mtime tracking cleared")


def get_tracked_mtime(store_path: str, array_name: str) -> Optional[float]:
    """For debugging/inspection. Returns None if mtime not yet tracked."""
    with _mtime_lock:
        return _array_mtime_cache.get((store_path, array_name))


# Compatibility wrapper: Session 7 exposed `invalidate_by_id_index(array)`.
# Keep the signature for external callers but delegate to the new cache-layer
# invalidation when possible. With refcounted eviction, explicit invalidation
# is mostly a reload concern — see `invalidate_array_caches`.
def invalidate_by_id_index(array) -> None:
    """
    Legacy Session 7 hook. Preferred replacement is
    `invalidate_array_caches(store_path, array_name)` (Session 9's reload
    command). This version scans the by_id_index_cache for any entry whose
    associated array object matches `array` and drops those entries.

    Since the new cache is keyed by (store_path, array_name) rather than
    Python object identity, the best we can do here is scan-and-match by
    a weakref-friendly sentinel. In practice: Session 7 callers that used
    this function as a force-flush mechanism should migrate to
    `invalidate_array_caches`, passing explicit identifiers.

    For Session 8 compatibility: a full clear of by_id_index_cache is the
    safest fallback when given only the array object.
    """
    # Conservative: no way to recover (store_path, array_name) from just the
    # array object. Clear the whole by_id cache. Callers who want targeted
    # invalidation should use invalidate_array_caches instead.
    _by_id_index_cache.clear()


# Session 7 kept this public — preserve it for tests that poke at internals.
# Under Session 8 it's still meaningful but now points at the new cache.
def _by_id_index_cache_clear():
    """Test-only helper to clear the by_id index cache."""
    _by_id_index_cache.clear()


# =============================================================================
# Annotation type registry
# =============================================================================

_ANNOTATION_GEOMETRY = {
    "LINE":  {"n_floats": 6, "shape": (2, 3)},
    "POINT": {"n_floats": 3, "shape": (3,)},
}

_ZV_DATATYPE_TO_ANNOTATION = {
    "streamline": "LINE",
    "point":      "POINT",
}

_ZV_DATATYPE_TO_ROUTE_KIND = {
    "streamline": "streamlines",
    "point":      "points",
    "mesh":       "mesh",
}

_ZV_DATATYPE_TO_LAYER_TYPE = {
    "streamline": "annotation",
    "point":      "annotation",
    "mesh":       "mesh",
}


def _resolve_annotation_type(array, annotation_type):
    inferred = _ZV_DATATYPE_TO_ANNOTATION.get(array.datatype)
    if annotation_type is None:
        if inferred is None:
            raise ValueError(
                f"cannot infer annotation_type for datatype "
                f"{array.datatype!r}; pass annotation_type explicitly"
            )
        return inferred
    if annotation_type not in _ANNOTATION_GEOMETRY:
        raise ValueError(
            f"unknown annotation_type {annotation_type!r}; "
            f"known: {sorted(_ANNOTATION_GEOMETRY)}"
        )
    if inferred is not None and inferred != annotation_type:
        raise ValueError(
            f"array datatype {array.datatype!r} produces annotation_type "
            f"{inferred!r}, but {annotation_type!r} requested"
        )
    return annotation_type


# =============================================================================
# Property type registry
# =============================================================================

_PROPERTY_TYPES = {
    "uint32":  {"width": 4, "alignment": 4, "dtype": "<u4"},
    "int32":   {"width": 4, "alignment": 4, "dtype": "<i4"},
    "float32": {"width": 4, "alignment": 4, "dtype": "<f4"},
    "uint16":  {"width": 2, "alignment": 2, "dtype": "<u2"},
    "int16":   {"width": 2, "alignment": 2, "dtype": "<i2"},
    "uint8":   {"width": 1, "alignment": 1, "dtype": "<u1"},
    "int8":    {"width": 1, "alignment": 1, "dtype": "<i1"},
    "rgb":     {"width": 3, "alignment": 1, "dtype": "<u1"},
    "rgba":    {"width": 4, "alignment": 1, "dtype": "<u1"},
}

_PROPERTY_ID_RE = re.compile(r'^[a-z][a-zA-Z0-9_]*$')


def _get_properties_schema(array) -> List[dict]:
    raw = getattr(array, "segment_properties", None) or []
    seen_ids = set()
    out = []
    for i, spec in enumerate(raw):
        if not isinstance(spec, dict):
            raise ValueError(
                f"segment_properties[{i}] must be a dict, got {type(spec).__name__}"
            )
        if "id" not in spec or "type" not in spec:
            raise ValueError(f"segment_properties[{i}] missing id/type: {spec!r}")
        pid = spec["id"]
        ptype = spec["type"]
        if not _PROPERTY_ID_RE.match(pid):
            raise ValueError(f"property id {pid!r} must match [a-z][a-zA-Z0-9_]*")
        if pid in seen_ids:
            raise ValueError(f"duplicate property id {pid!r}")
        if ptype not in _PROPERTY_TYPES:
            raise ValueError(
                f"property {pid!r}: unknown type {ptype!r}; "
                f"known: {sorted(_PROPERTY_TYPES)}"
            )
        if ("enum_values" in spec) != ("enum_labels" in spec):
            raise ValueError(
                f"property {pid!r}: enum_values and enum_labels must be specified together"
            )
        if "enum_values" in spec:
            if ptype in ("rgb", "rgba"):
                raise ValueError(f"property {pid!r}: enum_values not allowed for rgb/rgba")
            if len(spec["enum_values"]) != len(spec["enum_labels"]):
                raise ValueError(f"property {pid!r}: enum_values/enum_labels length mismatch")
        seen_ids.add(pid)
        out.append(dict(spec))
    return out


def _canonical_property_order(properties_schema):
    buckets = {4: [], 2: [], 1: []}
    for i, prop in enumerate(properties_schema):
        align = _PROPERTY_TYPES[prop["type"]]["alignment"]
        buckets[align].append(i)
    return buckets[4] + buckets[2] + buckets[1]


def _per_annotation_property_size(properties_schema):
    total = sum(_PROPERTY_TYPES[p["type"]]["width"] for p in properties_schema)
    pad = (4 - total % 4) % 4
    return total + pad


def _encode_property_value(value, prop_type):
    spec = _PROPERTY_TYPES[prop_type]
    if prop_type == "rgb":
        arr = np.asarray(value, dtype=np.uint8)
        if arr.shape != (3,):
            raise ValueError(f"rgb expects shape (3,), got {arr.shape}")
        return arr.tobytes()
    if prop_type == "rgba":
        arr = np.asarray(value, dtype=np.uint8)
        if arr.shape != (4,):
            raise ValueError(f"rgba expects shape (4,), got {arr.shape}")
        return arr.tobytes()
    arr = np.array(value, dtype=spec["dtype"])
    if arr.shape != ():
        raise ValueError(f"{prop_type} expects a scalar value, got shape {arr.shape}")
    return arr.tobytes()


def _encode_annotation_properties(segment, properties_schema, order):
    out = bytearray()
    for idx in order:
        prop = properties_schema[idx]
        if prop["id"] not in segment.properties:
            raise ValueError(f"segment {segment.id}: missing property {prop['id']!r}")
        value = segment.properties[prop["id"]]
        out.extend(_encode_property_value(value, prop["type"]))
    pad = (4 - len(out) % 4) % 4
    out.extend(b"\x00" * pad)
    return bytes(out)


# =============================================================================
# Real zarr_vectors adapter
# =============================================================================
# The handler in this file talks to opaque "store" and "array" objects through
# a documented contract (see _make_layer_spec, _build_info_raw, and the
# read_bin / bin_size / grid_shape call sites). The real `zarr_vectors`
# package exposes a different surface — one store per geometry type, with
# reads via top-level functions like `read_chunk_vertices` and
# `read_chunk_links`. This adapter wraps a `zarr_vectors.lazy.store.ZVRStore`
# in the per-array contract the handler expects.
#
# Mapping from geometry_type to virtual arrays:
#   graph / skeleton  ->  ("nodes", point) + ("edges", streamline)
#   point_cloud       ->  ("points", point)
#   line              ->  ("lines", streamline)
#   polyline          ->  ("polylines", streamline; each polyline exploded
#                          into N-1 two-point segments)
#   mesh              ->  ("mesh", mesh)
#
# v1 limitations (handler has correct fallbacks for all of these):
#  - Single resolution level only.
#  - Spatial bins precomputed eagerly on first `read_bin` call. Memory is
#    O(N + E); fine for the 10K-vertex notebook smoke test.
#  - bounds[0] (the lower corner) is subtracted off positions so the
#    handler's [0, voxel_extent] frame holds.
#  - mtime/locate/limit_per_level not implemented; handler scans/builds
#    on demand.

class _Segment:
    __slots__ = ("id", "prominence", "geometry", "properties", "relationships")

    def __init__(self, id, geometry, prominence=0, properties=None, relationships=None):
        self.id = int(id)
        self.geometry = geometry
        self.prominence = int(prominence)
        self.properties = properties if properties is not None else {}
        self.relationships = relationships if relationships is not None else {}


class _MeshObj:
    __slots__ = ("vertices", "indices")

    def __init__(self, vertices, indices):
        self.vertices = vertices
        self.indices = indices


def _voxel_extent_from_bounds(bounds):
    import math as _math
    bmax = bounds[1]
    return tuple(int(_math.ceil(float(v))) for v in bmax)


_POSITION_PROPERTIES = [
    {"id": "x", "type": "float32"},
    {"id": "y", "type": "float32"},
    {"id": "z", "type": "float32"},
]


# === SLAB/CONE-VIEW SANDBOX BEGIN ============================================
# Deterministic hash-based decimation. `_level_keep(seg_id, K)` is True for
# ~1/2^K of all seg_ids. Used by the 8 LOD-pinned virtual arrays
# (nodes_lod0..3, edges_lod0..3) so Neuroglancer's notebook-side cursor-LOD
# shader can render a different decimation tier in each cursor-distance ring.
def _level_keep(seg_id: int, level: int) -> bool:
    if level <= 0:
        return True
    stride = 1 << int(level)
    return ((int(seg_id) * 2654435761) & 0xFFFFFFFF) % stride == 0


# Default annotation shader applied to every zarrvectors:// layer at load
# time (see scene.py _register_zv_layer). Reads the per-annotation x/y/z
# properties published above (_POSITION_PROPERTIES) and uses
# uModelViewProjection to fade items by signed distance from the slice
# plane in cross-section panels. The 3D panel is unaffected.
#
# Sliders exposed in the layer's Render tab:
#   clip_sign      ±1  - flip if your build's depth sign convention
#                        is reversed (camera-side vs far-side swap).
#   fade_in_front  NDC - fade ramp width in front of the slice (0.05..10);
#                        with cross_section_depth ~ 2*volume_radius, NDC
#                        1.0 corresponds to ~ volume_radius voxels.
#   fade_behind    NDC - fade ramp width behind the slice. Set to ~10
#                        to effectively disable fade on that side.
_SHADER_HEAD = """
#uicontrol float clip_sign     slider(min=-1.0, max=1.0,  default=1.0, step=2.0)
#uicontrol float fade_in_front slider(min=0.05, max=10.0, default=0.6, step=0.05)
#uicontrol float fade_behind   slider(min=0.05, max=10.0, default=10.0, step=0.05)
"""

_SHADER_DEPTH_FADE = """
    if (!PROJECTION_VIEW) {
      vec4 clip = uModelViewProjection * vec4(p, 1.0);
      float depth = clip_sign * (clip.z / max(clip.w, 1e-6));
      if (depth >= 0.0) {
        fade = 1.0 - smoothstep(0.0, fade_in_front, depth);
      } else {
        fade = 1.0 - smoothstep(0.0, fade_behind, -depth);
      }
      if (fade <= 0.0) discard;
    }
"""

_DEFAULT_POINT_SHADER = _SHADER_HEAD + """
#uicontrol float base_size slider(min=1.0, max=20.0, default=8.0)

void main() {
  vec3 p = vec3(prop_x(), prop_y(), prop_z());
  float fade = 1.0;
""" + _SHADER_DEPTH_FADE + """
  setColor(vec4(defaultColor(), fade));
  setPointMarkerSize(base_size * fade);
}
"""

_DEFAULT_LINE_SHADER = _SHADER_HEAD + """
#uicontrol float base_width slider(min=0.3, max=5.0, default=1.5)

void main() {
  vec3 p = vec3(prop_x(), prop_y(), prop_z());
  float fade = 1.0;
""" + _SHADER_DEPTH_FADE + """
  setColor(vec4(defaultColor(), fade));
  setLineWidth(base_width * fade);
}
"""

_DEFAULT_SHADERS_BY_DATATYPE = {
    "point":      _DEFAULT_POINT_SHADER,
    "streamline": _DEFAULT_LINE_SHADER,
    # mesh layers go through SegmentationLayer's mesh-shader pipeline,
    # which is not the annotation shader DSL — no default published.
    "mesh":       None,
}
# === SLAB/CONE-VIEW SANDBOX END ==============================================


class _BaseArrayAdapter:
    """Common metadata for annotation adapters (point / streamline).

    Multi-level support: each level k uses bin shape = base_bin_shape * 2^k
    (binning-only LOD; same segments at every level, just coarser bin grid).
    Subclasses provide `_collect_segments()` returning a flat list of
    _Segment records and `_segment_position(seg)` returning the (D,) point
    used for binning (point coords for points, midpoint for streamlines).
    """

    LEVELS = 4

    # === SLAB/CONE-VIEW SANDBOX BEGIN ============================================
    # Per-annotation x/y/z properties published via the precomputed `info.properties`
    # schema. Notebook-side annotation shaders can read them (`prop_x()` / `prop_y()`
    # / `prop_z()`) to compute signed depth from the slice plane and clip the
    # camera-side half. Cost: 12 extra bytes per segment in the wire format.
    segment_properties = _POSITION_PROPERTIES
    # === SLAB/CONE-VIEW SANDBOX END ==============================================

    def __init__(self, zvr, level=0, pinned_level=None):
        # === SLAB/CONE-VIEW SANDBOX BEGIN ========================================
        # `pinned_level=K` collapses this adapter to a single LOD: it reports
        # `levels = 1`, uses bin_shape = base * 2^K, and decimates segments by
        # `_level_keep(seg.id, K)` (~1/2^K keep ratio). The notebook then loads
        # 4 such pinned-level layers per geometry kind, masks each by a cursor-
        # distance ring in the shader, and gets cursor-centred chunk-level LOD.
        # === SLAB/CONE-VIEW SANDBOX END ==========================================
        self._zvr = zvr
        self._level_idx = level
        self._pinned_level = pinned_level
        self._root_meta = zvr._meta
        self._ndim = zvr._meta.sid_ndim
        self._bin_shape = tuple(float(b) for b in zvr._meta.effective_bin_shape)
        self._chunk_shape = tuple(float(c) for c in zvr._meta.chunk_shape)
        self._bins_per_chunk = zvr._meta.bins_per_chunk
        self._bound_min = np.asarray(zvr._meta.bounds[0], dtype=np.float64)
        self._voxel_extent = _voxel_extent_from_bounds(zvr._meta.bounds)
        # Per-level state, populated lazily.
        self._segments: list | None = None
        self._bins_per_level: dict[int, dict] = {}
        self._limits: list[int] | None = None

    @property
    def voxel_extent(self):
        return self._voxel_extent

    @property
    def affine(self):
        return np.eye(4, dtype=np.float64)

    @property
    def levels(self):
        return 1 if self._pinned_level is not None else self.LEVELS

    def _level_bin_shape(self, level):
        # When pinned, every level in this adapter (only level 0 exists) maps to
        # the pinned source level's bin shape.
        src = self._pinned_level if self._pinned_level is not None else int(level)
        scale = 2 ** int(src)
        return tuple(b * scale for b in self._bin_shape)

    def bin_size(self, level):
        return tuple(int(round(b)) for b in self._level_bin_shape(level))

    def grid_shape(self, level):
        import math as _math
        bs = self._level_bin_shape(level)
        return tuple(
            max(1, int(_math.ceil(self._voxel_extent[d] / bs[d])))
            for d in range(self._ndim)
        )

    # --- subclass hooks ---
    def _collect_segments(self):
        raise NotImplementedError

    def _segment_position(self, seg):
        raise NotImplementedError

    # --- shared bin/level machinery ---
    def _ensure_segments(self):
        if self._segments is None:
            self._segments = list(self._collect_segments())

    def _ensure_bins(self, level):
        L = int(level)
        if L in self._bins_per_level:
            return
        self._ensure_segments()
        bs = self._level_bin_shape(L)
        # === SLAB/CONE-VIEW SANDBOX BEGIN ========================================
        # When `_pinned_level` is set, this adapter is one of the eight LOD-pinned
        # virtual arrays exposed by _ZvAdapterStore for graph stores. Decimate the
        # segment list by the deterministic hash _level_keep so a coarser pinned
        # level holds ~1/2^K of the segments.
        keep_level = self._pinned_level if self._pinned_level is not None else 0
        # === SLAB/CONE-VIEW SANDBOX END ==========================================
        bins: dict[tuple, list] = {}
        for s in self._segments:
            if not _level_keep(int(s.id), keep_level):
                continue
            pos = self._segment_position(s)
            key = tuple(int(pos[d] // bs[d]) for d in range(self._ndim))
            # The handler filters via `s.prominence == zv_level` (zarrvectors.py
            # serve_spatial_chunk). For binning-only LOD where the same segment
            # appears at every level, stamp prominence == L on the per-level
            # copy so the filter always passes.
            bin_seg = _Segment(
                id=s.id, geometry=s.geometry, prominence=L,
                properties=s.properties, relationships=s.relationships,
            )
            bins.setdefault(key, []).append(bin_seg)
        self._bins_per_level[L] = bins

    def read_bin(self, level, ix, iy, iz):
        L = int(level)
        if not (0 <= L < self.levels):
            return iter(())
        self._ensure_bins(L)
        return iter(self._bins_per_level[L].get((int(ix), int(iy), int(iz)), ()))

    def limit_per_level(self):
        if self._limits is None:
            limits: list[int] = []
            for L in range(self.levels):
                self._ensure_bins(L)
                bins = self._bins_per_level[L]
                m = max((len(v) for v in bins.values()), default=1)
                limits.append(int(m))
            self._limits = limits
        return list(self._limits)


class _PointArrayAdapter(_BaseArrayAdapter):
    datatype = "point"

    def __init__(self, zvr, level=0, name="points", pinned_level=None):
        super().__init__(zvr, level, pinned_level=pinned_level)
        self._name = name

    def _segment_position(self, seg):
        return seg.geometry  # (D,) for POINT

    def _collect_segments(self):
        from zarr_vectors.core.arrays import read_chunk_vertices
        zvl = self._zvr[self._level_idx]
        bound_min = self._bound_min.astype(np.float32)
        out = []
        next_id = 0
        for ck in zvl.chunk_keys:
            try:
                groups = read_chunk_vertices(
                    zvl._group, ck, dtype=np.float32, ndim=self._ndim,
                )
            except Exception as e:
                LOG.warning("point adapter: skipping chunk %r: %s", ck, e)
                continue
            for vg in groups:
                for v in vg:
                    pos = np.asarray(v, dtype=np.float32) - bound_min
                    next_id += 1
                    # SLAB/CONE-VIEW SANDBOX: publish position as x/y/z
                    # float32 properties for the behind-slice clipping shader.
                    out.append(_Segment(
                        id=next_id, geometry=pos.copy(), prominence=0,
                        properties={
                            "x": float(pos[0]),
                            "y": float(pos[1]),
                            "z": float(pos[2]),
                        },
                    ))
        return out


class _StreamlineArrayAdapter(_BaseArrayAdapter):
    datatype = "streamline"

    def __init__(self, zvr, level=0, name="lines", source="links", pinned_level=None):
        """source: "links" (graph/skeleton edges + cross-chunk),
                   "vg_pairs" (each vertex group is a 2-vertex line),
                   "polylines" (read object manifests and explode N-vertex chains)."""
        super().__init__(zvr, level, pinned_level=pinned_level)
        self._name = name
        self._source = source

    def _segment_position(self, seg):
        # geometry is (2, D); bin by midpoint
        return (seg.geometry[0] + seg.geometry[1]) * 0.5

    def _collect_segments(self):
        if self._source == "links":
            return self._collect_from_links()
        if self._source == "vg_pairs":
            return self._collect_from_vg_pairs()
        if self._source == "polylines":
            return self._collect_from_polylines()
        raise ValueError(f"unknown streamline source {self._source!r}")

    def _collect_from_links(self):
        from zarr_vectors.core.arrays import (
            read_chunk_vertices, read_chunk_links, read_cross_chunk_links,
        )
        out = []
        zvl = self._zvr[self._level_idx]
        bound_min = self._bound_min.astype(np.float32)
        next_id = 0
        chunk_vertices = {}
        for ck in zvl.chunk_keys:
            try:
                groups = read_chunk_vertices(
                    zvl._group, ck, dtype=np.float32, ndim=self._ndim,
                )
            except Exception:
                continue
            chunk_vertices[ck] = (
                np.concatenate(groups, axis=0)
                if groups else np.zeros((0, self._ndim), dtype=np.float32)
            )
        for ck, verts in chunk_vertices.items():
            try:
                link_groups = read_chunk_links(zvl._group, ck, link_width=2)
            except Exception:
                link_groups = []
            for lg in link_groups:
                for e in lg:
                    a, b = int(e[0]), int(e[1])
                    if not (0 <= a < len(verts)) or not (0 <= b < len(verts)):
                        continue
                    pa = verts[a].astype(np.float32) - bound_min
                    pb = verts[b].astype(np.float32) - bound_min
                    next_id += 1
                    # SLAB/CONE-VIEW SANDBOX: properties = midpoint coords
                    mid = (pa + pb) * 0.5
                    out.append(_Segment(
                        id=next_id, geometry=np.stack([pa, pb], axis=0),
                        prominence=0,
                        properties={
                            "x": float(mid[0]),
                            "y": float(mid[1]),
                            "z": float(mid[2]),
                        },
                    ))
        try:
            ccl = read_cross_chunk_links(zvl._group)
        except Exception:
            ccl = []
        for (ca, vi_a), (cb, vi_b) in ccl:
            ca = tuple(int(x) for x in ca)
            cb = tuple(int(x) for x in cb)
            va = chunk_vertices.get(ca)
            vb = chunk_vertices.get(cb)
            if va is None or vb is None:
                continue
            if not (0 <= vi_a < len(va)) or not (0 <= vi_b < len(vb)):
                continue
            pa = va[vi_a].astype(np.float32) - bound_min
            pb = vb[vi_b].astype(np.float32) - bound_min
            next_id += 1
            # SLAB/CONE-VIEW SANDBOX: properties = midpoint coords
            mid = (pa + pb) * 0.5
            out.append(_Segment(
                id=next_id, geometry=np.stack([pa, pb], axis=0),
                prominence=0,
                properties={
                    "x": float(mid[0]),
                    "y": float(mid[1]),
                    "z": float(mid[2]),
                },
            ))
        return out

    def _collect_from_vg_pairs(self):
        from zarr_vectors.core.arrays import read_chunk_vertices
        out = []
        zvl = self._zvr[self._level_idx]
        bound_min = self._bound_min.astype(np.float32)
        next_id = 0
        for ck in zvl.chunk_keys:
            try:
                groups = read_chunk_vertices(
                    zvl._group, ck, dtype=np.float32, ndim=self._ndim,
                )
            except Exception:
                continue
            for vg in groups:
                if len(vg) < 2:
                    continue
                pa = vg[0].astype(np.float32) - bound_min
                pb = vg[1].astype(np.float32) - bound_min
                next_id += 1
                # SLAB/CONE-VIEW SANDBOX: properties = midpoint coords
                mid = (pa + pb) * 0.5
                out.append(_Segment(
                    id=next_id, geometry=np.stack([pa, pb], axis=0),
                    prominence=0,
                    properties={
                        "x": float(mid[0]),
                        "y": float(mid[1]),
                        "z": float(mid[2]),
                    },
                ))
        return out

    def _collect_from_polylines(self):
        from zarr_vectors.core.arrays import (
            read_all_object_manifests, read_object_vertices,
        )
        out = []
        zvl = self._zvr[self._level_idx]
        bound_min = self._bound_min.astype(np.float32)
        try:
            manifests = read_all_object_manifests(zvl._group)
        except Exception as e:
            LOG.warning("polyline adapter: cannot read object manifests: %s", e)
            return out
        next_id = 0
        for obj_id in range(len(manifests)):
            try:
                groups = read_object_vertices(
                    zvl._group, obj_id, dtype=np.float32, ndim=self._ndim,
                )
            except Exception:
                continue
            if not groups:
                continue
            verts = np.concatenate(groups, axis=0)
            for i in range(len(verts) - 1):
                pa = verts[i].astype(np.float32) - bound_min
                pb = verts[i + 1].astype(np.float32) - bound_min
                next_id += 1
                # SLAB/CONE-VIEW SANDBOX: properties = midpoint coords
                mid = (pa + pb) * 0.5
                out.append(_Segment(
                    id=next_id, geometry=np.stack([pa, pb], axis=0),
                    prominence=0,
                    properties={
                        "x": float(mid[0]),
                        "y": float(mid[1]),
                        "z": float(mid[2]),
                    },
                ))
        return out


class _MeshArrayAdapter:
    """V1 mesh adapter: best-effort. The notebook smoke test does not
    exercise mesh stores, so this path is minimally validated. Refine
    when a mesh-store smoke test exists."""

    datatype = "mesh"

    def __init__(self, zvr, level=0, name="mesh"):
        self._zvr = zvr
        self._level_idx = level
        self._name = name
        rm = zvr._meta
        self._ndim = rm.sid_ndim
        self._voxel_extent = _voxel_extent_from_bounds(rm.bounds)
        self._segment_ids = None
        self._mesh_cache = {}

    @property
    def voxel_extent(self):
        return self._voxel_extent

    @property
    def affine(self):
        return np.eye(4, dtype=np.float64)

    @property
    def levels(self):
        return 1

    @property
    def segment_properties(self):
        return []

    def _load_segment_ids(self):
        if self._segment_ids is not None:
            return
        from zarr_vectors.core.arrays import read_all_object_manifests
        zvl = self._zvr[self._level_idx]
        try:
            manifests = read_all_object_manifests(zvl._group)
            self._segment_ids = list(range(len(manifests)))
        except Exception as e:
            LOG.warning("mesh adapter: could not enumerate segment IDs: %s", e)
            self._segment_ids = []

    def iter_segment_ids(self):
        self._load_segment_ids()
        return iter(self._segment_ids)

    def get_mesh(self, segment_id):
        sid = int(segment_id)
        if sid in self._mesh_cache:
            return self._mesh_cache[sid]
        from zarr_vectors.core.arrays import read_object_vertices
        zvl = self._zvr[self._level_idx]
        groups = read_object_vertices(
            zvl._group, sid, dtype=np.float32, ndim=self._ndim,
        )
        if not groups:
            raise KeyError(segment_id)
        vertices = np.concatenate(groups, axis=0).astype(np.float32, copy=False)
        # Triangle indices not yet wired; return an empty mesh so the
        # handler can at least register the layer. Real mesh support needs
        # the face-array convention nailed down against a real store.
        indices = np.zeros((0, 3), dtype=np.uint32)
        mesh = _MeshObj(vertices=vertices, indices=indices)
        self._mesh_cache[sid] = mesh
        return mesh


class _ZvAdapterStore:
    """Wraps a `zarr_vectors.lazy.store.ZVRStore` and exposes the
    handler's expected `store.items()` / `store[name]` contract."""

    def __init__(self, zvr):
        self._zvr = zvr
        self._geometry_types = list(zvr._meta.geometry_types)
        self._arrays = self._build_arrays()

    def _build_arrays(self):
        gtype = self._geometry_types[0] if self._geometry_types else None
        out = {}
        if gtype in ("graph", "skeleton"):
            # Single `nodes` (point) + single `edges` (streamline) layer per
            # graph store. Each has the adapter's native 4-level binning LOD;
            # Neuroglancer auto-selects the level based on screen-space pixel
            # density. The earlier LOD-pinned variants were removed — the
            # notebook driving this renders one of each layer.
            out["nodes"] = _PointArrayAdapter(self._zvr, name="nodes")
            out["edges"] = _StreamlineArrayAdapter(
                self._zvr, name="edges", source="links",
            )
        elif gtype == "point_cloud":
            out["points"] = _PointArrayAdapter(self._zvr, name="points")
        elif gtype == "line":
            out["lines"] = _StreamlineArrayAdapter(
                self._zvr, name="lines", source="vg_pairs",
            )
        elif gtype in ("polyline", "streamline"):
            out["polylines"] = _StreamlineArrayAdapter(
                self._zvr, name="polylines", source="polylines",
            )
        elif gtype == "mesh":
            out["mesh"] = _MeshArrayAdapter(self._zvr, name="mesh")
        else:
            LOG.warning(
                "unsupported zarr_vectors geometry_type %r; "
                "no virtual arrays exposed", gtype,
            )
        return out

    def items(self):
        return self._arrays.items()

    def __getitem__(self, name):
        return self._arrays[name]

    def __iter__(self):
        return iter(self._arrays)

    def keys(self):
        return self._arrays.keys()


# =============================================================================
# Store resolution hook
# =============================================================================

_open_store_hook = None


def set_open_store_hook(fn):
    global _open_store_hook
    _open_store_hook = fn


def _default_open_store(protocol, store_path):
    """Open a zarr_vectors store and wrap it in adapter classes that
    expose the per-array interface this handler expects."""
    from zarr_vectors.lazy.store import open_zvr
    if protocol == "file":
        full = store_path if store_path.startswith("/") else "/" + store_path
    else:
        full = f"{protocol}://{store_path}"
    zvr = open_zvr(full)
    return _ZvAdapterStore(zvr)


def _open_store(protocol, store_path):
    if _open_store_hook is not None:
        return _open_store_hook(protocol, store_path)
    return _default_open_store(protocol, store_path)


# =============================================================================
# URL parsing
# =============================================================================

_ARRAY_SEPARATOR = "/@"


def parse_url(path):
    if "/" not in path:
        raise ValueError(f"missing protocol segment: {path!r}")
    protocol, rest = path.split("/", 1)
    if not protocol:
        raise ValueError(f"empty protocol: {path!r}")
    sep_idx = rest.find(_ARRAY_SEPARATOR)
    if sep_idx < 0:
        raise ValueError(
            f"missing '/@array_name' separator in {rest!r}; "
            f"expected '.../@<array_name>/<resource>'"
        )
    store_path = rest[:sep_idx]
    after = rest[sep_idx + len(_ARRAY_SEPARATOR):]
    if "/" not in after:
        raise ValueError(f"missing resource after array name: {after!r}")
    array_name, resource = after.split("/", 1)
    if not array_name:
        raise ValueError("empty array name")
    return protocol, store_path, array_name, resource


_ZV_SCHEME_PREFIX = "zarrvectors://"


def parse_zarrvectors_url(url):
    if not url.startswith(_ZV_SCHEME_PREFIX):
        raise ValueError(f"not a zarrvectors URL: {url!r}")
    body = url[len(_ZV_SCHEME_PREFIX):]
    if not body:
        raise ValueError(f"empty body in zv URL: {url!r}")
    array_filter = None
    at_idx = body.rfind("/@")
    if at_idx >= 0 and "/" not in body[at_idx + len("/@"):]:
        array_filter = body[at_idx + len("/@"):]
        if not array_filter:
            raise ValueError(f"empty array name after /@ in {url!r}")
        body = body[:at_idx]
    scheme_idx = body.find("://")
    if scheme_idx >= 0:
        protocol = body[:scheme_idx]
        path = body[scheme_idx + len("://"):]
        if not protocol:
            raise ValueError(f"empty transport scheme in {url!r}")
        if protocol == "file" and path.startswith("/"):
            path = path[1:]
    else:
        protocol = "file"
        path = body[1:] if body.startswith("/") else body
    if not path:
        raise ValueError(f"empty store path in {url!r}")
    if "/@" in path:
        raise ValueError(f"store path contains reserved '/@' separator: {path!r}")
    return protocol, path, array_filter


# =============================================================================
# Annotation v1 encoders
# =============================================================================

def encode_annotation_chunk(segments, annotation_type, properties_schema=None):
    if annotation_type not in _ANNOTATION_GEOMETRY:
        raise ValueError(f"unknown annotation_type {annotation_type!r}")
    if properties_schema is None:
        properties_schema = []
    for i, prop in enumerate(properties_schema):
        if prop.get("type") not in _PROPERTY_TYPES:
            raise ValueError(f"properties_schema[{i}]: unknown type {prop.get('type')!r}")

    geom_spec = _ANNOTATION_GEOMETRY[annotation_type]
    n_floats = geom_spec["n_floats"]
    expected_shape = geom_spec["shape"]

    segments = list(segments)
    count = len(segments)

    out = bytearray()
    out.extend(struct.pack("<Q", count))
    if count == 0:
        return bytes(out)

    geom_buf = np.empty((count, n_floats), dtype="<f4")
    for i, s in enumerate(segments):
        g = np.asarray(s.geometry, dtype=np.float32)
        if g.shape != expected_shape:
            raise ValueError(
                f"segment {s.id}: expected geometry shape {expected_shape} "
                f"for {annotation_type}, got {g.shape}"
            )
        geom_buf[i] = g.ravel()

    if properties_schema:
        prop_order = _canonical_property_order(properties_schema)
        for i, s in enumerate(segments):
            out.extend(geom_buf[i].tobytes(order="C"))
            out.extend(_encode_annotation_properties(s, properties_schema, prop_order))
    else:
        out.extend(geom_buf.tobytes(order="C"))

    ids = np.array([int(s.id) for s in segments], dtype="<u8")
    out.extend(ids.tobytes(order="C"))
    return bytes(out)


def encode_line_chunk(segments, properties_schema=None):
    return encode_annotation_chunk(segments, "LINE", properties_schema)


def encode_point_chunk(segments, properties_schema=None):
    return encode_annotation_chunk(segments, "POINT", properties_schema)


def encode_single_annotation(segment, annotation_type,
                             properties_schema=None, relationships=None):
    if annotation_type not in _ANNOTATION_GEOMETRY:
        raise ValueError(f"unknown annotation_type {annotation_type!r}")
    if properties_schema is None:
        properties_schema = []
    if relationships is None:
        relationships = []
    for i, prop in enumerate(properties_schema):
        if prop.get("type") not in _PROPERTY_TYPES:
            raise ValueError(f"properties_schema[{i}]: unknown type {prop.get('type')!r}")

    geom_spec = _ANNOTATION_GEOMETRY[annotation_type]
    expected_shape = geom_spec["shape"]

    g = np.asarray(segment.geometry, dtype=np.float32)
    if g.shape != expected_shape:
        raise ValueError(
            f"segment {segment.id}: expected geometry shape {expected_shape} "
            f"for {annotation_type}, got {g.shape}"
        )

    out = bytearray()
    out.extend(np.ascontiguousarray(g.ravel(), dtype="<f4").tobytes())

    if properties_schema:
        prop_order = _canonical_property_order(properties_schema)
        out.extend(_encode_annotation_properties(segment, properties_schema, prop_order))

    seg_rels = getattr(segment, "relationships", {}) or {}
    for rel in relationships:
        rel_id = rel["id"]
        object_ids = seg_rels.get(rel_id, [])
        out.extend(struct.pack("<I", len(object_ids)))
        if object_ids:
            out.extend(np.asarray(object_ids, dtype="<u8").tobytes(order="C"))
    return bytes(out)


# =============================================================================
# by_id index (Session 7, now using _SizedLruCache)
# =============================================================================

def _build_by_id_index(array) -> dict:
    L = int(array.levels)
    out = {}
    for j in range(L):
        gx, gy, gz = array.grid_shape(j)
        for ix in range(gx):
            for iy in range(gy):
                for iz in range(gz):
                    for s in array.read_bin(j, ix, iy, iz):
                        if s.prominence == j:
                            out[int(s.id)] = (j, ix, iy, iz)
    return out


def _get_by_id_index(array, store_path: str, array_name: str) -> dict:
    """
    Return the by_id index, building+caching if needed.

    Cached entry is keyed by (store_path, array_name, "by_id_index") so it
    lives in the global _by_id_index_cache and participates in reference-
    counted eviction.
    """
    key = (store_path, array_name, "by_id_index")
    cached = _by_id_index_cache.get(key)
    if cached is not _SizedLruCache.MISS:
        return cached

    idx = _build_by_id_index(array)
    # Size estimate: each entry is int key + 4-tuple of ints.
    # Underestimate; ~80 bytes/entry is a safe rough figure.
    est_size = len(idx) * 80 + 64
    _by_id_index_cache.put(key, idx, est_size)
    return idx


def _locate_annotation(array, segment_id, store_path=None, array_name=None):
    """
    Locate a segment: return (zv_level, ix, iy, iz) for its home bin.

    Prefers `array.locate(id)` if available; falls back to cached lazy index.
    If `store_path` and `array_name` are provided, the lazy index is cached
    in the global cache keyed by (store_path, array_name, "by_id_index").
    If not provided (legacy callers from Session 7 tests), falls back to an
    inline rebuild each call — slower but functionally correct.
    """
    locate = getattr(array, "locate", None)
    if callable(locate):
        result = locate(int(segment_id))
        if result is None:
            raise KeyError(segment_id)
        prominence, cell = result
        return (int(prominence), int(cell[0]), int(cell[1]), int(cell[2]))

    if store_path is not None and array_name is not None:
        idx = _get_by_id_index(array, store_path, array_name)
    else:
        # Legacy path, no caching (Session 7 compatibility)
        idx = _build_by_id_index(array)

    if int(segment_id) not in idx:
        raise KeyError(segment_id)
    return idx[int(segment_id)]


def serve_annotation_by_id(array, segment_id, annotation_type=None,
                            store_path=None, array_name=None):
    """
    Serve a single annotation as a by_id resource.

    `store_path` and `array_name` enable caching of the lazy index; they
    should be passed by the Handler, which has them from parse_url. Omitted
    for direct test calls (functional but uncached).
    """
    annotation_type = _resolve_annotation_type(array, annotation_type)

    # Session 9: check staleness before the by_id index lookup
    _check_and_evict_if_stale(array, store_path, array_name)

    zv_level, ix, iy, iz = _locate_annotation(
        array, segment_id, store_path, array_name
    )

    for s in array.read_bin(zv_level, ix, iy, iz):
        if int(s.id) == int(segment_id) and s.prominence == zv_level:
            properties_schema = _get_properties_schema(array)
            relationships = []
            return encode_single_annotation(
                s, annotation_type, properties_schema, relationships
            )
    raise KeyError(segment_id)


# =============================================================================
# Limit computation
# =============================================================================

def next_power_of_two(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _compute_limit_per_level(array, cap=10000):
    L = int(array.levels)
    precomputed = getattr(array, "limit_per_level", None)
    if callable(precomputed):
        raw = list(precomputed())
        if len(raw) != L:
            raise ValueError(
                f"array.limit_per_level() returned {len(raw)} entries, expected {L}"
            )
        return [min(next_power_of_two(max(1, int(r))), cap) for r in raw]
    LOG.info("array lacks limit_per_level(); scanning %d levels", L)
    limits = []
    for k in range(L):
        zv_level = L - 1 - k
        gx, gy, gz = array.grid_shape(zv_level)
        max_count = 0
        for ix in range(gx):
            for iy in range(gy):
                for iz in range(gz):
                    count = 0
                    for s in array.read_bin(zv_level, ix, iy, iz):
                        if s.prominence == zv_level:
                            count += 1
                    if count > max_count:
                        max_count = count
        limits.append(min(next_power_of_two(max(1, max_count)), cap))
    return limits


# =============================================================================
# info JSON builder (with caching)
# =============================================================================

def _build_info_raw(array, annotation_type):
    annotation_type = _resolve_annotation_type(array, annotation_type)
    L = int(array.levels)
    if L < 1:
        raise ValueError(f"array.levels must be >= 1, got {L}")
    Nx, Ny, Nz = array.voxel_extent
    limits = _compute_limit_per_level(array)
    spatial = []
    for k in range(L):
        zv_level = L - 1 - k
        bin_sz = array.bin_size(zv_level)
        grid = array.grid_shape(zv_level)
        spatial.append({
            "key": f"spatial{k}",
            "chunk_size": [int(b) for b in bin_sz],
            "grid_shape": [int(g) for g in grid],
            "limit": int(limits[k]),
        })
    properties_schema = _get_properties_schema(array)
    return {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {"x": [1, ""], "y": [1, ""], "z": [1, ""]},
        "lower_bound": [0, 0, 0],
        "upper_bound": [int(Nx), int(Ny), int(Nz)],
        "annotation_type": annotation_type,
        "properties": properties_schema,
        "relationships": [],
        "by_id": {"key": "by_id"},
        "spatial": spatial,
    }


def build_info(array, annotation_type=None, *,
               store_path=None, array_name=None):
    """
    Build annotation info. If store_path/array_name provided, cache the
    resulting JSON (as a dict; each request gets a fresh copy to prevent
    handlers from accidentally mutating the cached value).
    """
    if store_path is None or array_name is None:
        # Uncached path (legacy test callers)
        return _build_info_raw(array, annotation_type)

    # Session 9: check staleness before cache lookup
    _check_and_evict_if_stale(array, store_path, array_name)

    # Include annotation_type in the key because different handlers can
    # request different declared types for the same array.
    key = (store_path, array_name, "info", annotation_type)
    cached = _info_cache.get(key)
    if cached is not _SizedLruCache.MISS:
        # Return a deep copy so callers can mutate without corrupting cache.
        return _copy_info(cached)

    info = _build_info_raw(array, annotation_type)
    # Cache a canonical copy; caller gets another copy.
    cached_copy = _copy_info(info)
    est = _estimate_json_size(cached_copy)
    _info_cache.put(key, cached_copy, est)
    return info


def _copy_info(info: dict) -> dict:
    """Cheap deep copy for info JSONs (lists + dicts + primitives)."""
    return json.loads(json.dumps(info))


def _estimate_json_size(obj) -> int:
    """Rough byte estimate of a JSON-serializable object."""
    return len(json.dumps(obj).encode())


# =============================================================================
# Spatial chunk server (with caching)
# =============================================================================

def serve_spatial_chunk(array, level, chunk_key, annotation_type=None,
                        *, store_path=None, array_name=None):
    """
    Serve a spatial chunk. Cached by (store_path, array_name, "spatial",
    level, chunk_key, annotation_type) when store_path/array_name provided.
    """
    annotation_type = _resolve_annotation_type(array, annotation_type)
    try:
        ix, iy, iz = (int(p) for p in chunk_key.split("_"))
    except ValueError as e:
        raise ValueError(f"malformed chunk key {chunk_key!r}: {e}")
    L = int(array.levels)
    if not (0 <= level < L):
        raise ValueError(f"level {level} out of range for L={L}")

    if store_path is not None and array_name is not None:
        # Session 9: check staleness before cache lookup
        _check_and_evict_if_stale(array, store_path, array_name)
        key = (store_path, array_name, "spatial", level, chunk_key, annotation_type)
        cached = _geometry_cache.get(key)
        if cached is not _SizedLruCache.MISS:
            return cached

    zv_level = L - 1 - level
    segments = array.read_bin(zv_level, ix, iy, iz)
    emit = [s for s in segments if s.prominence == zv_level]
    properties_schema = _get_properties_schema(array)
    body = encode_annotation_chunk(emit, annotation_type, properties_schema)

    if store_path is not None and array_name is not None:
        _geometry_cache.put(key, body, len(body))
    return body


# =============================================================================
# Mesh handler
# =============================================================================

_FRAGMENT_PREFIX = "frag_"
_MANIFEST_SUFFIX = ":0"
_SEGMENT_PROPERTIES_PREFIX = "segment_properties/"

_SEGMENT_PROP_SCALAR_TYPES = {
    "uint32", "int32", "float32", "uint16", "int16", "uint8", "int8",
}
_SEGMENT_PROP_STRING_TYPES = {"label", "description", "string"}
_SEGMENT_PROP_OTHER_TYPES = {"tags"}
_SEGMENT_PROP_ALLOWED = (
    _SEGMENT_PROP_SCALAR_TYPES
    | _SEGMENT_PROP_STRING_TYPES
    | _SEGMENT_PROP_OTHER_TYPES
)


def _check_mesh_datatype(array):
    if array.datatype != "mesh":
        raise ValueError(f"expected mesh array, got datatype {array.datatype!r}")


def build_mesh_info(array):
    _check_mesh_datatype(array)
    info = {"@type": "neuroglancer_legacy_mesh"}
    sp = getattr(array, "segment_properties", None)
    if sp:
        info["segment_properties"] = "segment_properties"
    return info


def serve_mesh_manifest(array, segment_id):
    _check_mesh_datatype(array)
    segment_ids = set(array.iter_segment_ids())
    if segment_id not in segment_ids:
        raise KeyError(segment_id)
    return {"fragments": [f"{_FRAGMENT_PREFIX}{segment_id}"]}


def serve_mesh_fragment(array, segment_id, *, bake_affine=False,
                        store_path=None, array_name=None):
    """Cached mesh fragment serving. Uses _geometry_cache."""
    _check_mesh_datatype(array)

    if store_path is not None and array_name is not None:
        # Session 9: check staleness before cache lookup
        _check_and_evict_if_stale(array, store_path, array_name)
        key = (store_path, array_name, "mesh_fragment", int(segment_id), bool(bake_affine))
        cached = _geometry_cache.get(key)
        if cached is not _SizedLruCache.MISS:
            return cached

    try:
        mesh = array.get_mesh(segment_id)
    except KeyError:
        raise
    except Exception as e:
        raise KeyError(segment_id) from e

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    indices = np.asarray(mesh.indices, dtype=np.uint32)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"segment {segment_id}: vertices must be shape (V, 3), "
            f"got {vertices.shape}"
        )
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(
            f"segment {segment_id}: indices must be shape (T, 3), "
            f"got {indices.shape}"
        )

    num_vertices = int(vertices.shape[0])
    num_triangles = int(indices.shape[0])

    if num_triangles > 0 and num_vertices > 0:
        max_idx = int(indices.max())
        if max_idx >= num_vertices:
            raise ValueError(
                f"segment {segment_id}: triangle index {max_idx} >= "
                f"num_vertices {num_vertices}"
            )
    elif num_triangles > 0 and num_vertices == 0:
        raise ValueError(f"segment {segment_id}: has triangles but zero vertices")

    if bake_affine:
        affine = np.asarray(array.affine, dtype=np.float64)
        if affine.shape != (4, 4):
            raise ValueError(f"array.affine must be (4, 4), got {affine.shape}")
        if num_vertices > 0:
            homog = np.c_[vertices.astype(np.float64), np.ones(num_vertices)]
            vertices = (homog @ affine.T)[:, :3].astype(np.float32)

    vertices_le = np.ascontiguousarray(vertices, dtype="<f4")
    indices_le = np.ascontiguousarray(indices, dtype="<u4")
    out = bytearray()
    out.extend(struct.pack("<I", num_vertices))
    out.extend(vertices_le.tobytes(order="C"))
    out.extend(indices_le.tobytes(order="C"))
    body = bytes(out)

    if store_path is not None and array_name is not None:
        _geometry_cache.put(key, body, len(body))
    return body


def build_segment_properties_info(array):
    _check_mesh_datatype(array)
    props = getattr(array, "segment_properties", None) or []
    seg_ids = getattr(array, "segment_property_ids", None)

    if not props:
        raise ValueError(f"{array!r}: segment_properties is empty or missing")
    if seg_ids is None:
        raise ValueError(
            f"{array!r}: segment_property_ids is required alongside segment_properties"
        )

    seg_ids_list = [str(int(sid)) for sid in seg_ids]
    n = len(seg_ids_list)

    validated = []
    seen_ids = set()
    label_index = None
    for i, p in enumerate(props):
        if not isinstance(p, dict):
            raise ValueError(
                f"segment_properties[{i}] must be a dict, got {type(p).__name__}"
            )
        if "id" not in p or "type" not in p or "values" not in p:
            raise ValueError(f"segment_properties[{i}] missing id/type/values: {p!r}")
        pid = p["id"]
        ptype = p["type"]
        pvalues = list(p["values"])
        if not _PROPERTY_ID_RE.match(pid):
            raise ValueError(f"property id {pid!r} must match [a-z][a-zA-Z0-9_]*")
        if pid in seen_ids:
            raise ValueError(f"duplicate property id {pid!r}")
        if ptype not in _SEGMENT_PROP_ALLOWED:
            raise ValueError(
                f"property {pid!r}: type {ptype!r} not allowed in "
                f"segment_properties; allowed: {sorted(_SEGMENT_PROP_ALLOWED)}"
            )
        if len(pvalues) != n:
            raise ValueError(
                f"property {pid!r}: {len(pvalues)} values for {n} segment ids"
            )
        if ptype == "label":
            label_index = i
        seen_ids.add(pid)
        out = {"id": pid, "type": ptype, "values": pvalues}
        if "description" in p:
            out["description"] = p["description"]
        validated.append(out)

    if label_index is not None and label_index != 0:
        validated.insert(0, validated.pop(label_index))

    return {
        "@type": "neuroglancer_segment_properties",
        "inline": {"ids": seg_ids_list, "properties": validated},
    }


def _parse_mesh_resource(resource):
    if resource == "info":
        return ("info", None)
    if resource == _SEGMENT_PROPERTIES_PREFIX + "info":
        return ("seg_props", None)
    if resource.endswith(_MANIFEST_SUFFIX):
        head = resource[:-len(_MANIFEST_SUFFIX)]
        if head.isdigit():
            return ("manifest", int(head))
    if resource.startswith(_FRAGMENT_PREFIX):
        tail = resource[len(_FRAGMENT_PREFIX):]
        if tail.isdigit():
            return ("fragment", int(tail))
    return (None, None)


# =============================================================================
# Fan-out
# =============================================================================

def _default_base_name(store_path):
    basename = store_path.rstrip("/").split("/")[-1]
    if not basename:
        return "zv"
    for ext in (".zv", ".zarr"):
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
            break
    return basename or "zv"


def _build_source_url(fileserver_base, protocol, store_path, array_name, datatype):
    kind = _ZV_DATATYPE_TO_ROUTE_KIND.get(datatype)
    if kind is None:
        raise ValueError(f"unknown datatype {datatype!r}")
    # Neuroglancer needs the `precomputed://` scheme prefix so it knows to
    # fetch `/info` rather than probing for zarr / OCDBT manifest files.
    return (
        "precomputed://"
        + fileserver_base.rstrip("/")
        + f"/zv/{kind}/{protocol}/{store_path}/@{array_name}"
    )


def _make_layer_spec(array, array_name, protocol, store_path,
                     base_name, fileserver_base):
    datatype = array.datatype
    if datatype not in _ZV_DATATYPE_TO_ROUTE_KIND:
        raise ValueError(f"unknown datatype {datatype!r}")
    layer_type = _ZV_DATATYPE_TO_LAYER_TYPE[datatype]
    annotation_type = _ZV_DATATYPE_TO_ANNOTATION.get(datatype)
    return {
        "name":            f"{base_name}/{array_name}",
        "source_url":      _build_source_url(
            fileserver_base, protocol, store_path, array_name, datatype),
        "layer_type":      layer_type,
        "annotation_type": annotation_type,
        "datatype":        datatype,
        "affine":          np.asarray(array.affine, dtype=np.float64),
        "voxel_extent":    tuple(int(v) for v in array.voxel_extent),
        # === SLAB/CONE-VIEW SANDBOX BEGIN ===================================
        # Default annotation shader, picked by datatype. scene.py applies
        # it on layer registration; the user can override via the layer
        # Render tab without editing the package.
        "default_shader":  _DEFAULT_SHADERS_BY_DATATYPE.get(datatype),
        # === SLAB/CONE-VIEW SANDBOX END =====================================
        # Session 8: Scene.load passes these back to unregister_array_usage
        # when unloading. They identify exactly which refcount entry to drop.
        "_cache_identity": (protocol, store_path, array_name),
    }


def expand_zarrvectors_url(url, fileserver_base, *, include_mesh=True):
    """
    Expand a zarrvectors:// URL to layer specs. Does NOT call
    register_array_usage — Scene.load does that after receiving the specs
    and deciding which to actually register with Neuroglancer.
    """
    protocol, store_path, array_filter = parse_zarrvectors_url(url)
    store = _open_store(protocol, store_path)
    base_name = _default_base_name(store_path)
    layers = []
    unknown_types = []
    for array_name, array in store.items():
        if array_filter is not None and array_name != array_filter:
            continue
        datatype = getattr(array, "datatype", None)
        if datatype is None:
            unknown_types.append((array_name, None))
            continue
        if datatype not in _ZV_DATATYPE_TO_ROUTE_KIND:
            unknown_types.append((array_name, datatype))
            continue
        if datatype == "mesh" and not include_mesh:
            continue
        try:
            layers.append(_make_layer_spec(
                array=array, array_name=array_name,
                protocol=protocol, store_path=store_path,
                base_name=base_name, fileserver_base=fileserver_base,
            ))
        except ValueError as e:
            LOG.warning("skipping array %r: %s", array_name, e)
    for array_name, datatype in unknown_types:
        LOG.warning(
            "skipping array %r with unsupported datatype %r",
            array_name, datatype,
        )
    if array_filter is not None and not layers:
        raise ValueError(
            f"array {array_filter!r} not found in store {store_path!r}"
        )
    return layers


# =============================================================================
# HTTP handler classes
# =============================================================================
# These classes expose the pure-logic functions above over the fileserver.
# Routes are registered in ngtools/local/viewer.py:
#
#   (r"^/zv/streamlines/(.*)", ZarrVectorsStreamlineHandler),
#   (r"^/zv/points/(.*)",      ZarrVectorsPointHandler),
#   (r"^/zv/mesh/(.*)",        ZarrVectorsMeshHandler),
#
# The `(.*)` capture becomes the `path` argument to `Handler.get()`.
#
# Handler base-class assumptions (match `TractAnnotationHandler` from PR #38):
#   - `self.headers` is a dict populated with response headers
#   - `self.body` holds the response bytes
#   - `self.status` is the HTTP status code (default 200)
#   - Returning `None` from the handler signals the response is ready
#
# If the fork's `Handler` base class differs, adjust `_respond_*` and `get()`
# accordingly — everything else here is framework-agnostic.

from ngtools.local.handlers import Handler


class _ZarrVectorsAnnotationHandlerBase(Handler):
    """
    Dispatches for annotation layers (streamline + point share this logic):
      info                      -> build_info
      spatial{k}/{ix}_{iy}_{iz} -> serve_spatial_chunk
      by_id/{id}                -> serve_annotation_by_id

    Subclass sets ANNOTATION_TYPE to "LINE" or "POINT".
    """

    ANNOTATION_TYPE: str = ""

    def get(self, path: str):
        try:
            protocol, store_path, array_name, resource = parse_url(path)
        except ValueError as e:
            return self._respond_404(str(e))

        try:
            store = _open_store(protocol, store_path)
            array = store[array_name]
        except KeyError:
            return self._respond_404(f"array {array_name!r} not found")
        except Exception as e:
            LOG.exception("failed to open zarr-vectors store")
            return self._respond_404(f"open failed: {e}")

        if resource == "info":
            try:
                info = build_info(
                    array, self.ANNOTATION_TYPE,
                    store_path=store_path, array_name=array_name,
                )
            except ValueError as e:
                return self._respond_404(str(e))
            self.status = 200
            self.headers["Content-type"] = "application/json"
            self.body = json.dumps(info).encode()
            return None

        if "/" in resource:
            head, tail = resource.split("/", 1)

            if head.startswith("spatial"):
                try:
                    level = int(head[len("spatial"):])
                except ValueError:
                    return self._respond_404(f"bad spatial key: {head}")
                try:
                    body = serve_spatial_chunk(
                        array, level, tail, self.ANNOTATION_TYPE,
                        store_path=store_path, array_name=array_name,
                    )
                except ValueError as e:
                    return self._respond_404(str(e))
                self.status = 200
                self.headers["Content-type"] = "application/octet-stream"
                self.body = body
                return None

            if head == "by_id":
                if not tail.isdigit():
                    return self._respond_404(f"bad by_id: {tail}")
                try:
                    body = serve_annotation_by_id(
                        array, int(tail), self.ANNOTATION_TYPE,
                        store_path=store_path, array_name=array_name,
                    )
                except KeyError:
                    return self._respond_404(f"segment {tail} not found")
                except ValueError as e:
                    return self._respond_404(str(e))
                self.status = 200
                self.headers["Content-type"] = "application/octet-stream"
                self.body = body
                return None

        return self._respond_404(f"unknown resource: {resource}")

    def _respond_404(self, msg: str):
        LOG.warning("ZV 404: %s", msg)
        self.status = 404
        self.headers["Content-type"] = "text/plain"
        self.body = msg.encode()
        return None


class ZarrVectorsStreamlineHandler(_ZarrVectorsAnnotationHandlerBase):
    """Serves streamline (LINE) annotation layers."""
    ANNOTATION_TYPE = "LINE"


class ZarrVectorsPointHandler(_ZarrVectorsAnnotationHandlerBase):
    """Serves point (POINT) annotation layers."""
    ANNOTATION_TYPE = "POINT"


class ZarrVectorsMeshHandler(Handler):
    """
    Dispatches for mesh layers:
      info                     -> build_mesh_info
      segment_properties/info  -> build_segment_properties_info
      {segment_id}:0           -> serve_mesh_manifest
      frag_{segment_id}        -> serve_mesh_fragment
    """

    def get(self, path: str):
        try:
            protocol, store_path, array_name, resource = parse_url(path)
        except ValueError as e:
            return self._respond_404(str(e))

        try:
            store = _open_store(protocol, store_path)
            array = store[array_name]
        except KeyError:
            return self._respond_404(f"array {array_name!r} not found")
        except Exception as e:
            LOG.exception("failed to open zarr-vectors store")
            return self._respond_404(f"open failed: {e}")

        try:
            _check_mesh_datatype(array)
        except ValueError as e:
            return self._respond_404(str(e))

        kind, seg_id = _parse_mesh_resource(resource)

        if kind == "info":
            self.status = 200
            self.headers["Content-type"] = "application/json"
            self.body = json.dumps(build_mesh_info(array)).encode()
            return None

        if kind == "seg_props":
            try:
                info = build_segment_properties_info(array)
            except ValueError as e:
                return self._respond_404(str(e))
            self.status = 200
            self.headers["Content-type"] = "application/json"
            self.body = json.dumps(info).encode()
            return None

        if kind == "manifest":
            try:
                manifest = serve_mesh_manifest(array, seg_id)
            except KeyError:
                return self._respond_404(f"segment {seg_id} not found")
            self.status = 200
            self.headers["Content-type"] = "application/json"
            self.body = json.dumps(manifest).encode()
            return None

        if kind == "fragment":
            try:
                body = serve_mesh_fragment(
                    array, seg_id,
                    store_path=store_path, array_name=array_name,
                )
            except KeyError:
                return self._respond_404(f"segment {seg_id} not found")
            except ValueError as e:
                return self._respond_404(str(e))
            self.status = 200
            self.headers["Content-type"] = "application/octet-stream"
            self.body = body
            return None

        return self._respond_404(f"unknown mesh resource: {resource}")

    def _respond_404(self, msg: str):
        LOG.warning("ZV 404: %s", msg)
        self.status = 404
        self.headers["Content-type"] = "text/plain"
        self.body = msg.encode()
        return None
