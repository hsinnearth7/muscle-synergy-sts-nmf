"""Decode MATLAB ``table`` / MCOS data stored in a Gait120 ``.mat`` file.

MATLAB's table class is serialized as an ``mxOPAQUE`` element pointing into
the hidden ``__function_workspace__`` MAT5 subsystem. Neither scipy nor
pymatreader decodes this automatically. This module parses the subsystem
manually, resolves object IDs, and returns per-trial, per-muscle numeric
arrays.

The format has been reverse-engineered; see:
- https://www.mathworks.com/help/matlab/import_export/mat-file-versions.html
- scipy.io.matlab._mio5 (for element layout)
- https://github.com/skjerns/mat7.3 (v7.3 parallel reference)

This parser targets the exact structure used in the Gait120 dataset:
each EMG recording is stored as a MATLAB ``table`` wrapping a single
``(n_samples, 12)`` double matrix.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# MAT5 element decoding
# ---------------------------------------------------------------------------

_MAX_PARSE_DEPTH = 64


_NUMERIC_TARGET = {
    6: "<f8", 7: "<f4", 8: "<i1", 9: "<u1",
    10: "<i2", 11: "<u2", 12: "<i4", 13: "<u4", 14: "<i8", 15: "<u8",
}

_MI_DTYPE = {
    1: "<i1", 2: "<u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4",
    7: "<f4", 9: "<f8", 12: "<i8", 13: "<u8",
}


def _read_tag(buf: bytes, off: int) -> tuple[int, int, int, int]:
    """Return (mi_type, nbytes, data_start, next_offset) for one MAT5 element tag."""
    first4 = struct.unpack_from("<I", buf, off)[0]
    n_small = (first4 >> 16) & 0xFFFF
    t_small = first4 & 0xFFFF
    if n_small != 0 and t_small != 0:
        # Small-data element: 4 byte tag with n_small bytes payload fitting into 4 bytes.
        return t_small, n_small, off + 4, off + 8
    t = struct.unpack_from("<I", buf, off)[0]
    n = struct.unpack_from("<I", buf, off + 4)[0]
    ds = off + 8
    total = 8 + n
    if total % 8:
        total += 8 - (total % 8)
    return t, n, ds, off + total


def _read_int8_string(buf: bytes, off: int) -> tuple[str, int]:
    t, n, ds, nx = _read_tag(buf, off)
    if t != 1:
        raise ValueError(f"expected miINT8 string at {off}, got t={t}")
    return buf[ds : ds + n].decode("ascii", "replace"), nx


@dataclass
class Node:
    """A single parsed miMATRIX element."""
    kind: str
    mxclass: int
    dims: tuple
    name: str
    value: Any
    next_off: int


def _parse_matrix(buf: bytes, off: int, _depth: int = 0) -> Node:
    """Parse one miMATRIX element starting at ``off`` in the MAT5 byte stream."""
    if _depth > _MAX_PARSE_DEPTH:
        raise ValueError(
            f"miMATRIX nesting depth exceeded {_MAX_PARSE_DEPTH} at offset {off}"
        )
    t, n, ds, nx = _read_tag(buf, off)
    if t != 14:
        raise ValueError(f"expected miMATRIX at offset {off}, got t={t} n={n}")
    if n == 0:
        # Empty miMATRIX placeholder (used in cell arrays for missing entries).
        return Node("empty", 0, (), "", None, nx)

    p = ds
    tt, tn, tds, tnx = _read_tag(buf, p)
    if tt != 6:
        raise ValueError(f"array_flags expected at {p}, got t={tt}")
    flags_u32 = struct.unpack_from("<I", buf, tds)[0]
    mxclass = flags_u32 & 0xFF
    p = tnx

    if mxclass == 17:  # mxOPAQUE
        s0, p = _read_int8_string(buf, p)
        s1, p = _read_int8_string(buf, p)
        s2, p = _read_int8_string(buf, p)
        inner = _parse_matrix(buf, p, _depth + 1)
        return Node("opaque", mxclass, (s0, s1, s2), "", inner, nx)

    # dims
    tt, tn, tds, tnx = _read_tag(buf, p)
    if tt != 5:
        raise ValueError(f"dims expected at {p}, got t={tt}")
    dims = tuple(int(x) for x in np.frombuffer(buf[tds : tds + tn], dtype="<i4"))
    p = tnx
    # array name
    tt, tn, tds, tnx = _read_tag(buf, p)
    name = buf[tds : tds + tn].decode("ascii", "replace") if tn else ""
    p = tnx

    if mxclass in _NUMERIC_TARGET:
        # possibly complex? For simplicity we ignore imag component.
        tt, tn, tds, tnx = _read_tag(buf, p)
        raw_dt = _MI_DTYPE.get(tt)
        if raw_dt is None:
            arr = None
        else:
            raw = np.frombuffer(buf[tds : tds + tn], dtype=raw_dt)
            arr = raw.astype(_NUMERIC_TARGET[mxclass])
            if dims:
                total_elems = int(np.prod(dims))
                if arr.size == total_elems:
                    arr = arr.reshape(dims, order="F")
        return Node("numeric", mxclass, dims, name, arr, nx)

    if mxclass == 1:  # CELL
        n_elements = int(np.prod(dims)) if dims else 1
        cells: list[Node] = []
        for _ in range(n_elements):
            sub = _parse_matrix(buf, p, _depth + 1)
            p = sub.next_off
            cells.append(sub)
        return Node("cell", mxclass, dims, name, cells, nx)

    if mxclass == 2:  # STRUCT
        tt, tn, tds, tnx = _read_tag(buf, p)
        if tt != 5:
            raise ValueError(f"fn_len expected miINT32 at {p}, got t={tt}")
        fn_len = int(np.frombuffer(buf[tds : tds + 4], dtype="<i4")[0])
        if fn_len <= 0:
            raise ValueError(f"invalid fn_len={fn_len} in STRUCT at offset {p}")
        p = tnx
        tt, tn, tds, tnx = _read_tag(buf, p)
        if tt != 1:
            raise ValueError(f"fieldnames expected miINT8 at {p}, got t={tt}")
        n_fields = tn // fn_len
        field_names = [
            buf[tds + i * fn_len : tds + (i + 1) * fn_len].split(b"\x00", 1)[0].decode("ascii")
            for i in range(n_fields)
        ]
        p = tnx
        n_elements = int(np.prod(dims)) if dims else 1
        out: list[dict[str, Node]] = []
        for _ in range(n_elements):
            record: dict[str, Node] = {}
            for fn in field_names:
                sub = _parse_matrix(buf, p, _depth + 1)
                p = sub.next_off
                record[fn] = sub
            out.append(record)
        return Node("struct", mxclass, dims, name, out, nx)

    if mxclass == 4:  # CHAR
        tt, tn, tds, tnx = _read_tag(buf, p)
        if tt in (1, 2):
            s = buf[tds : tds + tn].decode("utf-8", "replace")
        elif tt == 4:
            s = buf[tds : tds + tn].decode("utf-16-le", "replace")
        else:
            s = buf[tds : tds + tn].decode("latin-1", "replace")
        return Node("char", mxclass, dims, name, s, nx)

    # Fallback: skip unknown class body, but advance to nx.
    return Node("other", mxclass, dims, name, None, nx)


# ---------------------------------------------------------------------------
# MCOS object pool decoding
# ---------------------------------------------------------------------------


@dataclass
class McosPool:
    """Decoded MATLAB MCOS (FileWrapper__) object pool from __function_workspace__.

    Attributes
    ----------
    class_names : list[str]
        Class names referenced by objects in the pool.
    field_names : list[str]
        Shared identifier pool (class + property names).
    objects : list[dict]
        Decoded object records. Each record is a dict with keys
        ``{'class_id', 'properties'}`` where ``properties`` maps
        property name -> decoded Node (or numpy array for leaves).
    data_cells : list[Node]
        Raw cell[1..N] from the subsystem -- the numeric payloads referenced
        by object properties.
    """

    class_names: list[str] = field(default_factory=list)
    field_names: list[str] = field(default_factory=list)
    data_cells: list[Node] = field(default_factory=list)
    objects: list[dict[str, Any]] = field(default_factory=list)


def load_function_workspace(mat_path: Path | str) -> bytes:
    """Load the raw ``__function_workspace__`` byte buffer from a MAT5 file."""
    d = loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    fw = d["__function_workspace__"].ravel().tobytes()
    return fw


def parse_subsystem(fw: bytes) -> McosPool:
    """Parse ``__function_workspace__`` into a McosPool.

    The overall layout (for Gait120 and most MATLAB releases)::

        miMATRIX struct {1,1}
          field "MCOS": miMATRIX opaque
             s0 = ""
             s1 = "MCOS"
             s2 = "FileWrapper__"
             arr = miMATRIX cell{N,1}
                cell[0]         -> uint8 metadata blob (names + offsets)
                cell[1..N-2]    -> actual data payloads (numeric arrays)
                cell[N-1]       -> trailing struct with class / property tables
    """
    # The subsystem starts with an 8 byte mini-header (version + endian).
    # Skip it.
    if not (fw[:4] == b"\x00\x01IM\x00"[:4]
            or fw[:4] == b"\x00\x01MI\x00"[:4]
            or fw[:2] == b"\x00\x01"):
        raise ValueError(f"Unexpected MAT5 subsystem header: {fw[:8]!r}")
    root = _parse_matrix(fw, 8)
    if root.kind != "struct":
        raise ValueError(f"Expected root miMATRIX struct, got kind={root.kind!r}")
    mcos_field = root.value[0]["MCOS"]
    if mcos_field.kind != "opaque":
        raise ValueError(f"Expected MCOS opaque element, got kind={mcos_field.kind!r}")
    cell_root = mcos_field.value
    if cell_root.kind != "cell":
        raise ValueError(f"Expected cell array inside MCOS opaque, got kind={cell_root.kind!r}")
    cells: list[Node] = cell_root.value

    pool = McosPool(data_cells=cells)

    if not cells:
        return pool

    # The first cell is a uint8 metadata blob holding class and property names
    # along with the property table. We decode the names and class table here.
    meta = cells[0]
    if meta.kind == "numeric" and meta.value is not None and meta.mxclass == 9:
        _decode_metadata_blob(meta.value.reshape(-1).astype(np.uint8).tobytes(), pool)

    return pool


def _decode_metadata_blob(blob: bytes, pool: McosPool) -> None:
    """Decode the first uint8 cell of the MCOS subsystem.

    Layout (validated against scipy issue #6395 and skjerns/mat7.3):

        0..3   : uint32 "magic" (typically 2 or 4, represents version)
        4..7   : uint32 number of strings in the shared name pool
        8..39  : 8 uint32 header fields, most of which are offsets within blob
        40..   : null-terminated ASCII strings (name pool)
        ...    : region 1..5 records (class list, property tables, ...)

    We only extract the shared name pool, which is enough to drive a later
    property-level lookup. Full class decoding is not needed for Gait120
    since each trial's EMG data is always at a well-known data cell.
    """
    if len(blob) < 40:
        return
    header = struct.unpack_from("<10I", blob, 0)
    n_strings = int(header[1])
    # Strings start after 40-byte header.
    p = 40
    names: list[str] = []
    while len(names) < n_strings and p < len(blob):
        end = blob.find(b"\x00", p)
        if end < 0 or (end == p and len(names) > 0):
            break
        name = blob[p:end].decode("ascii", "replace")
        names.append(name)
        p = end + 1
        # Skip any extra null padding.
        while p < len(blob) and blob[p] == 0 and len(names) == n_strings:
            p += 1
    pool.field_names = names


# ---------------------------------------------------------------------------
# Gait120-specific helpers
# ---------------------------------------------------------------------------


def iter_double_arrays(pool: McosPool) -> list[tuple[int, np.ndarray]]:
    """Return ``[(cell_index, array), ...]`` for every cell that holds a double matrix.

    This is the main workhorse for Gait120: every ``table`` column holds a
    ``(n_samples, 12)`` double matrix, and those matrices show up as numeric
    mxDOUBLE cells in the subsystem pool.
    """
    out: list[tuple[int, np.ndarray]] = []
    for i, cell in enumerate(pool.data_cells):
        if cell.kind == "numeric" and cell.mxclass == 6 and cell.value is not None:
            out.append((i, cell.value))
    return out
