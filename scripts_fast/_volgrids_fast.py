#!/usr/bin/env python3
"""
`volgrids` CLI, same results (checked bit-identical on 1AJU), faster:

- Grid.reshape: volgrids evaluates RegularGridInterpolator on every point of
  an np.mgrid. The target points form a tensor grid, so the indices and
  weights are computed once per axis and broadcast, with exactly the same
  arithmetic as scipy's linear evaluation (same per-axis find_indices, same
  weight product order, same corner summation order), and points outside the
  source grid (fill value 0) are never evaluated. `smiffer -a` ~4x faster.
- DX reading: gridData's DX parser reads the data block line by line; it is
  read in one go instead (same values).
- VOLGRIDS_APBS_DONE / VOLGRIDS_APBS_DX: lets `smiffer -a` start before APBS
  has finished (see _wait_for_apbs).

Usage: python3 _volgrids_fast.py <volgrids arguments...>   (same as `volgrids ...`)
"""

import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate._rgi_cython import find_indices

import volgrids as vg


def interpolate_tensor(src_axes, values, new_axes, fill_value=0.0):
    """RegularGridInterpolator(src_axes, values, bounds_error=False,
    fill_value=fill_value) on the tensor grid new_axes, bit-identical."""
    out = np.full(tuple(len(a) for a in new_axes), fill_value, dtype=np.float64)
    idx, w, w1, box = [], [], [], []
    for a_new, a_src in zip(new_axes, src_axes):
        inb = np.flatnonzero((a_new >= a_src[0]) & (a_new <= a_src[-1]))
        if len(inb) == 0:
            return out
        lo, hi = inb[0], inb[-1] + 1
        i, d = find_indices((a_src,), a_new[lo:hi][None, :])
        idx.append(i[0]); w.append(d[0]); w1.append(1 - d[0]); box.append(slice(lo, hi))
    values = np.asarray(values, dtype=np.float64)
    shape = [(-1, 1, 1), (1, -1, 1), (1, 1, -1)]

    value = np.array([0.])
    for corner in itertools.product((0, 1), repeat=3):
        weight = np.array([1.])
        for ax, c in enumerate(corner):
            weight = weight * (w[ax] if c else w1[ax]).reshape(shape[ax])
        value = value + values[np.ix_(*(idx[ax] + c for ax, c in enumerate(corner)))] * weight
    out[tuple(box)] = value
    return out


def reshape(self, new_min, new_max, new_res):
    """Drop-in for vg.Grid.reshape (same result, see module docstring)."""
    new_xmin, new_ymin, new_zmin = new_min
    new_xmax, new_ymax, new_zmax = new_max
    new_xres, new_yres, new_zres = new_res

    # 1-D np.mgrid gives the same axis values as the 3-D one volgrids builds;
    # RegularGridInterpolator casts grid axes and query points to float64.
    new_axes = tuple(np.asarray(np.mgrid[lo:hi:complex(0, n)], dtype=float) for lo, hi, n in (
        (new_xmin, new_xmax, new_xres), (new_ymin, new_ymax, new_yres), (new_zmin, new_zmax, new_zres)))
    src_axes = tuple(np.asarray(a, dtype=float) for a in (
        np.linspace(self.xmin(), self.xmax(), self.xres()),
        np.linspace(self.ymin(), self.ymax(), self.yres()),
        np.linspace(self.zmin(), self.zmax(), self.zres()),
    ))
    self.arr = interpolate_tensor(src_axes, self.arr, new_axes).astype(vg.FLOAT_DTYPE)

    self.box.min_coords = np.array([new_xmin, new_ymin, new_zmin])
    self.box.max_coords = np.array([new_xmax, new_ymax, new_zmax])
    self.box.resolution = np.array([new_xres, new_yres, new_zres])
    self.box.deltas = (self.box.max_coords - self.box.min_coords) / (self.box.resolution - 1)
    self.box.infer_radius_and_cog()


def _dx_array(self):
    """gridData.OpenDX.DXParser.__array, except that the "data follows"
    block is read in one go and parsed by numpy instead of line by line into
    a list of strings (~1 s for an APBS grid). Values are the same floats:
    gridData converts that list with numpy.asarray(..., dtype=float) anyway."""
    consume = self._DXParser__consume
    try:
        tok = consume()
    except dx.DXParserNoTokens:
        return
    if tok.equals("type"):
        tok = consume()
        if not tok.iscode("STRING"):
            raise dx.DXParseError('array: type was "%s", not a string.' % tok.text)
        self.currentobject["type"] = tok.value()
    elif tok.equals("rank"):
        tok = consume()
        try:
            self.currentobject["rank"] = tok.value("INTEGER")
        except ValueError:
            raise dx.DXParseError('array: rank was "%s", not an integer.' % tok.text)
    elif tok.equals("items"):
        tok = consume()
        try:
            self.currentobject["size"] = tok.value("INTEGER")
        except ValueError:
            raise dx.DXParseError('array: items was "%s", not an integer.' % tok.text)
    elif tok.equals("data"):
        tok = consume()
        if not tok.iscode("STRING"):
            raise dx.DXParseError('array: data was "%s", not a string.' % tok.text)
        if tok.text != "follows":
            raise NotImplementedError('array: Only the "data follows header" format is supported.')
        size = self.currentobject["size"]
        if not size:
            raise dx.DXParseError("array: missing number of items")
        # APBS writes 3 values per line; read that many lines at once, then
        # fall back to the original line-by-line loop if any are missing.
        text = " ".join(itertools.islice(self.dxfile, -(-size // 3)))
        values = text.split()
        while len(values) < size:
            line = self.dxfile.readline()
            if not line:
                break
            values.extend(line.strip().split())
        self.currentobject["array"] = np.array(values, dtype=float)
    elif tok.equals("attribute"):
        consume().value()
        if not consume().equals("string"):
            raise dx.DXParseError('array: "string" expected.')
        consume().value()
    else:
        raise dx.DXParseError("array: " + str(tok) + " not recognized.")


import gridData.OpenDX as dx
dx.DXParser._DXParser__array = _dx_array


vg.Grid.reshape = reshape


def _wait_for_apbs(path_done, path_dx):
    """`smiffer -a <cache>` can be started before APBS has finished: <cache>
    must exist (a placeholder). Startup, PQR generation and molecule loading
    happen meanwhile; when <path_done> appears, the APBS .dx is converted
    here, exactly like `volgrids vgtools convert <dx> -f MRC` (which is what
    `volgrids apbs --mrc` does), moved onto <cache>, and read as usual."""
    import volgrids.smiffer as smf
    original = smf.SmifAPBS._apbs_to_smif

    def wait_then_load(grid, path_apbs_in, timer=None):
        while not os.path.exists(path_done):
            time.sleep(0.01)
        if os.path.exists(path_dx):
            tmp = Path(path_dx).with_suffix(".tmp.mrc")
            vg.Grid.load(Path(path_dx)).save(tmp, vg.GridFormat.MRC)
            os.replace(tmp, path_apbs_in)
        return original(grid, path_apbs_in, timer)

    smf.SmifAPBS._apbs_to_smif = staticmethod(wait_then_load)


if os.environ.get("VOLGRIDS_APBS_DONE"):
    _wait_for_apbs(os.environ["VOLGRIDS_APBS_DONE"], os.environ["VOLGRIDS_APBS_DX"])

if __name__ == "__main__":
    vg.PATH_DEFAULT_CONFIG = None   # as for the installed `volgrids` entry point
    vg.AppMain(sys.argv).run()
