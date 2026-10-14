# scripts_fast

Speed-tuned copy of `../scripts`. **Same pockets**: every output of the
original pipeline was checked identical on `1AJU_fixed` (MRC voxel data and
full header except the creation-time label, CSV/TXT byte for byte except
file paths), in sphere, fluid and both modes.

```
python3 run_pipeline_new_spocker.py <input.pdb> <output_dir> [--pocket-mode sphere|fluid|both] [--keep-hotspots] [--keep-intermediate] [--plots]
```

Plots are off by default (`--plots` turns them on), and hotspots are only
copied to `<out>/hotspots` with `--keep-hotspots`. The SLURM wrapper takes the
same options and asks for 8 CPUs.

| 1AJU, wall time       | `../scripts` (bash) | here    |
|-----------------------|---------------------|---------|
| `--pocket-mode sphere`| ~50 s               | ~13.5 s |
| `--pocket-mode both`  | ~57 s (with plots)  | ~16 s   |

About 6.4 s of what is left is APBS itself (pdb2pqr + apbs), which sets the
results and was left alone.

## What changed

- `run_pipeline_new_spocker.py`: one process; Scripts imported once, during
  field generation. Independent stages run as forked processes (see its
  docstring), work dir in `/dev/shm`, MRC writes in the background
  (`_bgwrite.py`), drained between steps.
- `_fields.py`: all volgrids calls in parallel. hphob/stk/hba/hbd don't wait
  for APBS (bit-identical without `-a`); the APBS-field `smiffer` starts at
  once and waits for the APBS grid itself.
- `_volgrids_fast.py`: runs volgrids with an exact, much faster
  `Grid.reshape` (APBS grid -> SMIF grid), a faster DX reader, and the
  wait-for-APBS hook.
- Script1/Script8: matplotlib only imported when plotting.
- Script2: grid totals computed once; component labelling on bounding boxes
  (same labels); no per-label full-grid scans; cached reference headers.
- Script3: distance transform, labelling and marker dilation on bounding
  boxes (exact: the nearest background voxel is always inside box + 1).
- Script4: same tensor-grid interpolation as `_volgrids_fast.py`; only the
  small HBA/HBD masks are dilated (`dilate(ele) & ele == ele`), in their box;
  marker sphere computed in its window.
- Script3b: hotspot centres from integer indices (exact).
- Script2/3/6/7/8: header statistics were computed twice per MRC; once now.
