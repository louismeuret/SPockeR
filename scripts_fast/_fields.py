"""
Wraps the `volgrids` CLI to generate SMIF grids for one structure, using
volgrids 1.0.0's native workflow throughout.

Field file names produced by volgrids (v1.0.0): apbs, stk (stacking),
hphob (hydrophobic), hphil (hydrophilic, unused downstream), hba/hbd
(H-bond acceptor/donor).

CONFIRMED ROOT CAUSE (isolated by direct testing on 2F4U):
Requesting multiple SMIF field types in ONE `smiffer` call (e.g. the
original single call with -a + all SMIF_* left at their true defaults,
only SMIF_HPHIL=false) produces CORRECT apbs/hphob fields but WRONG
stk/hba/hbd fields -- their density collapses onto the same wrong ~8
junction residues (e.g. A/B.1411, 1412, 1488, 1489/1492) regardless of
residue restriction, APBS cache use, or the -r/-a flags. This reproduced
reliably every time multiple fields were requested together.

Running each field type in its OWN separate `smiffer` call (one call per
SMIF_* flag, all others false) was confirmed via ChimeraX to always place
every field -- apbs, hphob, stk, hba, hbd -- correctly across the whole
structure. This is therefore the only currently-verified-correct calling
convention, and is what this module now uses throughout: every field is
computed via its own isolated `smiffer` invocation, never combined.

Non-canonical / non-base-paired residue selection for HBond fields is
still delegated to volgrids' own `smutils res_nobp`. The residue-
restricted (-r) HBond-subset calls also each request ONLY hba, then ONLY
hbd, as separate calls, for the same reason.
"""

import shutil
import importlib.util
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# One entry per field: which SMIF_* flag must be true, and which must be
# forced false alongside it (every other SMIF_* flag). Verified via
# isolated single-field `smiffer` calls to always place fields correctly.
WHOLE_STRUCTURE_FIELD_FLAGS = {
    "apbs":   {"SMIF_APBS": "true",  "SMIF_HBA": "false", "SMIF_HBD": "false", "SMIF_HPHIL": "false", "SMIF_HPHOB": "false", "SMIF_STK": "false"},
    "hphob":  {"SMIF_APBS": "false", "SMIF_HBA": "false", "SMIF_HBD": "false", "SMIF_HPHIL": "false", "SMIF_HPHOB": "true",  "SMIF_STK": "false"},
    "stk":    {"SMIF_APBS": "false", "SMIF_HBA": "false", "SMIF_HBD": "false", "SMIF_HPHIL": "false", "SMIF_HPHOB": "false", "SMIF_STK": "true"},
    "hba":    {"SMIF_APBS": "false", "SMIF_HBA": "true",  "SMIF_HBD": "false", "SMIF_HPHIL": "false", "SMIF_HPHOB": "false", "SMIF_STK": "false"},
    "hbd":    {"SMIF_APBS": "false", "SMIF_HBA": "false", "SMIF_HBD": "true",  "SMIF_HPHIL": "false", "SMIF_HPHOB": "false", "SMIF_STK": "false"},
}

# HBond-subset (-r restricted) calls: same one-field-per-call rule, plus
# SMIF_HB_ONLY_NBASE=true. SMIF_APBS is never requested here (the
# whole-structure apbs field, computed separately above, is reused
# downstream instead).
HBOND_SUBSET_FIELD_FLAGS = {
    "hba": {"SMIF_APBS": "false", "SMIF_HBA": "true",  "SMIF_HBD": "false", "SMIF_HPHIL": "false", "SMIF_HPHOB": "false", "SMIF_STK": "false", "SMIF_HB_ONLY_NBASE": "true"},
    "hbd": {"SMIF_APBS": "false", "SMIF_HBA": "false", "SMIF_HBD": "true",  "SMIF_HPHIL": "false", "SMIF_HPHOB": "false", "SMIF_STK": "false", "SMIF_HB_ONLY_NBASE": "true"},
}

FIELD_NAME_MAP = {
    "apbs": "apbs",
    "stk": "stacking",
    "hphob": "hydrophobic",
    "hba": "hba",
    "hbd": "hbd",
}


class FieldGenerationError(RuntimeError):
    pass


# `volgrids` through _volgrids_fast.py: same CLI and results, faster Grid.reshape.
VOLGRIDS = [sys.executable, str(Path(__file__).resolve().parent / "_volgrids_fast.py")]


def _run(cmd, cwd, env=None):
    if cmd[0] == "volgrids":
        cmd = VOLGRIDS + cmd[1:]
    if env:
        env = {**os.environ, **env}
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise FieldGenerationError(
            f"Command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}"
        )
    return result


def _config_args(overrides: dict) -> list:
    if not overrides:
        return []
    pairs = " ".join(f"{k}={v}" for k, v in overrides.items())
    return ["-c", pairs]


def prepare_workdir(pdb_path: Path, work_dir: Path) -> Path:
    """Copy the input structure into an isolated working directory, since
    volgrids writes its intermediate/output files next to the input file."""
    work_dir.mkdir(parents=True, exist_ok=True)
    local_path = work_dir / pdb_path.name
    shutil.copy(pdb_path, local_path)
    return local_path


def compute_non_canonical_indices(local_pdb: Path) -> str:
    """Run volgrids' own `smutils res_nobp` to get the non-base-paired
    residue index string (space-separated "chain.resid" tokens). Returns
    "" if none are found."""
    result = _run(["volgrids", "smutils", "res_nobp", local_pdb.name], cwd=local_pdb.parent)
    return result.stdout.strip()


def _smiffer_one(local_pdb: Path, field: str, flags: dict, out_dir: Path, extra: list, env=None) -> Path:
    field_out = out_dir / field
    field_out.mkdir(parents=True, exist_ok=True)
    cmd = ["volgrids", "smiffer", local_pdb.name, *extra, "-o", str(field_out)] + _config_args(flags)
    _run(cmd, cwd=local_pdb.parent, env=env)
    return field_out / f"{local_pdb.stem}.{field}.mrc"


# volgrids runs in-process with this interpreter (VOLGRIDS above), not from PATH.
_spec = importlib.util.find_spec("volgrids")
if _spec is None:
    sys.exit(f"volgrids is not installed for {sys.executable}: run with the "
             "SPockeR conda env's python (`conda activate SPockeR`).")
APBS_SH = Path(_spec.submodule_search_locations[0]) / "apbs" / "apbs.sh"


def compute_apbs_dx(local_pdb: Path, done: Path) -> Path:
    """`volgrids apbs <pdb> --mrc` is `apbs.sh <pdb> --mrc`. Here apbs.sh
    runs without --mrc; its last step (DX -> MRC conversion) is done by the
    waiting apbs-field smiffer process (see _volgrids_fast.py). `done` is
    created even on failure, so that process stops waiting (and fails on the
    empty placeholder cache)."""
    try:
        _run(["bash", str(APBS_SH), local_pdb.name], cwd=local_pdb.parent)
        return local_pdb.parent / f"{local_pdb.name}.dx"
    finally:
        done.touch()


def compute_all_fields(local_pdb: Path, whole_dir: Path, hbond_dir: Path):
    """Every volgrids call needed for one structure (APBS, one smiffer call
    per whole-structure field, res_nobp, one smiffer call per HBond-subset
    field), run in parallel. Only the apbs field reads the APBS cache: the hphob/stk/hba/hbd
    smiffer calls give bit-identical grids without -a (checked on 1AJU), so
    they don't wait for APBS. The apbs-field smiffer call starts right away
    too, and only waits for the APBS grid when it actually reads it
    (VOLGRIDS_APBS_DONE, see _volgrids_fast.py).
    Returns (whole-structure paths, indices string, hbond-subset paths)."""
    cache = local_pdb.parent / f"{local_pdb.name}.mrc"
    done = local_pdb.parent / "apbs.done"
    cache.touch()  # placeholder: smiffer checks that the -a path exists
    with ThreadPoolExecutor(max_workers=10) as pool:
        apbs = pool.submit(compute_apbs_dx, local_pdb, done)
        indices = pool.submit(compute_non_canonical_indices, local_pdb)
        whole = {f: pool.submit(_smiffer_one, local_pdb, f, flags, whole_dir, [])
                 for f, flags in WHOLE_STRUCTURE_FIELD_FLAGS.items() if f != "apbs"}
        whole["apbs"] = pool.submit(_smiffer_one, local_pdb, "apbs", WHOLE_STRUCTURE_FIELD_FLAGS["apbs"],
                                    whole_dir, ["-a", str(cache)],
                                    {"VOLGRIDS_APBS_DONE": str(done),
                                     "VOLGRIDS_APBS_DX": str(local_pdb.parent / f"{local_pdb.name}.dx")})
        idx = indices.result()
        hb = ({f: pool.submit(_smiffer_one, local_pdb, f, flags, hbond_dir, ["-r", idx])
               for f, flags in HBOND_SUBSET_FIELD_FLAGS.items()} if idx else {})
        apbs.result()
        whole_paths = {FIELD_NAME_MAP[f]: p for f, fut in whole.items() if (p := fut.result()).exists()}
        hb_paths = {FIELD_NAME_MAP[f]: p for f, fut in hb.items() if (p := fut.result()).exists()}
    return whole_paths, idx, hb_paths
