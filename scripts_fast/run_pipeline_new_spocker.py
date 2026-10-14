#!/usr/bin/env python3
"""
Fast SPOCKER runner: same pockets as ../scripts/run_pipeline_new_spocker.sh,
in as little wall time as possible.

- Scripts are imported once; independent stages run as forked processes
  (fork keeps the imports), so each branch only waits for what it needs:
      fields --+-- Script1 -> Script2 -> Script3 --+
               +-- Script6, Script7 ---------------+-- Script8 (sphere)
               +-- Script4 -> Script5 -------------+-- Script3b -> Script8 (fluid)
- All volgrids calls of the field generation run in parallel (_fields.py).
- The work dir is in RAM (/dev/shm) unless SPOCKER_WORK_DIR says otherwise.
- With --pocket-mode both, sphere and fluid scoring run in parallel (fluid
  works on its own copy of the analysis dirs, since Script3b overwrites them).
- Hotspots are only copied to <out>/hotspots with --keep-hotspots.

Usage: python3 run_pipeline_new_spocker.py <input.pdb> <output_dir> [--keep-intermediate] [--keep-hotspots] [--plots] [--pocket-mode sphere|fluid|both]
Plots are off by default here (--plots turns them on).
SPOCKER_PROFILE_DIR=<dir> writes one pyinstrument report per step.
"""

import argparse
import importlib
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

STEPS = {
    "Prepare": "_new_spocker_prepare_fields",
    "Script1": "Script1_Pipeline1_Slope_Derived_Fixed_Iso_Values_for_Hotspot",
    "Script2": "Script2_Pipeline1_Detection_of_Binding_Site_Hotspots",
    "Script3": "Script3_Pipeline1_Making_Pocket_Volume_Using_Hotspots",
    "Script3b": "Script3b_Fluid_Pocket_Volume_From_Hotspots_v3",
    "Script4": "Script4_Pipeline2_Hydrogen_Bond_Pocket_Hotspots_Using_HBA_HBD_ELE_Fields",
    "Script5": "Script5_Pipeline2_Making_Hydrogen_Bond_Pocket_Volume",
    "Script6": "Script6_Trimming_APBS_for_Scoring_Unique_Pockets",
    "Script7": "Script7_Trimming_Hydrophobic_for_Scoring_Unique_Pockets",
    "Script8": "Script8_Making_Unique_Pockets_Using_All_Previous_Pockets",
}
PROFILE_DIR = os.environ.get("SPOCKER_PROFILE_DIR")
FORK = mp.get_context("fork")
T0 = time.perf_counter()


def mark(what):
    print(f"--- {time.perf_counter() - T0:5.1f}s {what}", flush=True)


class StepFailed(Exception):
    pass


def run_step(name, *args, report=None):
    """Call <step>.main() with sys.argv set as if it had been run by python3."""
    report = report or name
    print(f">>> Running {report}", flush=True)
    module = MODULES[name]
    sys.argv = [module.__file__, *map(str, args)]
    t0 = time.perf_counter()
    profiler = None
    if PROFILE_DIR:
        from pyinstrument import Profiler
        profiler = Profiler()
        profiler.start()
    try:
        try:
            module.main()
        finally:
            _bgwrite.drain()  # the next step may read what this one wrote
    except SystemExit as e:
        if e.code not in (None, 0):
            if not isinstance(e.code, int):
                print(e.code, file=sys.stderr)
            raise StepFailed(report) from e
    finally:
        if profiler:
            profiler.stop()
            Path(PROFILE_DIR).mkdir(parents=True, exist_ok=True)
            profiler.write_html(Path(PROFILE_DIR) / f"{report}.html")
        print(f"<<< {report} {time.perf_counter() - t0:.1f}s", flush=True)


def _branch(func, args):
    try:
        ok = func(*args)
    except Exception as e:
        print(f"!!! {e!r}", file=sys.stderr, flush=True)
        ok = False
    sys.stdout.flush()
    os._exit(0 if ok is not False else 1)


def parallel(*jobs):
    """Run each (func, *args) in a forked child; True if all of them succeeded."""
    procs = [FORK.Process(target=_branch, args=(job[0], job[1:])) for job in jobs]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    return all(p.exitcode == 0 for p in procs)


def steps(*calls):
    for name, *args in calls:
        run_step(name, *args)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pdb")
    ap.add_argument("output_dir")
    ap.add_argument("--keep-intermediate", action="store_true")
    ap.add_argument("--keep-hotspots", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--no-plots", action="store_true", help=argparse.SUPPRESS)  # default here; kept for old command lines
    ap.add_argument("--pocket-mode", choices=("sphere", "fluid", "both"), default="sphere")
    a = ap.parse_args()

    pdb_path = Path(a.pdb).resolve()
    out_dir = Path(a.output_dir).resolve()
    pdb_id = pdb_path.stem
    plot_args = [] if a.plots else ["--no-plots"]

    if os.environ.get("SPOCKER_WORK_DIR"):
        work = Path(os.environ["SPOCKER_WORK_DIR"])
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
    else:
        shm = "/dev/shm" if os.access("/dev/shm", os.W_OK) else None
        work = Path(tempfile.mkdtemp(prefix="spocker_", dir=shm))

    fields1 = work / f"Fields_Pipeline1_{pdb_id}"
    fields2 = work / f"Fields_Pipeline2_{pdb_id}"
    analysis1 = work / f"Analysis_Pipeline1_{pdb_id}"
    analysis2 = work / f"Analysis_Pipeline2_{pdb_id}"

    def score(dest, mode, a1, a2):
        dest.mkdir(parents=True, exist_ok=True)
        try:  # like `|| true` in the bash version: only the CSV decides success
            run_step("Script8", "--analysis1_dir", a1, "--analysis2_dir", a2,
                     "--fields_dir", fields1, "--pdb_file", pdb_path, "--pdb_id", pdb_id,
                     "--output_dir", dest, *plot_args, report=f"Script8_{mode}")
        except Exception as e:
            print(f"!!! Script8 ({mode}) failed: {e!r}")
        if not (dest / f"{pdb_id}_field_contributions.csv").is_file():
            print(f"!!! Script8 ({mode}) produced no field-contributions CSV in {dest}")
            return False
        return True

    def fluid(hotspot_dest, score_dest, a1, a2):
        try:
            run_step("Script3b", "--pdb_file", pdb_path, "--pdb_id", pdb_id, "--fields_dir", fields1,
                     "--analysis1_dir", a1, "--analysis2_dir", a2,
                     "--output_dir", hotspot_dest, "--pipeline_names")
        except Exception as e:
            print(f"!!! Script3b failed; no fluid scoring: {e!r}")
            return False
        return score(score_dest, "fluid", a1, a2)

    try:
        # Field generation is all subprocesses: import the Scripts meanwhile.
        with ThreadPoolExecutor(1) as pool:
            prepare = pool.submit(run_step, "Prepare", pdb_path, work, pdb_id)
            for name, mod in STEPS.items():
                MODULES.setdefault(name, importlib.import_module(mod))
            mark("scripts imported")
            prepare.result()
        mark("fields done")
        if not any(fields1.glob("*.mrc")):
            print(f"!!! Field generation produced no .mrc files in {fields1}")
            return 1

        jobs = [
            (steps, ("Script1", "--input_dir", fields1, "--pdb_id", pdb_id, "--output_dir", analysis1, *plot_args),
                    ("Script2", "--fields_dir", fields1, "--pdb_file", pdb_path, "--analysis_dir", analysis1, "--pdb_id", pdb_id),
                    ("Script3", "--pdb_file", pdb_path, "--analysis_dir", analysis1, "--pdb_id", pdb_id)),
            (steps, ("Script6", "--fields_dir", fields1, "--pdb_file", pdb_path, "--pdb_id", pdb_id),
                    ("Script7", "--fields_dir", fields1, "--pdb_id", pdb_id)),
        ]
        if (fields2 / f"{pdb_id}.apbs.mrc").is_file():
            jobs.append((steps, ("Script4", "--fields_dir", fields2, "--analysis_dir", analysis2, "--pdb_id", pdb_id),
                                ("Script5", "--pdb_file", pdb_path, "--analysis_dir", analysis2, "--pdb_id", pdb_id)))
        else:
            print(">>> No non-canonical residues / HBond fields; skipping Script4/5")
        if not parallel(*jobs):
            print("!!! A hotspot stage failed")
            return 1
        mark("hotspots done")

        copier = None
        if a.keep_hotspots:  # copy to the (possibly slow) output disk while scoring runs
            def copy_hotspots():
                for d in (analysis1, analysis2):
                    if d.is_dir():
                        shutil.copytree(d, out_dir / "hotspots" / d.name, dirs_exist_ok=True)
            copier = FORK.Process(target=_branch, args=(copy_hotspots, ()))
            copier.start()

        if a.pocket_mode == "sphere":
            ok = score(out_dir, "sphere", analysis1, analysis2)
        elif a.pocket_mode == "fluid":
            ok = fluid(out_dir / "hotspots" / "fluid_pockets", out_dir, analysis1, analysis2)
        else:
            # Script3b overwrites the pocket files in place: give it copies.
            fluid1, fluid2 = work / "fluid" / analysis1.name, work / "fluid" / analysis2.name
            for src, dst in ((analysis1, fluid1), (analysis2, fluid2)):
                if src.is_dir():
                    shutil.copytree(src, dst)
            ok = parallel((score, out_dir / "sphere", "sphere", analysis1, analysis2),
                          (fluid, out_dir / "fluid" / "hotspot_pockets", out_dir / "fluid", fluid1, fluid2))
        if copier is not None:
            copier.join()
            ok = ok and copier.exitcode == 0
        mark("pockets done")
        return 0 if ok else 1
    finally:
        if a.keep_intermediate:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(out_dir / "intermediate", ignore_errors=True)
            shutil.move(str(work), out_dir / "intermediate")
        else:
            shutil.rmtree(work, ignore_errors=True)
        mark("cleanup done")


MODULES = {"Prepare": importlib.import_module(STEPS["Prepare"])}
import _bgwrite  # noqa: E402  (needs SCRIPT_DIR on sys.path)

if __name__ == "__main__":
    sys.exit(main())
