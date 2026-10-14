#!/usr/bin/env python3
"""
Single-process version of run_pipeline_new_spocker.sh: same steps, same
options, same outputs, but every Script is imported once and its main() is
called in this process, so numpy/scipy/mrcfile/volgrids/... are loaded once
instead of once per step.

Usage: python3 run_pipeline_new_spocker.py <input.pdb> <output_dir> [--keep-intermediate] [--no-plots] [--pocket-mode sphere|fluid|both]
SPOCKER_WORK_DIR and SPOCKER_PROFILE_DIR work as in the bash version.
"""

import argparse
import importlib
import os
import shutil
import sys
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
MODULES = {name: importlib.import_module(mod) for name, mod in STEPS.items()}

PROFILE_DIR = os.environ.get("SPOCKER_PROFILE_DIR")
if PROFILE_DIR:
    from pyinstrument import Profiler


class StepFailed(Exception):
    pass


def run_step(name, *args, report=None):
    """Call <step>.main() with sys.argv set as if it had been run by python3."""
    print(f">>> Running {report or name}", flush=True)
    module = MODULES[name]
    sys.argv = [module.__file__, *map(str, args)]
    profiler = Profiler() if PROFILE_DIR else None
    if profiler:
        profiler.start()
    try:
        module.main()
    except SystemExit as e:
        if e.code not in (None, 0):
            if not isinstance(e.code, int):
                print(e.code, file=sys.stderr)
            raise StepFailed(name) from e
    finally:
        if profiler:
            profiler.stop()
            Path(PROFILE_DIR).mkdir(parents=True, exist_ok=True)
            profiler.write_html(Path(PROFILE_DIR) / f"{report or name}.html")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pdb")
    ap.add_argument("output_dir")
    ap.add_argument("--keep-intermediate", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--pocket-mode", choices=("sphere", "fluid", "both"), default="sphere")
    a = ap.parse_args()

    pdb_path = Path(a.pdb).resolve()
    out_dir = Path(a.output_dir).resolve()
    pdb_id = pdb_path.stem
    plot_args = ["--no-plots"] if a.no_plots else []

    work = Path(os.environ.get("SPOCKER_WORK_DIR") or SCRIPT_DIR / "testdata" / "new_spocker_work")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    fields1 = work / f"Fields_Pipeline1_{pdb_id}"
    fields2 = work / f"Fields_Pipeline2_{pdb_id}"
    analysis1 = work / f"Analysis_Pipeline1_{pdb_id}"
    analysis2 = work / f"Analysis_Pipeline2_{pdb_id}"

    def score(dest, mode):
        dest.mkdir(parents=True, exist_ok=True)
        print(f">>> Merging pockets and scoring ({mode} pockets)")
        try:  # like `|| true` in the bash version: only the CSV decides success
            run_step("Script8", "--analysis1_dir", analysis1, "--analysis2_dir", analysis2,
                     "--fields_dir", fields1, "--pdb_file", pdb_path, "--pdb_id", pdb_id,
                     "--output_dir", dest, *plot_args, report=f"Script8_{mode}")
        except Exception as e:
            print(f"!!! Script8 ({mode}) failed: {e!r}")
        if not (dest / f"{pdb_id}_field_contributions.csv").is_file():
            print(f"!!! Script8 ({mode}) produced no field-contributions CSV in {dest}")
            return False
        return True

    def fluid(dest):
        try:
            run_step("Script3b", "--pdb_file", pdb_path, "--pdb_id", pdb_id, "--fields_dir", fields1,
                     "--analysis1_dir", analysis1, "--analysis2_dir", analysis2,
                     "--output_dir", dest, "--pipeline_names")
            return True
        except Exception as e:
            print(f"!!! Script3b failed: {e!r}")
            return False

    try:
        print(f">>> [1/3] Generating fields (APBS, stacking, hydrophobic; HBond fields if applicable) for {pdb_id}")
        run_step("Prepare", pdb_path, work, pdb_id)
        if not any(fields1.glob("*.mrc")):
            print(f"!!! Field generation produced no .mrc files in {fields1}")
            return 1

        print(">>> [2/3] Running the Pipeline 1 hotspot / pocket-volume stages")
        run_step("Script1", "--input_dir", fields1, "--pdb_id", pdb_id, "--output_dir", analysis1, *plot_args)
        run_step("Script2", "--fields_dir", fields1, "--pdb_file", pdb_path, "--analysis_dir", analysis1, "--pdb_id", pdb_id)
        run_step("Script3", "--pdb_file", pdb_path, "--analysis_dir", analysis1, "--pdb_id", pdb_id)
        run_step("Script6", "--fields_dir", fields1, "--pdb_file", pdb_path, "--pdb_id", pdb_id)
        run_step("Script7", "--fields_dir", fields1, "--pdb_id", pdb_id)

        if (fields2 / f"{pdb_id}.apbs.mrc").is_file():
            print(">>> [3/3] Running the Pipeline 2 H-bond pocket stages")
            run_step("Script4", "--fields_dir", fields2, "--analysis_dir", analysis2, "--pdb_id", pdb_id)
            run_step("Script5", "--pdb_file", pdb_path, "--analysis_dir", analysis2, "--pdb_id", pdb_id)
        else:
            print(">>> No non-canonical residues / HBond fields; skipping Script4/5")

        # Keep the hotspots before Script3b overwrites the pocket files.
        for d in (analysis1, analysis2):
            if d.is_dir():
                shutil.copytree(d, out_dir / "hotspots" / d.name, dirs_exist_ok=True)

        if a.pocket_mode == "sphere":
            ok = score(out_dir, "sphere")
        elif a.pocket_mode == "fluid":
            ok = fluid(out_dir / "hotspots" / "fluid_pockets") and score(out_dir, "fluid")
        else:
            ok = score(out_dir / "sphere", "sphere")
            if fluid(out_dir / "fluid" / "hotspot_pockets"):
                ok = score(out_dir / "fluid", "fluid") and ok
            else:
                print("!!! Script3b failed; no fluid scoring")
                ok = False
        return 0 if ok else 1
    finally:
        if a.keep_intermediate:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(out_dir / "intermediate", ignore_errors=True)
            shutil.move(str(work), out_dir / "intermediate")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
