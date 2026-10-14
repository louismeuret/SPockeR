#!/bin/bash
# SLURM job-array wrapper around run_pipeline_new_spocker.py: one array task
# per PDB, each with its own SPOCKER_WORK_DIR so tasks don't clobber each other.
#
# Usage (run on the login node, NOT via sbatch -- it submits itself):
#   bash run_pipeline_new_spocker_slurm.sh <pdb_dir_or_list.txt> <output_dir> [options...] [-- sbatch options...]
# e.g.
#   bash run_pipeline_new_spocker_slurm.sh ../data/pdbs ../results --pocket-mode both -- --partition=short
#   bash run_pipeline_new_spocker_slurm.sh .../DB_RNA/pdb_structures ../results_db --pattern '*_nucleic_only.pdb'
#   bash run_pipeline_new_spocker_slurm.sh .../DB_RNA_v3 ../results_v3 --compressed --pattern '*_m1_nucleic_only.pdb'
#
# <pdb_dir_or_list.txt>: a directory (searched recursively for --pattern) or a
#   text file with one PDB path per line.
# Wrapper options:
#   --pattern GLOB   file name pattern searched in the directory (default '*.pdb')
#   --compressed     inputs are .pdb.gz (".gz" is appended to --pattern); each
#                    task gunzips its PDB to scratch. Same as COMPRESSED=true.
#   --max-jobs N     max tasks running at the same time (default 50, MAX_JOBS=N)
# Any other option is passed as-is to run_pipeline_new_spocker.py.
# Each PDB is cleaned with 0_fix_pdb.sh before the pipeline runs.
# Results land in <output_dir>/<pdb_id>/, logs in <output_dir>/logs/.
# sbatch options after "--" override the #SBATCH defaults below.
#SBATCH --job-name=spocker
#SBATCH --cpus-per-task=8   # field generation and hotspot stages run in parallel
#SBATCH --mem=8G
#SBATCH --time=04:00:00
set -euo pipefail

# ---------------------------------------------------------------- array task
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # Load your environment here if the compute nodes need it, e.g.:
    # module load python ; source activate spocker
    PDB=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SPOCKER_LIST")
    PDB_ID=$(basename "$PDB" .gz); PDB_ID="${PDB_ID%.*}"
    # Prefer node-local scratch when the cluster provides it.
    export SPOCKER_WORK_DIR="${TMPDIR:-$SPOCKER_OUT/work}/spocker_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    echo ">>> Task $SLURM_ARRAY_TASK_ID on $(hostname): $PDB"
    # Outside SPOCKER_WORK_DIR: the pipeline wipes that dir on start.
    IN_DIR="${SPOCKER_WORK_DIR}_input"
    mkdir -p "$IN_DIR"
    trap 'rm -rf "$IN_DIR"' EXIT
    if [[ "$PDB" == *.gz ]]; then
        gunzip -c "$PDB" > "$IN_DIR/raw.pdb"
        PDB="$IN_DIR/raw.pdb"
    fi
    # Clean first (model 1, no HETATM, pdbfixer); keep the stem as PDB id.
    bash "$SPOCKER_DIR/0_fix_pdb.sh" "$PDB" "$IN_DIR/$PDB_ID.pdb"
    python3 "$SPOCKER_DIR/run_pipeline_new_spocker.py" "$IN_DIR/$PDB_ID.pdb" "$SPOCKER_OUT/$PDB_ID" $SPOCKER_ARGS
    exit
fi

# ---------------------------------------------------------------- submission
USAGE="Usage: $0 <pdb_dir_or_list.txt> <output_dir> [--pattern GLOB] [--compressed] [--max-jobs N] [pipeline options...] [-- sbatch options...]"
[[ $# -lt 2 ]] && { echo "$USAGE"; exit 1; }
INPUT="$1"
OUT_DIR=$(realpath -m "$2")
shift 2
COMPRESSED="${COMPRESSED:-false}"
MAX_JOBS="${MAX_JOBS:-50}"
PATTERN='*.pdb'
PIPE_ARGS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
    case "$1" in
        --compressed) COMPRESSED=true ;;
        --pattern) PATTERN="$2"; shift ;;
        --max-jobs) MAX_JOBS="$2"; shift ;;
        *) PIPE_ARGS+=("$1") ;;
    esac
    shift
done
[[ "${1:-}" == "--" ]] && shift
[[ "$COMPRESSED" == true ]] && PATTERN="$PATTERN.gz"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT_DIR/logs"
# One list per submission: pending tasks of an earlier array still read theirs.
LIST="$OUT_DIR/pdb_list_$(date +%Y%m%d_%H%M%S).txt"
if [[ -d "$INPUT" ]]; then
    find "$(realpath "$INPUT")" -name "$PATTERN" | sort > "$LIST"
elif [[ -f "$INPUT" ]]; then
    grep -v '^\s*$' "$INPUT" | xargs -d '\n' realpath > "$LIST"
else
    echo "!!! $INPUT is neither a directory nor a file"; exit 1
fi
N=$(wc -l < "$LIST")
[[ "$N" -eq 0 ]] && { echo "!!! No file matching '$PATTERN' in $INPUT"; exit 1; }

# Each array task names its output dir after the PDB stem: duplicates would collide.
DUPES=$(xargs -d '\n' -n1 basename < "$LIST" | sed 's/\.gz$//; s/\.[^.]*$//' | sort | uniq -d)
[[ -n "$DUPES" ]] && { echo "!!! Duplicate PDB ids: $DUPES"; exit 1; }

echo ">>> Submitting $N PDB(s) from $INPUT -> $OUT_DIR (max $MAX_JOBS at a time)"
sbatch --array="0-$((N - 1))%${MAX_JOBS}" \
    --output="$OUT_DIR/logs/%x_%A_%a.out" \
    --export=ALL,SPOCKER_DIR="$SCRIPT_DIR",SPOCKER_LIST="$LIST",SPOCKER_OUT="$OUT_DIR",SPOCKER_ARGS="${PIPE_ARGS[*]:-}" \
    "$@" "$0"
