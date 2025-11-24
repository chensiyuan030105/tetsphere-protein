#!/bin/bash

LIST_FILE="../dataset/masif/lists/full_list.txt"
RAW_DIR="../dataset/masif/00-raw_pdbs"
PDB_OUT_DIR="../dataset/masif/01-benchmark_pdbs"
SURFACE_IN_DIR="../dataset/masif/01-benchmark_pdbs_surfaces"
SURFACE_OUT_DIR="../dataset/masif/01-benchmark_pdbs_surfaces_split"

RES=$1

if [ -z "$RES" ]; then
    echo "Usage: ./run_surface.sh <resolution>"
    exit 1
fi

echo "[INFO] Using resolution: $RES"
echo "[INFO] Reading NAME list from: $LIST_FILE"

# ------------------------------------------------------
# Count number of non-empty lines to know total progress
# ------------------------------------------------------
TOTAL_LINES=$(grep -cvE '^[[:space:]]*$' "$LIST_FILE")
if [ "$TOTAL_LINES" -eq 0 ]; then
    echo "[ERROR] LIST_FILE seems to be empty: $LIST_FILE"
    exit 1
fi

# Simple function to print a progress bar
print_progress() {
    local current=$1
    local total=$2
    local width=40  # bar width in characters

    local percent=$(( 100 * current / total ))
    local filled=$(( width * current / total ))
    local empty=$(( width - filled ))

    # Build the bar string
    local bar_filled
    local bar_empty
    bar_filled=$(printf "%${filled}s" | tr ' ' '#')
    bar_empty=$(printf "%${empty}s" | tr ' ' '-')

    # Print progress bar + percentage + counts
    printf "[%s%s] %3d%% (%d / %d)\n" "$bar_filled" "$bar_empty" "$percent" "$current" "$total"
}

CURRENT_LINE=0

# Use IFS + read -r to preserve spaces and avoid backslash escapes
while IFS= read -r LINE; do
    # Skip empty lines
    if [[ -z "$LINE" ]]; then
        continue
    fi

    CURRENT_LINE=$((CURRENT_LINE + 1))

    # Split fields
    PDB_ID=$(echo "$LINE" | cut -d"_" -f1)
    CHAIN1=$(echo "$LINE" | cut -d"_" -f2)
    CHAIN2=$(echo "$LINE" | cut -d"_" -f3)

    # Build two names
    NAME1="${PDB_ID}_${CHAIN1}"
    NAME2="${PDB_ID}_${CHAIN2}"

    # Raw pdb path only needs PDB_ID
    PDB_PATH="${RAW_DIR}/${PDB_ID}.pdb"

    # Expected chain PDB paths
    CHAIN_PDB1="${PDB_OUT_DIR}/${NAME1}.pdb"
    CHAIN_PDB2="${PDB_OUT_DIR}/${NAME2}.pdb"
    
    # Expected PLY output paths
    IN_PLY1="${SURFACE_IN_DIR}/res_${RES}/${NAME1}.ply"
    IN_PLY2="${SURFACE_IN_DIR}/res_${RES}/${NAME2}.ply"
    OUT_PLY1="${SURFACE_OUT_DIR}/res_${RES}/${NAME1}.ply"
    OUT_PLY2="${SURFACE_OUT_DIR}/res_${RES}/${NAME2}.ply"

    if [ ! -f "$PDB_PATH" ]; then
        echo "[WARN] Missing PDB: $PDB_PATH"
        # Still count this line in progress
        print_progress "$CURRENT_LINE" "$TOTAL_LINES"
        continue
    fi

    echo "------------------------------------------------------"
    echo "[INFO] Processing two chains from: $LINE"
    echo "[INFO] Chains: $NAME1 and $NAME2"
    echo "------------------------------------------------------"

    # Print progress bar for this line
    print_progress "$CURRENT_LINE" "$TOTAL_LINES"

    # -------- Chain 1 --------
    # Skip only if BOTH extracted_chain_pdb and PLY already exist
    if [ -f "$CHAIN_PDB1" ] && [ -f "$OUT_PLY1" ]; then
        echo "[INFO] Chain 1 already done (PDB + PLY exist), skipping: $NAME1"
    else
        python compute_protein_surface_split.py "$PDB_PATH" "$NAME1" "$RES" "$PDB_OUT_DIR" "$SURFACE_IN_DIR" "$SURFACE_OUT_DIR"
    fi

    # -------- Chain 2 --------
    if [ -f "$CHAIN_PDB2" ] && [ -f "$OUT_PLY2" ]; then
        echo "[INFO] Chain 2 already done (PDB + PLY exist), skipping: $NAME2"
    else
        python compute_protein_surface_split.py "$PDB_PATH" "$NAME2" "$RES" "$PDB_OUT_DIR" "$SURFACE_IN_DIR" "$SURFACE_OUT_DIR"
    fi

done < "$LIST_FILE"

echo "[INFO] Finished batch processing."


