#!/usr/bin/env bash
set -euo pipefail

# Path to the input PDB file
PDB="../dataset/masif/00-raw_pdbs/1A0G.pdb"

# Resolution tag (the fifth argument passed to compute_surface.py, e.g. 9 -> res_9)
RES=$1

# Root directory for output meshes, including resolution
# e.g. /home/mhg/ForSiyuan/tssplat/mesh_data/amino_acid/res_9
MESH_ROOT="../dataset/masif/01-benchmark_amino_acid_surfaces/res_${RES}"

# Path to the compute_surface.py script (use full path if needed)
SCRIPT="compute_amino_acid_surface.py"

# Loop over amino-acid / residue-id pairs
while read -r AA ID; do
  # Skip empty lines
  [[ -z "$AA" ]] && continue

  echo ">>> Running: $AA $ID A (res_${RES})"
  python -W ignore "$SCRIPT" "$PDB" "$AA" "$ID" A "$RES" "$MESH_ROOT"
done << 'EOF'
ALA 55
ARG 98
ASN 118
ASP 136
CYS 142
GLN 157
GLU 123
GLY 128
HIS 160
ILE 137
LEU 122
LYS 127
MET 199
PHE 133
PRO 121
SER 146
THR 132
TRP 139
TYR 2
VAL 129
EOF
