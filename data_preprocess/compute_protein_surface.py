#!/usr/bin/python
import numpy as np
import os
import shutil
import sys
from Bio.PDB import *
from IPython.core.debugger import set_trace

import pymesh

# Local includes
from triangulation.computeMSMS import computeMSMS
from input_output.save_ply import save_ply
from input_output.protonate import protonate
from input_output.extractPDB import extractPDB   # <-- ensure this exists!


# ============================
# Main
# ============================
if len(sys.argv) != 6:
    print("Usage:")
    print("  python compute_protein_surface.py <input_pdb> <name> <resolution> <pdb_out_dir> <surface_out_dir>")
    sys.exit(1)


# ---------- Parse arguments ----------
input_pdb       = sys.argv[1]     # path/to/PDB_ID.pdb (full molecule)
name            = sys.argv[2]     # e.g., 1A0G_A
resolution      = sys.argv[3]     # e.g., 10
pdb_out_dir     = sys.argv[4]     # directory for chain PDBs
surface_out_dir = sys.argv[5]     # directory for surface meshes (PLY)


# Extract PDB ID and CHAIN ID from NAME
pdb_id   = name.split("_")[0]      # e.g., "1A0G"
chain_id = name.split("_")[1]      # e.g., "A"


# ---------- Set up output directory ----------
os.makedirs(pdb_out_dir, exist_ok=True)


# ---------- Prepare PLY output path & early exit ----------
ply_out_dir = os.path.join(surface_out_dir, f"res_{resolution}")
os.makedirs(ply_out_dir, exist_ok=True)
ply_out = os.path.join(ply_out_dir, name + ".ply")

# If the PLY already exists, skip all processing
if os.path.exists(ply_out):
    print(f"[INFO] PLY already exists, skipping: {ply_out}")
    sys.exit(0)


# ---------- Step 1 & 2: protonate + extract chain (can be skipped) ----------
tmp_dir = "../tmp"
os.makedirs(tmp_dir, exist_ok=True)

protonated_pdb = os.path.join(tmp_dir, f"{pdb_id}.pdb")
extracted_chain_pdb = os.path.join(pdb_out_dir, f"{name}.pdb")

if os.path.exists(extracted_chain_pdb):
    # If the extracted chain PDB already exists, assume previous preprocessing is done
    print(f"[INFO] Extracted chain PDB already exists, skipping protonation & extraction: {extracted_chain_pdb}")
else:
    # Step 1: protonate original full-molecule PDB
    print(f"[INFO] Protonating PDB: {input_pdb} -> {protonated_pdb}")
    protonate(input_pdb, protonated_pdb)

    # Step 2: extract the requested chain to pdb_out_dir
    print(f"[INFO] Extracting chain {chain_id} → {extracted_chain_pdb}")
    extractPDB(protonated_pdb, extracted_chain_pdb, chain_id)


# ---------- Step 3: compute MSMS on extracted chain ----------
print(f"[INFO] Computing MSMS surface for extracted chain: {extracted_chain_pdb}")

try:
    vertices, faces, normals, names, areas = computeMSMS(
        extracted_chain_pdb,
        protonate=True,             # Set to False if computeMSMS expects a non-protonated input
        resolution=str(resolution)
    )
    print(f"[INFO] MSMS computed: vertices={len(vertices)}, faces={len(faces)}")
except Exception as e:
    print(f"[ERROR] MSMS failed for {extracted_chain_pdb}")
    print(e)
    set_trace()


# ---------- Step 4: save PLY ----------
print(f"[INFO] Saving mesh → {ply_out}")
save_ply(ply_out, vertices, faces)

print("[INFO] Done.")
