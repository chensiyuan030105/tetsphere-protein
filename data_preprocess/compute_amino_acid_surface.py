#!/usr/bin/python
import numpy as np
import os
import Bio
import shutil
from Bio.PDB import * 
import sys
import importlib
from IPython.core.debugger import set_trace
import pymesh

# Local includes
from triangulation.computeMSMS import computeMSMS
from triangulation.fixmesh import fix_mesh
from input_output.save_ply import save_ply
from input_output.protonate import protonate


class ResidueSelect(Select):
    """Select a single residue by (resname, resid, chain_id)."""

    def __init__(self, resname, resid, chain_id):
        super().__init__()
        self.resname = resname
        self.resid = resid
        self.chain_id = chain_id

    def accept_residue(self, residue):
        return (
            residue.get_resname() == self.resname
            and residue.get_id()[1] == self.resid
            and residue.get_parent().id == self.chain_id
        )


def main():
    if len(sys.argv) != 7:
        print(
            "Usage:\n"
            "  python compute_surface.py <pdb_path> <RESNAME> <RESID> <CHAIN_ID> <RESOLUTION> <MESH_ROOT>\n\n"
            "Example:\n"
            "  python compute_surface.py 1A0G.pdb ALA 55 A 10 /home/user/mesh_data/amino_acid/res_10"
        )
        sys.exit(1)

    pdb_path = sys.argv[1]
    resname = sys.argv[2].upper()
    resid = int(sys.argv[3])
    chain_id = sys.argv[4]
    resolution = sys.argv[5]       # passed to MSMS
    mesh_root = sys.argv[6]        # base directory for this resolution, e.g. /.../res_10

    print(f"[INFO] Input PDB       : {pdb_path}")
    print(f"[INFO] Target residue  : {resname} {resid} {chain_id}")
    print(f"[INFO] Output mesh root: {mesh_root}")

    # ---------------------------
    # Step 1: Extract residue PDB
    # ---------------------------
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)

    # Save the single–residue PDB to:
    #   <pdb_root>/01-benchmark_pdbs/AminoAcid/<RESNAME>.pdb
    base_dp_dir = os.path.dirname(os.path.dirname(os.path.abspath(pdb_path)))
    amino_pdb_dir = os.path.join(base_dp_dir, "01-benchmark_pdbs", "AminoAcid")
    os.makedirs(amino_pdb_dir, exist_ok=True)

    out_pdb = os.path.join(amino_pdb_dir, f"{resname}.pdb")
    print(f"[INFO] Saving selected residue to: {out_pdb}")

    selector = ResidueSelect(resname, resid, chain_id)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb, selector)

    # Base name without extension for MSMS
    out_filename1 = os.path.splitext(out_pdb)[0]

    # ---------------------------
    # Step 2: Compute MSMS
    # ---------------------------
    # Output mesh:
    #   <mesh_root>/<RESNAME>/<RESNAME>.ply
    aa_mesh_dir = os.path.join(mesh_root)
    os.makedirs(aa_mesh_dir, exist_ok=True)

    out_filename2 = os.path.join(aa_mesh_dir, resname)

    print(f"[INFO] Computing MSMS surface for: {out_filename1}.pdb")
    try:
        vertices1, faces1, normals1, names1, areas1 = computeMSMS(
            out_filename1 + ".pdb", protonate=True, resolution=resolution
        )
        print(f"[INFO] MSMS computed: vertices={len(vertices1)}, faces={len(faces1)}")
    except Exception as e:
        print(f"[ERROR] MSMS failed for {out_filename1}.pdb")
        print(e)
        set_trace()

    # ---------------------------
    # Step 3: Save PLY
    # ---------------------------
    ply_out = out_filename2 + ".ply"
    print(f"[INFO] Saving mesh to: {ply_out}")
    save_ply(ply_out, vertices1, faces1)


if __name__ == "__main__":
    main()
