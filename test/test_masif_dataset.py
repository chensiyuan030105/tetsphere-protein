#!/usr/bin/env python3
import os
from collections import defaultdict
from Bio.PDB import PDBParser
import csv
from tqdm import tqdm   # <-- NEW

PDB_DIR = "/home/mhg/ForSiyuan/tssplat/dataset/masif/00-raw_pdbs"

# Atom bins
ATOM_BINS = [
    (0, 500),
    (500, 1000),
    (1000, 1500),
    (1500, 2000),
    (2000, 2500),
    (2500, 3000),
    (3000, 3500),
    (3500, 4000),
    (4000, 4500),
    (4500, 5000),
    (5000, 6000),
    (6000, 7000),
    (7000, 8000),
    (8000, 1000000),  # 8000+
]

# Residue bins
RES_BINS = [
    (0, 200),
    (200, 400),
    (400, 600),
    (600, 800),
    (800, 1000),
    (1000, 2000000),  # 1000+
]


def classify(n, bins):
    """Return bin index for a given value."""
    for i, (low, high) in enumerate(bins):
        if low <= n < high:
            return i
    return None


def count_residues(structure):
    """Count protein residues (ignoring hetero atoms)."""
    num = 0
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] == " ":  # skip HETATM
                    num += 1
    return num


def parse_pdb(path):
    """Return atom count and residue count."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", path)
    atom_count = len(list(structure.get_atoms()))
    residue_count = count_residues(structure)
    return atom_count, residue_count


def save_csv(filename, header, rows):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    pdb_files = [f for f in os.listdir(PDB_DIR) if f.endswith(".pdb")]
    pdb_files.sort()

    print(f"[INFO] Found {len(pdb_files)} PDB files.")

    atom_stats = defaultdict(list)
    res_stats = defaultdict(list)
    pdb_table = []

    # -------------------------------
    # Progress bar with tqdm
    # -------------------------------
    print("\n[INFO] Analyzing PDB files...\n")
    for pdb in tqdm(pdb_files, desc="Processing PDBs", ncols=100):
        path = os.path.join(PDB_DIR, pdb)

        try:
            atoms, residues = parse_pdb(path)
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to parse {pdb}: {e}")
            continue

        atom_bin = classify(atoms, ATOM_BINS)
        res_bin = classify(residues, RES_BINS)

        atom_stats[atom_bin].append(pdb)
        res_stats[res_bin].append(pdb)

        pdb_table.append([pdb, atoms, residues])

    # --------------------------
    # Print Atom Count Table
    # --------------------------
    print("\n======= Atom Count Distribution =======")
    for i, (low, high) in enumerate(ATOM_BINS):
        count = len(atom_stats[i])
        label = f"{low} - {high if high != 1000000 else 'inf'}"
        print(f"{label:15s} : {count}")

    # --------------------------
    # Print Residue Count Table
    # --------------------------
    print("\n======= Residue Count Distribution =======")
    for i, (low, high) in enumerate(RES_BINS):
        count = len(res_stats[i])
        label = f"{low} - {high if high != 2000000 else 'inf'}"
        print(f"{label:15s} : {count}")

    # --------------------------
    # Sort PDB files by Atom and Residue Count
    # --------------------------
    print("\n======= Sorting PDB files by atom and residue count =======")
    
    # Sort by atom count (ascending)
    sorted_by_atoms = sorted(pdb_table, key=lambda x: x[1])
    # Sort by residue count (ascending)
    sorted_by_residues = sorted(pdb_table, key=lambda x: x[2])

    # Print sorted lists
    print("\nPDB files sorted by atom count:")
    for pdb, atoms, residues in sorted_by_atoms:
        print(f"{pdb:30s} | Atoms: {atoms:5d} | Residues: {residues:5d}")

    print("\nPDB files sorted by residue count:")
    for pdb, atoms, residues in sorted_by_residues:
        print(f"{pdb:30s} | Atoms: {atoms:5d} | Residues: {residues:5d}")

    # --------------------------
    # Save CSV Files
    # --------------------------
    save_csv("pdb_stats.csv", ["pdb", "atoms", "residues"], pdb_table)
    save_csv("atom_bins.csv", ["range", "count"], [
        [f"{low}-{high}", len(atom_stats[i])] for i, (low, high) in enumerate(ATOM_BINS)
    ])
    save_csv("residue_bins.csv", ["range", "count"], [
        [f"{low}-{high}", len(res_stats[i])] for i, (low, high) in enumerate(RES_BINS)
    ])

    print("\n[INFO] CSV tables saved: pdb_stats.csv, atom_bins.csv, residue_bins.csv")
    print("[INFO] Done.\n")


if __name__ == "__main__":
    main()

