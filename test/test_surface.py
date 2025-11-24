#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import pymesh

def compute_bounding_sphere_radius(vertices: np.ndarray) -> float:
    """
    Compute a simple bounding sphere radius.
    Take centroid as center, radius = max distance from center.
    """
    center = vertices.mean(axis=0)
    dists_sq = np.sum((vertices - center) ** 2, axis=1)
    radius = float(np.sqrt(dists_sq.max()))
    return radius


def print_table(info: dict):
    """
    Print a simple ASCII table from a dict of {field: value}.
    """
    str_info = {k: str(v) for k, v in info.items()}

    key_width = max(len(k) for k in str_info.keys())
    val_width = max(len(v) for v in str_info.values())

    sep = "+" + "-" * (key_width + 2) + "+" + "-" * (val_width + 2) + "+"

    print(sep)
    print("| " + "Field".ljust(key_width) + " | " + "Value".ljust(val_width) + " |")
    print(sep)
    for k, v in str_info.items():
        print("| " + k.ljust(key_width) + " | " + v.ljust(val_width) + " |")
    print(sep)


def build_ply_path(mode: str, resolution: str, name: str) -> str:
    """
    Build PLY file path based on mode / resolution / name.
    Expected structure examples:

    amino_acid:
        amino_acid/res_1/ALA/ALA.ply

    protein:
        protein/res_1/7P4C/7P4C.ply
    """
    base_dir = f"../mesh_data/{mode}/"  # amino_acid or protein
    ply_path = os.path.join(base_dir, f"res_{resolution}", name, f"{name}.ply")
    return ply_path


def main():
    parser = argparse.ArgumentParser(description="Surface mesh inspection tool")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["amino_acid", "protein"],
                        help="Input type: amino_acid or protein")

    parser.add_argument("--resolution", type=str, required=True,
                        help="Resolution tag, e.g., res_1")

    parser.add_argument("--name", type=str, required=True,
                        help="Residue name or protein ID, e.g., ALA or 7P4C")

    args = parser.parse_args()

    ply_path = build_ply_path(args.mode, args.resolution, args.name)

    if not os.path.isfile(ply_path):
        print(f"[ERROR] PLY file does not exist: {ply_path}")
        sys.exit(1)

    print(f"[INFO] Loading mesh from: {ply_path}")
    try:
        mesh = pymesh.load_mesh(ply_path)
    except Exception as e:
        print(f"[ERROR] Failed to load mesh: {e}")
        sys.exit(1)

    vertices = mesh.vertices
    faces = mesh.faces

    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]
    radius = compute_bounding_sphere_radius(vertices)

    info = {
        "Mode": args.mode,
        "Resolution": args.resolution,
        "Name": args.name,
        "PLY Path": ply_path,
        "Num vertices": num_vertices,
        "Num faces": num_faces,
        "Bounding sphere radius": f"{radius:.4f}",
    }

    print_table(info)


if __name__ == "__main__":
    main()
