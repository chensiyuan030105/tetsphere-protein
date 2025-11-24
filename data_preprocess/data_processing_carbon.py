#!/usr/bin/env python3
import os
import sys
import json
import shutil
import argparse

import numpy as np
import trimesh

# --------------------------------------------------------------------------
# Make project root importable so we can do: from utils.config import load_config
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.config import load_config


# --------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------
def parse_config():
    parser = argparse.ArgumentParser(
        description="Process amino-acid PDB + surface mesh into OBJ + JSON keypoints."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g. ./config/amino_acid/ALA.yaml)",
    )
    args, extras = parser.parse_known_args()

    cfg_file = args.config
    cfg = load_config(cfg_file, cli_args=extras)
    return cfg_file, cfg


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
vdw_radii = {
    "H": 0.66,
    "C": 1.02,
    "N": 0.93,
    "O": 0.91,
    "S": 1.08,
    "P": 1.08,
}

color_map = {
    "H": [1.0, 1.0, 1.0],
    "C": [0.3, 0.3, 0.3],
    "N": [0.0, 0.0, 1.0],
    "O": [1.0, 0.0, 0.0],
    "S": [1.0, 1.0, 0.0],
    "P": [1.0, 0.5, 0.0],
    "CA": [0.0, 1.0, 0.0],  # C alpha atoms: green
}


def create_spheres(coords_array, elems_array, atom_names_array, scale, filename):
    """
    Create OBJ file with colored atom spheres (all atoms).
    Coordinates are assumed to be already transformed (e.g. center-subtracted and scaled).
    Radii are multiplied by `scale` as well.
    """
    meshes = []

    for (x, y, z), e, name in zip(coords_array, elems_array, atom_names_array):
        color_key = "CA" if name == "CA" else e
        r = vdw_radii.get(e, 1.5) * scale

        sphere = trimesh.creation.icosphere(subdivisions=2, radius=r)
        sphere.apply_translation([x, y, z])

        color = np.array(color_map.get(color_key, [0.5, 0.5, 0.5]))
        sphere.visual.vertex_colors = (color * 255).astype(np.uint8)
        meshes.append(sphere)

        if name == "CA":
            print(f"[CA sphere] center=({x:.4f}, {y:.4f}, {z:.4f}), r={r:.4f}")

    if not meshes:
        print("Warning: no spheres created.")
        return

    molecule = trimesh.util.concatenate(meshes)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    molecule.export(filename)
    print(f"Exported sphere OBJ: {filename}")


def save_alpha_carbon_keypoints_json(pdb_path, output_json, center, radius=0.05):
    """Save keypoints at all C-alpha atom positions, translated by `center` (no scaling)."""
    ca_coords = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ca_coords.append([x, y, z])

    if len(ca_coords) == 0:
        print("Warning: no C-alpha atoms found in PDB.")
        pts = np.zeros((0, 3), dtype=float)
    else:
        ca_coords = np.array(ca_coords, dtype=float)
        pts = ca_coords - center  # only translation

    data = {
        "pt": pts.tolist(),
        "r": [[radius] for _ in range(len(pts))],
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)

    print(f"JSON keypoints (C-alpha) saved: {output_json}")


def save_average_of_atom_keypoints_json(
    pdb_path,
    output_json,
    center,
    scale,
    radius,
    valid_residues=None,
    use_residue_correspondence=False,
):
    """
    Compute the average 3D coordinates of all atoms within each residue (residue centroid),
    translate by `center`, then apply global `scale`, and save as JSON keypoints.

    If use_residue_correspondence is True and valid_residues is provided (set of residue ids),
    only residues whose residue id is in valid_residues will be kept.
    """
    residue_coords = {}
    used_res_ids = set()

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                chain_id = line[21]
                res_id = int(line[22:26])
                key = (chain_id, res_id)

                if use_residue_correspondence and valid_residues is not None:
                    # Keep only residues that appear in the surface mesh
                    if res_id not in valid_residues:
                        continue

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                residue_coords.setdefault(key, []).append([x, y, z])
                used_res_ids.add(res_id)

    avg_coords = []
    for key, atoms in residue_coords.items():
        atoms = np.array(atoms, dtype=float)
        avg = atoms.mean(axis=0)
        avg_coords.append(avg)

    if len(avg_coords) == 0:
        if use_residue_correspondence and valid_residues is not None:
            print(
                "Warning: no residues remained after residue_correspondence filtering "
                "(check PLY residue ids vs PDB residue numbers)."
            )
        else:
            print("Warning: no residues found in PDB.")
        pts = np.zeros((0, 3), dtype=float)
    else:
        avg_coords = np.array(avg_coords, dtype=float)
        pts = avg_coords - center  # only translation
        pts *= scale

    if use_residue_correspondence and valid_residues is not None:
        print(
            f"[average_of_atom] using {len(used_res_ids)} residues "
            f"out of {len(valid_residues)} residues that have surface vertices"
        )

    data = {
        "pt": pts.tolist(),
        "r": [[radius] for _ in range(len(pts))],
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)

    print(f"JSON keypoints (residue-averaged) saved: {output_json}")


def load_ply_with_residue(ply_path):
    """
    Load an ASCII PLY file of the form:

        ply
        format ascii 1.0
        element vertex N
        property float x
        property float y
        property float z
        property float residue
        element face M
        property list uchar int vertex_indices
        end_header
        ...

    Returns:
        verts   : (N_used, 3) float32 array of vertex coordinates
        faces   : (M, k) int32 array of faces (0-based indices into verts)
        residue : (N_used,) float32 array of residue indices (aligned with verts)

    Note:
        Isolated vertices that do not appear in any face are removed,
        and the residue entries for those vertices are removed as well.
    """
    with open(ply_path, "r") as f:
        line = f.readline().strip()
        if line != "ply":
            raise ValueError(f"{ply_path} is not a PLY file (missing 'ply' header).")

        header = [line]
        num_verts = 0
        num_faces = 0
        in_header = True

        while in_header:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of file while reading PLY header.")
            line = line.strip()
            header.append(line)
            if line.startswith("element vertex"):
                # e.g. "element vertex 13494"
                num_verts = int(line.split()[2])
            elif line.startswith("element face"):
                num_faces = int(line.split()[2])
            elif line == "end_header":
                in_header = False

        if num_verts <= 0 or num_faces < 0:
            raise ValueError(
                f"Invalid PLY header in {ply_path}: vertices={num_verts}, faces={num_faces}"
            )

        # Read vertices: assume last column is residue, first 3 are x, y, z
        verts = []
        residue = []
        for _ in range(num_verts):
            vals = f.readline().split()
            if len(vals) < 4:
                raise ValueError(
                    "Vertex line has fewer than 4 values; cannot read residue."
                )
            x, y, z = map(float, vals[0:3])
            res = float(vals[-1])  # assume the last property is 'residue'
            verts.append([x, y, z])
            residue.append(res)

        verts = np.asarray(verts, dtype=np.float32)
        residue = np.asarray(residue, dtype=np.float32)

        # Read faces (list uchar int vertex_indices)
        faces_list = []
        for _ in range(num_faces):
            vals = f.readline().split()
            if len(vals) < 4:
                # at least: "3 i0 i1 i2"
                continue
            n = int(vals[0])
            idx = list(map(int, vals[1:1 + n]))
            faces_list.append(idx)

        if len(faces_list) == 0:
            # No faces: return as-is (nothing to filter by)
            faces = np.zeros((0, 3), dtype=np.int32)
            return verts, faces, residue

        faces = np.asarray(faces_list, dtype=np.int32)

    # ------------------------------------------------------------------
    # Remove isolated vertices (those that do not appear in any face)
    # and update verts, residue, and faces accordingly.
    # ------------------------------------------------------------------
    num_verts_original = verts.shape[0]

    # Collect all vertex indices that are referenced by faces
    used_indices = np.unique(faces.reshape(-1))
    used_indices = used_indices.astype(np.int64)

    # Build an array mapping from old vertex index -> new vertex index
    # Vertices that are not used will keep the default value -1
    mapping = np.full(num_verts_original, -1, dtype=np.int64)
    mapping[used_indices] = np.arange(used_indices.shape[0], dtype=np.int64)

    # Filter verts and residue to keep only used vertices, in the order of used_indices
    verts = verts[used_indices]
    residue = residue[used_indices]

    # Remap face indices to the new 0..N_used-1 range
    faces = mapping[faces]
    faces = faces.astype(np.int32)

    return verts, faces, residue


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    # Read the config file and get the configuration options
    cfg_file, cfg = parse_config()

    # Get relevant settings from config
    protein_id = cfg.get("protein_id", None)
    radius = float(cfg.get("radius", 1.0))
    scale = float(cfg.get("scale", 1.0))
    init_scheme = cfg.get("init_scheme", None)
    use_residue_correspondence = cfg.get(
        "use_residue_correspondence", False
    )

    pdb_dir = cfg.pdb_dir
    raw_surface_dir = cfg.raw_surface_dir
    output_dir = cfg.output_path

    if protein_id is None:
        raise ValueError("`protein_id` must be set in the config.")
    if init_scheme not in ("alpha_carbon", "average_of_atom"):
        raise ValueError("`init_scheme` must be 'alpha_carbon' or 'average_of_atom'.")

    os.makedirs(output_dir, exist_ok=True)

    # Copy config yaml into output_dir
    cfg_basename = os.path.basename(cfg_file)
    dst_cfg = os.path.join(output_dir, cfg_basename)
    try:
        shutil.copy2(cfg_file, dst_cfg)
        print(f"Copied config to: {dst_cfg}")
    except Exception as e:
        print(f"Warning: failed to copy config file to output_dir: {e}")

    pdb_file = os.path.join(pdb_dir, f"{protein_id}.pdb")
    ply_file = os.path.join(raw_surface_dir, f"{protein_id}.ply")

    sphere_obj_file = os.path.join(output_dir, f"{protein_id}_sphere.obj")
    residue_npy_file = os.path.join(output_dir, f"{protein_id}_residue.npy")
    ply_raw_obj_file = os.path.join(output_dir, f"{protein_id}_raw.obj")
    ply_norm_obj_file = os.path.join(output_dir, f"{protein_id}.obj")
    json_file = os.path.join(output_dir, f"{protein_id}.json")

    print(f"[INFO] protein_id  : {protein_id}")
    print(f"[INFO] init_scheme : {init_scheme}")
    print(f"[INFO] radius      : {radius}")
    print(f"[INFO] scale       : {scale}")
    print(f"[INFO] PDB file    : {pdb_file}")
    print(f"[INFO] PLY file    : {ply_file}")
    print(f"[INFO] output_dir  : {output_dir}")

    # ---------- Step 1: Read PDB ----------
    coords, elems, atom_names = [], [], []
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = line[76:78].strip().capitalize() or line[12:14].strip()[0]
                name = line[12:16].strip()

                coords.append([x, y, z])
                elems.append(elem)
                atom_names.append(name)

    if len(coords) == 0:
        raise RuntimeError(f"No ATOM/HETATM records found in PDB: {pdb_file}")

    coords = np.array(coords, dtype=float)

    # Only translate by center, optional global scaling
    center = coords.mean(axis=0)
    coords_shifted = coords - center
    coords_shifted *= scale

    print("Shifted C-alpha coordinates (center-subtracted only):")
    for (x, y, z), name in zip(coords_shifted, atom_names):
        if name == "CA":
            print(f"  CA: ({x:.4f}, {y:.4f}, {z:.4f})")

    # ---------- Step 1c: Sphere OBJ ----------
    create_spheres(coords_shifted, elems, atom_names, scale, sphere_obj_file)

    # ---------- Step 2: PLY -> raw OBJ (keep residue) ----------
    verts_ply, faces_ply, residue = load_ply_with_residue(ply_file)

    residue_ids = residue.astype(int)
    unique_ids, counts = np.unique(residue_ids, return_counts=True)
    print("Residue id -> vertex count")
    for rid, cnt in zip(unique_ids, counts):
        print(f"{rid:4d} : {cnt}")

    # Build set of residues that have at least one surface vertex
    valid_residues = None
    if use_residue_correspondence:
        # Typically residue == 0 means "no corresponding residue"
        valid_residues = set(int(rid) for rid in unique_ids if rid != 0)
        print(
            "[Residue correspondence] residues with surface vertices (excluding 0): "
            f"{len(valid_residues)}"
        )

    # Save residue as sidecar .npy so that vertex i <-> residue[i]
    np.save(residue_npy_file, residue)
    print(f"[PLY] Saved residue array to: {residue_npy_file}")

    # Construct a trimesh from vertices and faces for OBJ export
    ply_mesh = trimesh.Trimesh(vertices=verts_ply, faces=faces_ply, process=False)
    ply_mesh.export(ply_raw_obj_file)
    print(f"[PLY] Raw OBJ saved: {ply_raw_obj_file}")

    # ---------- Step 3: Translate PLY with same center ----------
    v = ply_mesh.vertices.copy()
    v -= center
    ply_mesh.vertices = v * scale
    ply_mesh.export(ply_norm_obj_file)
    print(f"[PLY] Translated OBJ saved: {ply_norm_obj_file}")

    # Reload the exported OBJ and check vertex count
    mesh_reload = trimesh.load(ply_norm_obj_file, process=False)
    print("mesh_reload.vertices:", mesh_reload.vertices.shape[0])
    print("mesh_reload.faces   :", mesh_reload.faces.shape[0])

    # ---------- Step 4: JSON keypoints ----------
    if init_scheme == "alpha_carbon":
        save_alpha_carbon_keypoints_json(
            pdb_file, json_file, center=center, radius=radius
        )
    elif init_scheme == "average_of_atom":
        save_average_of_atom_keypoints_json(
            pdb_file,
            json_file,
            center=center,
            scale=scale,
            radius=radius,
            valid_residues=valid_residues,
            use_residue_correspondence=use_residue_correspondence,
        )


if __name__ == "__main__":
    main()
