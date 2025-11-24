#!/usr/bin/python
import numpy as np
import os
import sys
from Bio.PDB import *
import pymesh
import trimesh

from input_output.save_ply import save_ply

# ============================
# Main Logic Functions
# ============================

def get_residue_atoms(pdb_file):
    """Extract residue atoms from PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    residue_atoms = {}
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip water molecules (HOH/WAT) and non-standard residues
                if residue.id[0] == " " and residue.get_resname() not in ["HOH", "WAT"]:
                    # NOTE: this is only using the residue sequence index.
                    # If you have multiple chains, you may want to include chain.id here.
                    residue_id = str(residue.get_id()[1])
                    atoms = []
                    for atom in residue:
                        atoms.append(atom.get_coord())  # (x, y, z)
                    residue_atoms[residue_id] = atoms
    return residue_atoms


def map_surface_to_residues(surface_points, residue_atoms):
    """
    Map each surface point to its closest residue using the nearest atom.

    For each surface point:
        1) find the nearest atom among all residue atoms
        2) assign the residue of that atom to the surface point

    Args:
        surface_points: (Ns, 3) numpy array of surface vertices.
        residue_atoms: dict[residue_id -> list of atom coords (3,)]

    Returns:
        surface_to_residue: dict[int -> residue_id]
            key   = surface vertex index (0..Ns-1)
            value = residue_id (same type as keys in residue_atoms, e.g. str)
    """
    # Ensure numpy array
    surface_points = np.asarray(surface_points, dtype=float)  # (Ns, 3)
    Ns = surface_points.shape[0]

    # --------------------------------------------------
    # 1. Flatten all atoms into a single array
    # --------------------------------------------------
    all_atom_coords = []
    all_atom_resids = []

    for residue_id, atoms in residue_atoms.items():
        atoms_arr = np.asarray(atoms, dtype=float)  # (Na_i, 3)
        if atoms_arr.shape[0] == 0:
            continue
        all_atom_coords.append(atoms_arr)
        # For each atom in this residue, store its residue_id
        all_atom_resids.extend([residue_id] * atoms_arr.shape[0])

    if len(all_atom_coords) == 0:
        raise ValueError("No atoms found in residue_atoms; check input PDB/processing.")

    all_atom_coords = np.vstack(all_atom_coords)  # (Na_total, 3)
    all_atom_resids = np.array(all_atom_resids, dtype=object)  # (Na_total,)

    Na = all_atom_coords.shape[0]

    # --------------------------------------------------
    # 2. Compute distances: every surface point -> every atom
    # --------------------------------------------------
    # surface_points: (Ns, 3)
    # all_atom_coords: (Na, 3)

    # Broadcasting:
    #   surface_expanded: (Ns, 1, 3)
    #   atoms_expanded:   (1, Na, 3)
    surface_expanded = surface_points[:, np.newaxis, :]  # (Ns, 1, 3)
    atoms_expanded = all_atom_coords[np.newaxis, :, :]   # (1, Na, 3)

    # distances_sq[i, j] = ||surface_points[i] - all_atom_coords[j]||^2
    diff = surface_expanded - atoms_expanded  # (Ns, Na, 3)
    distances_sq = np.sum(diff * diff, axis=2)  # (Ns, Na)

    # --------------------------------------------------
    # 3. For each surface point, pick the closest atom,
    #    then assign the residue of that atom.
    # --------------------------------------------------
    surface_to_residue = {}

    # nearest_atom_indices[i] = index (0..Na-1) of closest atom to surface point i
    nearest_atom_indices = np.argmin(distances_sq, axis=1)  # (Ns,)

    for i in range(Ns):
        atom_idx = nearest_atom_indices[i]
        residue_id = all_atom_resids[atom_idx]
        surface_to_residue[i] = residue_id

    return surface_to_residue


def update_ply_with_residues(input_ply_file, output_ply_file, surface_to_residue):
    """Update PLY file with residue information and per-vertex colors."""

    # Load the mesh from the input PLY file
    mesh = pymesh.load_mesh(input_ply_file)

    num_vertices = mesh.vertices.shape[0]

    # --------------------------------------------------
    # 1) Build per-vertex residue id array
    # --------------------------------------------------
    # residue_ids[i] = residue id string for vertex i
    residue_ids = np.empty(num_vertices, dtype=object)
    for i in range(num_vertices):
        residue_ids[i] = surface_to_residue.get(i, "Unknown")

    # Optional: keep also an Nx1 array if your save_ply expects that
    residue_array = residue_ids.reshape(-1, 1)  # (N, 1), dtype=object

    # --------------------------------------------------
    # 2) Build a color table: residue_id -> RGB
    # --------------------------------------------------
    unique_residues = np.unique(residue_ids)

    rng = np.random.RandomState(0)

    color_table = {}
    for rid in unique_residues:
        if rid == "Unknown":
            # Gray for unknown
            color_table[rid] = np.array([128, 128, 128], dtype=np.uint8)
        else:
            # Random color in [0, 255]
            color_table[rid] = rng.randint(0, 256, size=3, dtype=np.uint8)

    # --------------------------------------------------
    # 3) Assign colors per vertex according to residue id
    # --------------------------------------------------
    # PLY vertex colors are usually uint8 [0..255] or float [0..1]
    vertex_colors_uint8 = np.zeros((num_vertices, 3), dtype=np.uint8)
    for i, rid in enumerate(residue_ids):
        vertex_colors_uint8[i] = color_table[rid]

    # If your save_ply expects float colors in [0, 1]:
    vertex_colors = vertex_colors_uint8.astype(np.float32) / 255.0  # (N, 3)

    # --------------------------------------------------
    # 4) Save the updated mesh with residue + color info
    # --------------------------------------------------
    # This assumes save_ply can take arbitrary per-vertex attributes as kwargs,
    # e.g. it internally does something like:
    #   mesh = pymesh.form_mesh(vertices, faces)
    #   mesh.add_attribute("vertex_color")
    #   mesh.set_attribute("vertex_color", vertex_color)
    #   pymesh.save_mesh(path, mesh, "vertex_color", "residue", ...)
    #
    # If your save_ply uses a different attribute name, adjust "vertex_color".
    save_ply(
        output_ply_file,
        mesh.vertices,
        mesh.faces,
        residue=residue_array,          # keep residue id if you still want it
    )

    print(f"Updated PLY file saved to: {output_ply_file}")
    print(f"  num_vertices = {num_vertices}")
    print(f"  num_residues = {len(unique_residues)}")

    vertex_colors = np.asarray(vertex_colors)
    if vertex_colors.dtype != np.uint8:

        vertex_colors = np.clip(vertex_colors, 0.0, 1.0)
        vertex_colors = (vertex_colors * 255.0).astype(np.uint8)

    print("vertex_colors =", vertex_colors.shape, vertex_colors.dtype)
    print("vertex_colors[0:5] =\n", vertex_colors[:5])

    cloud = trimesh.points.PointCloud(mesh.vertices, colors=vertex_colors)

    base, ext = os.path.splitext(output_ply_file)
    vis_ply_file = base + "_vis" + ext
    cloud.export(vis_ply_file)

    print("[DEBUG] saved vis ply:", vis_ply_file)


# ============================
# Main Execution
# ============================

def compute_surface_and_map_residues(input_pdb, input_ply, output_ply, resolution):
    """Process the surface, map residues, and save new PLY file."""
    
    # Step 1: Extract residues and their atoms
    residue_atoms = get_residue_atoms(input_pdb)
    
    # Step 2: Load the surface PLY file (MSMS result)
    if not os.path.exists(input_ply):
        print(f"[ERROR] Input PLY file not found: {input_ply}")
        sys.exit(1)
        
    mesh = pymesh.load_mesh(input_ply)
    surface_points = mesh.vertices  # [num_surface_points, 3]
    
    # Step 3: Map surface points to residues using numpy for distance calculations
    surface_to_residue = map_surface_to_residues(surface_points, residue_atoms)
    
    # Step 4: Update PLY file with residue information and save it to output path
    update_ply_with_residues(input_ply, output_ply, surface_to_residue)


# ============================
# Input and Output Handling
# ============================

def process_input_and_output(input_pdb, input_ply, output_ply, resolution):
    """Main function to process the input PDB file, map residues, and output the PLY file."""
    print("[INFO] Processing surface and mapping residues.")
    compute_surface_and_map_residues(input_pdb, input_ply, output_ply, resolution)
    print("[INFO] Process completed successfully.")


# ============================
# Main Logic for Argument Parsing
# ============================

if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 7:
        print("Usage:")
        print("  python compute_protein_surface_with_residues.py <input_pdb> <name> <resolution> <pdb_out_dir> <surface_IN_dir> <surface_out_dir>")
        sys.exit(1)

    # Parse arguments
    input_pdb = sys.argv[1]        # path/to/PDB_ID.pdb (full molecule)
    name = sys.argv[2]             # e.g., 1A0G_A
    resolution = sys.argv[3]       # e.g., 10
    pdb_out_dir = sys.argv[4]      # directory for chain PDBs
    surface_in_dir = sys.argv[5]  # directory for input surface meshes (PLY)
    surface_out_dir = sys.argv[6]  # directory for surface meshes (PLY)

    # Extract PDB ID and CHAIN ID from NAME
    pdb_id = name.split("_")[0]  # e.g., "1A0G"
    chain_id = name.split("_")[1]  # e.g., "A"

    # Set up output directory
    os.makedirs(pdb_out_dir, exist_ok=True)

    # Prepare the output PLY file path
    ply_out_dir = os.path.join(surface_out_dir, f"res_{resolution}")
    os.makedirs(ply_out_dir, exist_ok=True)
    ply_out = os.path.join(ply_out_dir, name + ".ply")
    
    # Prepare the input PLY file path
    ply_in = os.path.join(surface_in_dir, f"res_{resolution}", name + ".ply")

    # If the PLY already exists, skip all processing
    if os.path.exists(ply_out):
        print(f"[INFO] PLY already exists, skipping: {ply_out}")
        sys.exit(0)

    # Process input and output handling
    process_input_and_output(input_pdb, ply_in, ply_out, resolution)

