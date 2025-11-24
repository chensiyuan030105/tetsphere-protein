#!/usr/bin/env python3
import os
from pathlib import Path

import yaml


def build_pdb_config(protein_id: str) -> dict:
    """
    Build a config dict for a given protein_id using the provided PDB template.

    Only `protein_id` and `wandb_run_name` are customized per protein.
    All other fields follow the template you provided.
    """
    cfg = {
        "expr_name": "pdb",
        "protein_id": protein_id,
        "resolution": "res_1",
        "fitting_stage": "geometry",
        "wandb_project": "tetsphere",
        "wandb_run_name": protein_id,

        "geometry_type": "TetMeshMultiSphereGeometry",
        "geometry": {
            "initial_mesh_path": "",
            "use_smooth_barrier": True,
            "smooth_barrier_param": {
                "smooth_eng_coeff": 1e-4,
                "barrier_coeff": 1e-4,
                "increase_order_iter": 1000,
            },
            "template_surface_sphere_path": "./data_preprocess/s.1.obj",
            "key_points_file_path": (
                "./dataset/masif/02-benchmark_pdbs_surfaces/"
                "${resolution}/${protein_id}/${protein_id}.json"
            ),
            "tetwild_exec": "../TetWild/build/TetWild",
            "tetwild_cache_folder": ".tetwild_cache",
            "tetwild_scale": 1,
            "load_precomputed_tetwild_mesh": False,
            "debug_mode": False,
        },

        "dataloader_type": "MistubaImgDataLoader",
        "data": {
            "dataset_config": {
                "image_root": "img_data/${expr_name}",
            },
            "world_size": 1,
            "rank": 0,
            "batch_size": 2048,
            "total_num_iter": 15000,
        },

        "renderer": {
            "context_type": "cuda",
            "is_orhto": False,
        },

        "optimizer": {
            "lr": 0.1,
            "grad_limit": True,
            "grad_limit_values": [0.01, 0.01],
            "grad_limit_iters": [2000],
        },

        "milestone_lr_decay": {
            "milestones": [80000, 160000, 320000],
            "gamma": 0.1,
        },

        "pdb_dir": "./dataset/masif/01-benchmark_pdbs",
        "raw_surface_dir": "./dataset/masif/01-benchmark_pdbs_surfaces/${resolution}",
        "output_path": (
            "./dataset/masif/02-benchmark_pdbs_surfaces/"
            "${resolution}/${protein_id}"
        ),
        # Keep interpolation as a string so OmegaConf-style ${...} works
        "total_num_iter": "${data.total_num_iter}",

        "use_permute_surface_v": False,
        "permute_surface_v_param": {
            "start_iter": 2000,
            "end_iter": "${data.total_num_iter}",
            "freq": 1000,
            "start_val": 0.01,
            "end_val": 0.001,
        },

        "verbose": True,

        # Target mesh for fitting
        "target_mesh_path": (
            "./dataset/masif/02-benchmark_pdbs_surfaces/"
            "${resolution}/${protein_id}/${protein_id}.obj"
        ),
        "scale": 0.02,
        "mesh_loss_weight": 0.9999,
        "reg_loss_weight": 1.0e-4,
        "mesh_loss_batch_size": 2048,
        "mesh_loss_sample_ratio": 1.0,
        "epsilon": 0.001,
        "tri_v_scale": 1,

        "sample_points_schedule": {
            "init_num_surface_samples": 100000,
            "milestones": [1000000],
            "gamma": 10,
        },

        "mesh_loss_weight_schedule": {
            "milestones": [320000],
            "forward_facing_loss_weight": [0.5],
            "backward_facing_loss_weight": [0.5],
        },

        # init_scheme: "alpha_carbon", "average_of_atom"
        "init_scheme": "average_of_atom",
        "radius": 0.1,
    }
    return cfg


def main():
    # Folder containing pdb files: ./dataset/masif/01-benchmark_pdbs/*.pdb
    pdb_root = Path("./dataset/masif/01-benchmark_pdbs")
    if not pdb_root.is_dir():
        raise FileNotFoundError(f"PDB directory not found: {pdb_root}")

    # Output config folder: ./config/pdb
    out_dir = Path("./config/pdb")
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(pdb_root.glob("*.pdb"))
    if not pdb_files:
        print(f"[WARN] No .pdb files found in {pdb_root}")
        return

    print(f"[INFO] Found {len(pdb_files)} PDB files in {pdb_root}")

    for pdb_path in pdb_files:
        protein_id = pdb_path.stem  # "1A0G_A" from "1A0G_A.pdb"
        cfg = build_pdb_config(protein_id)

        out_path = out_dir / f"{protein_id}.yaml"

        # If you don't want to overwrite, you can guard here:
        # if out_path.exists(): ...

        with out_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"[INFO] Wrote config: {out_path}")

    print("[INFO] Done generating PDB configs.")


if __name__ == "__main__":
    main()
