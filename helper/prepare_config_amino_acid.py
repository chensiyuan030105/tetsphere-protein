#!/usr/bin/env python3
import os
from pathlib import Path

import yaml


def build_config(protein_id: str) -> dict:
    """
    Build a config dict for a given protein_id using the amino_acid template.

    Only `protein_id` and `wandb_run_name` are specific per protein.
    All other fields follow the template you provided.
    """
    cfg = {
        "expr_name": "amino_acid",
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
            "key_points_file_path": "./dataset/masif/02-benchmark_amino_acid_surfaces/${resolution}/${protein_id}/${protein_id}.json",

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
            "total_num_iter": 5000,
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

        "pdb_dir": "./dataset/masif/01-benchmark_amino_acid",
        "raw_surface_dir": "./dataset/masif/01-benchmark_amino_acid_surfaces/${resolution}",
        "output_path": "./dataset/masif/02-benchmark_amino_acid_surfaces/${resolution}/${protein_id}",
        # Keep the OmegaConf-style interpolation string
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
        "target_mesh_path": "./dataset/masif/02-benchmark_amino_acid_surfaces/${resolution}/${protein_id}/${protein_id}.obj",
        "scale": 0.05,
        "mesh_loss_weight": 0.99999,
        "reg_loss_weight": 0.00001,
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
    # Template path (used only as an existence check)
    template_path = Path("./config/pdb/1A0G_A.yaml")
    if not template_path.is_file():
        raise FileNotFoundError(f"Template config not found: {template_path}")

    # Source directory with .ply surfaces:
    #   ./dataset/masif/01-benchmark_surfaces/res_1/{protein_id}.ply
    src_root = Path("./dataset/masif/01-benchmark_surfaces/res_1")
    if not src_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    # Output directory for new pdb configs
    out_dir = Path("./config/pdb")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate all .ply files and use the stem as protein_id
    ply_files = sorted(src_root.glob("*.ply"))

    if not ply_files:
        print(f"[WARN] No .ply files found in {src_root}")
        return

    for ply_path in ply_files:
        protein_id = ply_path.stem  # e.g. "1A0G_A"

        # Skip template protein_id
        if protein_id == "1A0G_A":
            print(f"[INFO] Skipping template protein_id: {protein_id}")
            continue

        cfg = build_config(protein_id)
        out_path = out_dir / f"{protein_id}.yaml"

        # Write YAML with keys in the order defined by the dict
        with out_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"[INFO] Wrote config: {out_path}")

    print("[INFO] Done generating amino_acid configs.")


if __name__ == "__main__":
    main()
