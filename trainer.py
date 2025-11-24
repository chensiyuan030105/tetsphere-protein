import torch
import pypgo  # NOQA
import sys
import os
from tqdm import trange, tqdm
import argparse
import numpy as np
import math
from PIL import Image

from geometry import load_geometry
from materials import load_material
from renderers import MeshRasterizer
from data import load_dataloader
from utils.config import load_config
from utils.optimizer import AdamUniform
from pytorch3d.structures import Meshes

device = torch.device("cuda:0")

import torch
import wandb
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance
from sphere_generation import load_target_mesh
from pytorch3d.ops import sample_points_from_meshes, knn_points
import igl
import trimesh
import torch.nn.functional as F

class SamplePointsScheduler:
    def __init__(self, init_num_surface_samples, milestones, gamma):
        # Initialize with the starting number of surface samples,
        # milestones (steps at which adjustments should happen),
        # and gamma (the factor by which surface samples will be multiplied at each milestone).
        self.init_num_surface_samples = init_num_surface_samples
        self.milestones = milestones
        self.gamma = gamma
    
    def get_num_surface_samples(self, step):
        # Set the initial number of surface samples.
        num_surface_samples = self.init_num_surface_samples
        
        # Iterate through the milestones to check if the current step has reached any milestone.
        for milestone in self.milestones:
            # If the current step is greater than or equal to the milestone,
            # multiply the number of surface samples by gamma raised to the power of the milestone's index.
            if step >= milestone:
                num_surface_samples = int(self.init_num_surface_samples * (self.gamma ** (self.milestones.index(milestone) + 1)))
            else:
                break  # If the current step is less than the milestone, stop checking further.
        
        return num_surface_samples

class MeshLossWeightScheduler:
    def __init__(self, milestones, values):
        assert len(milestones) == len(values), "Milestones and values must have the same length."
        self.milestones = milestones
        self.values = values

    def get_value(self, iter_num):
        for i, milestone in enumerate(self.milestones):
            if iter_num < milestone:
                return self.values[i]
        return self.values[-1]

def export_correspondences_to_obj(pred_vertices, nearest_points, filename="correspondences.obj"):
    """
    Export predicted points, nearest surface points, and connecting lines to an OBJ file.
    Open in MeshLab and adjust point/line thickness in the Render menu.

    Args:
        pred_vertices: [N, 3] tensor of predicted vertex coordinates
        nearest_points: [N, 3] tensor of nearest surface coordinates
        filename: output OBJ file path
    """
    pred_np = pred_vertices.detach().cpu().numpy()
    near_np = nearest_points.detach().cpu().numpy()
    N = pred_np.shape[0]

    verts = np.vstack([pred_np, near_np])
    lines = np.array([[i + 1, i + 1 + N] for i in range(N)], dtype=np.int32)  # OBJ indices are 1-based

    with open(filename, "w") as f:
        f.write("# OBJ for MeshLab visualization\n")
        f.write(f"# {2*N} vertices, {N} connecting lines\n\n")

        # --- Predicted points (blue) ---
        f.write("o PredictedPoints\n")
        for v in pred_np:
            f.write(f"v {v[0]} {v[1]} {v[2]} 0 0 1\n")  # Blue color (r g b)
        f.write("\n")

        # --- Nearest points (red) ---
        f.write("o NearestPoints\n")
        for v in near_np:
            f.write(f"v {v[0]} {v[1]} {v[2]} 1 0 0\n")  # Red color (r g b)
        f.write("\n")

        # --- Connecting lines ---
        f.write("o Connections\n")
        for e in lines:
            f.write(f"l {e[0]} {e[1]}\n")

    print(f"[MeshLab] Exported {filename} with {2*N} vertices and {N} lines.")

def export_vertex_gradients_to_obj(geometry, filename: str, eps: float = 1e-12):
    """
    Export surface vertex gradient magnitudes to an OBJ file with per-vertex colors.

    Colors encode gradient norm (blue = small, red = large), so you can inspect
    which regions are actually receiving gradients.

    Requirements on `geometry`:
        - geometry.tet_v           : [N_verts, 3] learnable vertex positions (torch.Tensor)
        - geometry.tet_v.grad      : [N_verts, 3] gradients after loss.backward()
        - geometry.surface_vid     : [N_surface_verts] indices of surface vertices (torch.Tensor or np.ndarray)
        - geometry.surface_fid     : [N_faces, 3] surface faces in local (surface) indexing
    """
    if geometry.tet_v.grad is None:
        print("[GradVis] geometry.tet_v.grad is None (backward has not been called or gradients were cleared).")
        return

    # All vertices and gradients
    v_all = geometry.tet_v.detach().cpu().numpy()        # [Nv, 3]
    g_all = geometry.tet_v.grad.detach().cpu()           # [Nv, 3]
    g_norm_all = g_all.norm(dim=1).numpy()               # [Nv]

    # Restrict to surface vertices
    surface_vid = geometry.surface_vid.detach().cpu().numpy()   # [Ns]
    surface_fid = geometry.surface_fid.detach().cpu().numpy()   # [Fs, 3] (0-based local indices)
    v_surf = v_all[surface_vid]                                 # [Ns, 3]
    g_surf = g_norm_all[surface_vid]                            # [Ns]

    g_min = float(g_surf.min())
    g_max = float(g_surf.max())
    print(f"[GradVis] surface grad norm range = [{g_min:.3e}, {g_max:.3e}]")

    # Normalize gradient magnitudes to [0, 1]
    denom = max(g_max - g_min, eps)
    t = (g_surf - g_min) / denom  # 0 ~ 1

    # Simple blue->red colormap:
    #   t = 0 -> blue  (0, 0, 1)
    #   t = 1 -> red   (1, 0, 0)
    colors = np.stack([t, np.zeros_like(t), 1.0 - t], axis=1)  # [Ns, 3]

    with open(filename, "w") as f:
        f.write("# Gradient visualization OBJ\n")
        f.write(f"# {v_surf.shape[0]} vertices, {surface_fid.shape[0]} faces\n\n")

        # Write vertices with RGB colors
        for (vx, vy, vz), (r, g, b) in zip(v_surf, colors):
            f.write(f"v {vx} {vy} {vz} {r} {g} {b}\n")

        f.write("\n")
        # Faces use 1-based indexing in OBJ
        for (i0, i1, i2) in surface_fid:
            f.write(f"f {i0+1} {i1+1} {i2+1}\n")

    print(f"[GradVis] Exported vertex gradients to {filename}")

def closest_points_and_normals_libigl(
    query_points: torch.Tensor,
    mesh_vertices: torch.Tensor,
    mesh_faces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Use libigl to compute closest points and normals on a triangle mesh for each query point.

    This runs entirely in NumPy / libigl, then converts back to torch tensors
    with requires_grad=False. Gradients will NOT flow through the closest
    points or normals, only through the query_points when you build a loss
    like ||query_points - closest_points||^2.

    Args:
        query_points: [Nq, 3] torch tensor (any device, with or without grad).
        mesh_vertices: [Nv, 3] torch tensor (same mesh as target/pred surface).
        mesh_faces: [F, 3] torch tensor (int, 0-based).

    Returns:
        closest_points: [Nq, 3] torch tensor, closest point on mesh for each query point.
        normals_at_closest: [Nq, 3] torch tensor, face normal at the closest triangle.
        sqr_dists: [Nq] torch tensor, squared distances.
    """
    device = query_points.device
    dtype = query_points.dtype

    # Move to CPU & NumPy, detach to break autograd graph
    P = query_points.detach().cpu().numpy()                  # (Nq, 3)
    V = mesh_vertices.detach().cpu().numpy()                 # (Nv, 3)
    F = mesh_faces.detach().cpu().numpy().astype(np.int32)   # (F, 3)

    # For each point, compute:
    #   sqrD[i] = squared distance to closest point C[i] on triangle mesh (V, F)
    #   I[i]    = index of closest triangle
    #   C[i]    = coordinates of closest point
    sqrD, I, C = igl.point_mesh_squared_distance(P, V, F)    # (Nq,), (Nq,), (Nq, 3)

    # Per-face normals, then pick the face normal for each closest face I[i]
    FN = igl.per_face_normals(V, F)                          # (F, 3), unit normals
    N = FN[I]                                                # (Nq, 3)

    closest_points = torch.from_numpy(C).to(device=device, dtype=dtype)
    normals_at_closest = torch.from_numpy(N).to(device=device, dtype=dtype)
    sqr_dists = torch.from_numpy(sqrD).to(device=device, dtype=dtype)

    # Explicitly mark them as constants (no grad)
    closest_points.requires_grad_(False)
    normals_at_closest.requires_grad_(False)
    sqr_dists.requires_grad_(False)

    return closest_points, normals_at_closest, sqr_dists

def compute_mesh_surface_distance_loss(
    pred_vertices: torch.Tensor,
    pred_faces: torch.Tensor,  # [F_pred, 3] faces of predicted surface (for bi_loss)
    current_surface_normal: torch.Tensor,  # per-vertex normals of predicted surface
    target_vertices: torch.Tensor,
    target_faces: torch.Tensor,
    sample_ratio: float = 1.0,           # IGNORED in libigl version
    num_surface_samples: int = 5000,     # IGNORED in libigl version
    squared: bool = True,
    debug: bool = False,
    bi_loss: bool = False,
    forward_facing_loss_weight: float = 0.5,
    backward_facing_loss_weight: float = 0.5,
    use_euclidean_only: bool = True,
    normal_weight: float = 0.0,          # 新增: 法线惩罚权重 (乘在距离上)
):
    """
    Compute mesh-to-mesh distance using libigl point-mesh distance, with
    optional normal-based multiplicative penalty on the forward term.

    Forward (pred -> target surface):
        - Use libigl.point_mesh_squared_distance(pred_vertices, target_vertices, target_faces)
          to get closest points C_fwd and their face normals N_fwd.
        - Geometric distance term:
              if use_euclidean_only:
                  distance_geom = ||x_pred - C_fwd|| (or squared)
              else:
                  distance_geom = |N_fwd · (x_pred - C_fwd)| (or squared)
        - Normal penalty (if normal_weight > 0 and current_surface_normal is given):
              n_pred = per-vertex normals of predicted surface
              n_tgt  = normals_fwd (from libigl)
              cosθ   = <n_pred, n_tgt> ∈ [-1, 1]
              mis    = 0.5 * (1 - cosθ) ∈ [0, 1]
                       (0° -> 0, 90° -> 0.5, 180° -> 1)
              factor = 1 + normal_weight * mis
              distance = distance_geom * factor

    Backward (optional, target -> pred surface, only if bi_loss=True and pred_faces is provided):
        - Same idea as before, currently only geometric distance is used.

    All closest points and normals are computed in NumPy/libigl and converted back
    as constant tensors (no gradient through them). Gradient flows ONLY through
    `pred_vertices` via the (x - C) difference.

    Returns:
        loss: total loss (forward or forward+backward).
        loss_forward: forward loss.
        loss_backward: backward loss (0 if bi_loss=False or pred_faces is None).
        nearest_points: [Np, 3] closest points on target surface for each predicted vertex.
        nearest_pred: [Nv, 3] closest points on predicted surface for each target vertex
                      (None if bi_loss=False or pred_faces is None).
        dists_pred2target: [Np] per-point forward *geometric* distances
                           (before applying normal penalty).
    """
    device = pred_vertices.device

    # Move target data to the same device (values will still be detached inside igl helper)
    target_vertices = target_vertices.to(device)
    target_faces = target_faces.to(device)

    # ------------------------------------------------------
    # Forward: pred_vertices -> target surface (libigl)
    # ------------------------------------------------------
    closest_fwd, normals_fwd, sqrD_fwd = closest_points_and_normals_libigl(
        pred_vertices,        # query points
        target_vertices,      # mesh vertices
        target_faces,         # mesh faces
    )  # shapes: [Np,3], [Np,3], [Np]

    # ---- 1) pure geometric distance ----
    if use_euclidean_only:
        if squared:
            dist_geom_fwd = sqrD_fwd  # already squared distance [Np]
        else:
            dist_geom_fwd = sqrD_fwd.sqrt()
    else:
        # Normal projection distance |n · (x - C)|
        diff_fwd = pred_vertices - closest_fwd        # [Np,3]
        proj_fwd = (diff_fwd * normals_fwd).sum(dim=1)  # [Np]
        dist_geom_fwd = proj_fwd.abs() if not squared else proj_fwd ** 2

    # Keep the pure geometric distances for debugging
    dists_full = dist_geom_fwd

    # ---- 2) normal-based multiplicative penalty ----
    if (
        normal_weight > 0.0
        and current_surface_normal is not None
        and current_surface_normal.numel() > 0
    ):
        n_pred = current_surface_normal.to(device)

        # Basic shape guard: expect [Np,3]
        if n_pred.shape != normals_fwd.shape:
            Np = min(n_pred.shape[0], normals_fwd.shape[0])
            if debug:
                print(
                    f"[SurfaceLoss-igl] Warning: normal shape mismatch, "
                    f"clipping to {Np} vertices."
                )
            n_pred = n_pred[:Np]
            normals_fwd = normals_fwd[:Np]
            dist_geom_fwd = dist_geom_fwd[:Np]
            closest_fwd = closest_fwd[:Np]
            dists_full = dists_full[:Np]

        # Normalize normals
        n_pred = F.normalize(n_pred, dim=1, eps=1e-8)
        n_tgt = F.normalize(normals_fwd, dim=1, eps=1e-8)

        # cos(theta) in [-1, 1]
        cos_theta = (n_pred * n_tgt).sum(dim=1).clamp(-1.0, 1.0)

        # misalignment in [0, 1]: 0° -> 0, 90° -> 0.5, 180° -> 1
        mis = 0.5 * (1.0 - cos_theta)

        # multiplicative factor: mis越大，factor越大
        factor = 1.0 + normal_weight * mis    # [Np]

        dist_fwd = dist_geom_fwd * factor
    else:
        dist_fwd = dist_geom_fwd

    loss_forward = dist_fwd.mean()

    # ------------------------------------------------------
    # Backward: target_vertices -> predicted surface (optional)
    # ------------------------------------------------------
    loss_backward = pred_vertices.new_tensor(0.0)
    nearest_pred = None

    if bi_loss and pred_faces is not None:
        pred_faces = pred_faces.to(device)

        closest_back, normals_back, sqrD_back = closest_points_and_normals_libigl(
            target_vertices,   # query points
            pred_vertices,     # predicted surface vertices
            pred_faces,        # predicted surface faces
        )  # [Nv,3], [Nv,3], [Nv]

        if use_euclidean_only:
            if squared:
                dist_back = sqrD_back
            else:
                dist_back = sqrD_back.sqrt()
        else:
            diff_back = target_vertices - closest_back
            proj_back = (diff_back * normals_back).sum(dim=1)
            dist_back = proj_back.abs() if not squared else proj_back ** 2

        loss_backward = dist_back.mean()
        nearest_pred = closest_back

        loss = (
            forward_facing_loss_weight * loss_forward
            + backward_facing_loss_weight * loss_backward
        )
    else:
        if bi_loss and pred_faces is None and debug:
            print("[SurfaceLoss] bi_loss=True but pred_faces is None, disabling backward term.")
        loss = loss_forward

    if debug:
        print(
            f"[SurfaceLoss-igl] "
            f"loss={loss.item():.10f}, "
            f"loss_forward={loss_forward.item():.10f}, "
            f"loss_backward={loss_backward.item():.10f}, "
            f"use_euclidean_only={use_euclidean_only}, "
            f"normal_weight={normal_weight}"
        )

    return loss, loss_forward, loss_backward, closest_fwd, nearest_pred, dists_full

def groupwise_knn_surface_loss(
    pred_vertices: torch.Tensor,        # [Np, 3]
    pred_group_ids: torch.Tensor = None,       # [Np], int64, e.g. residue ids for predicted points
    target_vertices: torch.Tensor = None,      # [Nt, 3]
    target_group_ids: torch.Tensor = None,     # [Nt], int64, e.g. residue ids for target points
    pred_normals: torch.Tensor = None,         # [Np, 3], used for backward normal-projection
    target_faces: torch.Tensor = None,         # [Ft, 3] faces of target mesh (for normals if needed)
    target_normals: torch.Tensor = None,       # [Nt, 3] or None
    use_normal_projection: bool = False,
    squared: bool = True,
    debug: bool = False,
    bi_loss: bool = False,                     # <--- NEW: enable target->pred term
    forward_facing_loss_weight: float = 0.5,
    backward_facing_loss_weight: float = 0.5,
):
    """
    Group-wise KNN surface loss with optional bidirectional term.

    Forward (pred -> target):
        For each predicted point p_i with group id g:
            - Only search nearest neighbor among target points whose group id == g.
            - Distance definition:
                If use_normal_projection is False:
                    d_i = || p_i - t_j ||      (or squared)
                If use_normal_projection is True:
                    d_i = | n_j · (p_i - t_j) | (or squared),
                    where n_j is the normal at the matched target point t_j.

    Optional backward (target -> pred), if bi_loss=True:
        For each target point t_k with group id g:
            - Only search nearest neighbor among predicted points whose group id == g.
            - Distance definition is the same form, but using normals on the
              predicted surface (pred_normals), if use_normal_projection=True.

    All normals used in the projection are detached, so no gradient flows through
    the normals themselves.

    Args:
        pred_vertices: [Np, 3] predicted surface points.
        pred_group_ids: [Np] group id for each predicted point. If None, all points
            are treated as belonging to a single group.
        target_vertices: [Nt, 3] target surface points.
        target_group_ids: [Nt] group id for each target point. If None, all points
            are treated as belonging to a single group.
        pred_normals: [Np, 3] predicted surface normals. Required if
            bi_loss=True and use_normal_projection=True.
        target_faces: [Ft, 3] faces of the target mesh. Required if
            use_normal_projection=True and target_normals is None.
        target_normals: [Nt, 3] per-target-vertex normals. If
            use_normal_projection=True and this is None, normals will be
            computed from (target_vertices, target_faces) via PyTorch3D.
        use_normal_projection: if False, use Euclidean distance between points;
            if True, use normal-projection distance |n · (p - t)|.
        squared: if True, use squared distances; otherwise use sqrt/abs.
        debug: if True, print debug info.
        bi_loss: if True, also compute target->pred term and include it in the loss.

    Returns:
        raw_loss:
            Scalar tensor. Weighted sum of forward and backward losses:
                loss = loss_fwd              (if bi_loss=False)
                loss = 0.5*loss_fwd + 0.5*loss_bwd (default weights, adjust as needed)
        loss_forward:
            Scalar tensor. Mean forward loss over all valid predicted points.
        loss_backward:
            Scalar tensor. Mean backward loss over all valid target points,
            or 0 if bi_loss=False.
        nearest_points:
            [Np, 3]. For each predicted point, the nearest target point found
            inside the same group. For groups that do not exist in the target,
            the distances will be NaN and excluded from the loss.
        nearest_pred:
            [Nt, 3] if bi_loss=True, else None. For each target point, the nearest
            predicted point in the same group.
        dists_forward:
            [Np]. Per-predicted-point forward distance value used in the loss.
            Entries are NaN where the point's group had no match in the target.
    """
    # Sanity checks to catch shape mismatches on CPU rather than through CUDA assert
    assert pred_vertices.shape[0] == pred_group_ids.shape[0], (
        f"pred_vertices N={pred_vertices.shape[0]} vs "
        f"pred_group_ids N={pred_group_ids.shape[0]} (must match)"
    )
    assert target_vertices.shape[0] == target_group_ids.shape[0], (
        f"target_vertices N={target_vertices.shape[0]} vs "
        f"target_group_ids N={target_group_ids.shape[0]} (must match)"
    )
    
    device = pred_vertices.device
    pred_vertices = pred_vertices.to(device)
    target_vertices = target_vertices.to(device)

    # If group ids are None, treat all points as one group (id 0).
    if pred_group_ids is None:
        pred_group_ids = torch.zeros(
            pred_vertices.shape[0], dtype=torch.long, device=device
        )
    else:
        pred_group_ids = pred_group_ids.to(device)

    if target_group_ids is None:
        target_group_ids = torch.zeros(
            target_vertices.shape[0], dtype=torch.long, device=device
        )
    else:
        target_group_ids = target_group_ids.to(device)

    if target_faces is not None:
        target_faces = target_faces.to(device)
    if target_normals is not None:
        target_normals = target_normals.to(device)

    # ---------------------------------------------------------
    # 1) Prepare target normals (for forward normal-projection)
    # ---------------------------------------------------------
    if use_normal_projection and target_normals is None:
        if target_faces is None:
            raise ValueError(
                "use_normal_projection=True but target_normals is None "
                "and target_faces is None; cannot compute target normals."
            )
        mesh = Meshes(verts=[target_vertices], faces=[target_faces])
        # mesh_normals returns [1, Nt, 3]
        target_normals = mesh_normals(mesh)[0]  # [Nt, 3]

    if use_normal_projection and target_normals is None:
        raise ValueError(
            "target_normals is still None after attempting to compute them. "
            "Please check your inputs."
        )

    # ---------------------------------------------------------
    # 2) Forward: pred -> target (group-wise KNN)
    # ---------------------------------------------------------
    Np = pred_vertices.shape[0]

    nearest_points = torch.zeros_like(pred_vertices)  # [Np, 3]
    nearest_normals_forward = (
        torch.zeros_like(pred_vertices) if target_normals is not None else None
    )
    dists_forward = pred_vertices.new_full((Np,), float("nan"))

    unique_groups_pred = torch.unique(pred_group_ids)

    for g in unique_groups_pred:
        mask_p = (pred_group_ids == g)   # [Np]
        mask_t = (target_group_ids == g) # [Nt]

        if not (mask_p.any() and mask_t.any()):
            continue

        idx_p = mask_p.nonzero(as_tuple=False).view(-1)  # [Np_g]
        idx_t = mask_t.nonzero(as_tuple=False).view(-1)  # [Nt_g]

        P = pred_vertices[idx_p][None, ...]    # [1, Np_g, 3]
        T = target_vertices[idx_t][None, ...]  # [1, Nt_g, 3]

        knn = knn_points(P, T, K=1)
        local_nn_idx = knn.idx[0, :, 0]        # [Np_g]
        global_nn_idx = idx_t[local_nn_idx]    # [Np_g]

        nearest_points[idx_p] = target_vertices[global_nn_idx]

        if use_normal_projection:
            diff = pred_vertices[idx_p] - target_vertices[global_nn_idx]   # [Np_g, 3]
            # Detach normals so no gradient flows through them
            n_sel = target_normals[global_nn_idx].detach()                 # [Np_g, 3]
            proj = (diff * n_sel).sum(dim=1)                               # [Np_g]
            if squared:
                dist = proj ** 2
            else:
                dist = proj.abs()
        else:
            if squared:
                dist = knn.dists[0, :, 0]          # [Np_g], squared ||p - t||
            else:
                dist = knn.dists[0, :, 0].sqrt()   # [Np_g]

        dists_forward[idx_p] = dist

        if nearest_normals_forward is not None:
            nearest_normals_forward[idx_p] = target_normals[global_nn_idx]

    valid_fwd = ~torch.isnan(dists_forward)
    if valid_fwd.any():
        loss_forward = dists_forward[valid_fwd].mean()
    else:
        loss_forward = pred_vertices.new_tensor(0.0)

    # ---------------------------------------------------------
    # 3) Backward: target -> pred (optional, group-wise KNN)
    # ---------------------------------------------------------
    Nt = target_vertices.shape[0]
    nearest_pred = None
    dists_backward = target_vertices.new_full((Nt,), float("nan"))
    loss_backward = pred_vertices.new_tensor(0.0)

    if bi_loss:
        # If we want normal-projection in backward, we need pred_normals.
        if use_normal_projection and pred_normals is None:
            raise ValueError(
                "bi_loss=True and use_normal_projection=True but pred_normals is None. "
                "Provide predicted normals or disable normal projection for backward."
            )
        if pred_normals is not None:
            pred_normals = pred_normals.to(device)

        nearest_pred = torch.zeros_like(target_vertices)  # [Nt, 3]

        unique_groups_target = torch.unique(target_group_ids)

        for g in unique_groups_target:
            mask_t = (target_group_ids == g)  # [Nt]
            mask_p = (pred_group_ids == g)    # [Np]

            if not (mask_p.any() and mask_t.any()):
                continue

            idx_t = mask_t.nonzero(as_tuple=False).view(-1)  # [Nt_g]
            idx_p = mask_p.nonzero(as_tuple=False).view(-1)  # [Np_g]

            T = target_vertices[idx_t][None, ...]  # [1, Nt_g, 3]
            P = pred_vertices[idx_p][None, ...]    # [1, Np_g, 3]

            knn_back = knn_points(T, P, K=1)
            local_nn_idx_back = knn_back.idx[0, :, 0]      # [Nt_g]
            global_pred_idx = idx_p[local_nn_idx_back]     # [Nt_g]

            nearest_pred[idx_t] = pred_vertices[global_pred_idx]

            if use_normal_projection:
                diff_back = target_vertices[idx_t] - pred_vertices[global_pred_idx]  # [Nt_g, 3]
                # Detach normals again to prevent gradient through normals
                n_sel_back = pred_normals[global_pred_idx].detach()                  # [Nt_g, 3]
                proj_back = (diff_back * n_sel_back).sum(dim=1)                      # [Nt_g]
                if squared:
                    dist_back = proj_back ** 2
                else:
                    dist_back = proj_back.abs()
            else:
                if squared:
                    dist_back = knn_back.dists[0, :, 0]         # [Nt_g]
                else:
                    dist_back = knn_back.dists[0, :, 0].sqrt()  # [Nt_g]

            dists_backward[idx_t] = dist_back

        valid_bwd = ~torch.isnan(dists_backward)
        if valid_bwd.any():
            loss_backward = dists_backward[valid_bwd].mean()
        else:
            loss_backward = pred_vertices.new_tensor(0.0)

    # ---------------------------------------------------------
    # 4) Combine forward and backward losses
    # ---------------------------------------------------------
    if bi_loss:
        # You can expose these weights as arguments if you want
        w_fwd = forward_facing_loss_weight
        w_bwd = backward_facing_loss_weight
        loss = w_fwd * loss_forward + w_bwd * loss_backward
    else:
        loss = loss_forward

    if debug:
        num_valid_fwd = valid_fwd.sum().item()
        num_valid_bwd = int((~torch.isnan(dists_backward)).sum().item()) if bi_loss else 0
        print(
            f"[GroupKNN] "
            f"valid_forward={num_valid_fwd}/{Np}, "
            f"valid_backward={num_valid_bwd}/{Nt}, "
            f"loss={loss.item():.10f}, "
            f"loss_forward={loss_forward.item():.10f}, "
            f"loss_backward={loss_backward.item():.10f}, "
            f"use_normal_projection={use_normal_projection}, "
            f"bi_loss={bi_loss}"
        )

    return loss, loss_forward, loss_backward, nearest_points, nearest_pred, dists_forward

class LinearInterpolateScheduler:
    def __init__(self, start_iter, end_iter, start_val, end_val, freq):
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.start_val = start_val
        self.end_val = end_val
        self.freq = freq

    def __call__(self, iter):
        if iter < self.start_iter or iter % self.freq != 0 or iter == 0:
            return None

        p = (iter - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_val * (1 - p) + self.end_val * p


def save_group_colored_pointcloud_pair(
    pred_vertices,        # torch.Tensor [Np, 3] or np.ndarray
    pred_group_ids,       # torch.Tensor [Np]   or np.ndarray (int)
    target_vertices,      # torch.Tensor [Nt, 3] or np.ndarray
    target_group_ids,     # torch.Tensor [Nt]   or np.ndarray (int)
    out_dir,
    file_prefix="debug_res",
):
    """
    Save multiple PLY files, one per residue (group id).

    For each residue id 'gid' (except -1), this function writes a PLY file:
        {out_dir}/{file_prefix}_{gid}.ply

    Each file contains:
      - predicted vertices belonging to residue gid
      - target vertices belonging to residue gid
    in the SAME coordinate frame (no offset).

    Colors:
      - predicted  vertices: blue  (0, 0, 255)
      - target     vertices: red   (255, 0, 0)
    """

    # ------------------------------------------------------------------
    # 0. Convert to numpy
    # ------------------------------------------------------------------
    if isinstance(pred_vertices, torch.Tensor):
        pred_vertices = pred_vertices.detach().cpu().numpy()
    if isinstance(pred_group_ids, torch.Tensor):
        pred_group_ids = pred_group_ids.detach().cpu().numpy()
    if isinstance(target_vertices, torch.Tensor):
        target_vertices = target_vertices.detach().cpu().numpy()
    if isinstance(target_group_ids, torch.Tensor):
        target_group_ids = target_group_ids.detach().cpu().numpy()

    pred_vertices = np.asarray(pred_vertices, dtype=np.float32)
    target_vertices = np.asarray(target_vertices, dtype=np.float32)
    pred_group_ids = np.asarray(pred_group_ids, dtype=np.int64)
    target_group_ids = np.asarray(target_group_ids, dtype=np.int64)

    # ------------------------------------------------------------------
    # 1. Collect all residue ids (exclude -1)
    # ------------------------------------------------------------------
    valid_pred = pred_group_ids >= 0
    valid_tgt = target_group_ids >= 0
    all_group_ids = np.concatenate(
        [pred_group_ids[valid_pred], target_group_ids[valid_tgt]]
    )
    unique_gids = np.unique(all_group_ids)

    if unique_gids.size == 0:
        print("[WARN] No valid residue ids found (all group ids < 0).")
        return

    # Fixed colors for clarity
    color_pred = np.array([0, 0, 255], dtype=np.uint8)   # blue
    color_tgt  = np.array([255, 0, 0], dtype=np.uint8)   # red

    # ------------------------------------------------------------------
    # 2. For each residue, dump a separate PLY
    # ------------------------------------------------------------------
    # Create a separate subfolder for this residue
    res_dir = os.path.join(out_dir, f"binding_point")
    os.makedirs(res_dir, exist_ok=True)

    for gid in unique_gids:
        # boolean masks
        mask_p = (pred_group_ids == gid)
        mask_t = (target_group_ids == gid)

        if (not mask_p.any()) and (not mask_t.any()):
            # should not happen, but just in case
            continue

        verts_p = pred_vertices[mask_p]   # [Np_g, 3]
        verts_t = target_vertices[mask_t] # [Nt_g, 3]

        # concatenate vertices
        all_vertices = np.concatenate([verts_p, verts_t], axis=0)

        # assign colors
        colors_p = np.tile(color_pred[None, :], (verts_p.shape[0], 1))
        colors_t = np.tile(color_tgt[None, :], (verts_t.shape[0], 1))
        all_colors = np.concatenate([colors_p, colors_t], axis=0)

        # build point cloud and export as PLY
        cloud = trimesh.points.PointCloud(all_vertices, colors=all_colors)

        out_path = os.path.join(res_dir, f"{file_prefix}_{int(gid)}.ply")
        cloud.export(out_path)

        print(
            f"[DEBUG] Saved residue {int(gid)} to: {out_path} "
            f"(N_pred={verts_p.shape[0]}, N_tgt={verts_t.shape[0]})"
        )

def train(cfg):

    # Initialize wandb run
    wandb.init(
        project=cfg.get("wandb_project", "tetsphere"),
        name=cfg.get("wandb_run_name", "1N9V_A"),
    )

    verbose = cfg.get("verbose", False)

    # Initialize the scheduler with the provided config values.
    point_scheduler = SamplePointsScheduler(
        cfg["sample_points_schedule"]["init_num_surface_samples"], 
        cfg["sample_points_schedule"]["milestones"], 
        cfg["sample_points_schedule"]["gamma"]
    )

    mesh_loss_weight_cfg = cfg.get("mesh_loss_weight_schedule", None)
    if mesh_loss_weight_cfg is not None:
        forward_scheduler = MeshLossWeightScheduler(
            mesh_loss_weight_cfg["milestones"],
            mesh_loss_weight_cfg["forward_facing_loss_weight"]
        )
        backward_scheduler = MeshLossWeightScheduler(
            mesh_loss_weight_cfg["milestones"],
            mesh_loss_weight_cfg["backward_facing_loss_weight"]
        )
    else:
        forward_scheduler = None
        backward_scheduler = None

    # Load target mesh if provided
    target_vertices = None
    if cfg.get("target_mesh_path", None) is not None:
        target_vertices, target_surfaces = load_target_mesh(cfg.target_mesh_path)
        target_vertices = target_vertices.to(device)
        target_surfaces = target_surfaces.to(device)

        print(f"Loaded target mesh from {cfg.target_mesh_path} with {target_vertices.shape[0]} vertices")
    # main loop
    # is_mesh_fitting = target_vertices is not None and cfg.get("fitting_stage", None) == "geometry"
    loss_batch_size = cfg.data.get("mesh_loss_batch_size", 2048) 
    mesh_sample_ratio = cfg.get("mesh_loss_sample_ratio", 1.0)  

    material = None
    cfg.geometry.optimize_geo = True
    cfg.geometry.output_path = cfg.output_path
    os.makedirs(os.path.join(cfg.output_path, "final/"), exist_ok=True)

    shade_loss = torch.nn.MSELoss()

    if cfg.get("fitting_stage", None) == "texture":
        assert cfg.get("material", None) is not None
        material = load_material(cfg.material_type)(cfg.material)
        cfg.geometry.optimize_geo = False
        shade_loss = torch.nn.L1Loss()

    print("Loading geometry...")
    geometry = load_geometry(cfg.geometry_type)(cfg.geometry)
    it = 0
    os.makedirs(f"{cfg.output_path}/mesh{it:05d}", exist_ok=True)
    geometry.export(f"{cfg.output_path}/mesh{it:05d}", f"{it:05d}")

    print("Setting up renderer and dataloader...")
    renderer = MeshRasterizer(geometry, material, cfg.renderer)
    num_forward_per_iter = 1

    optimizer = AdamUniform(renderer.parameters(), **cfg.optimizer)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestone_lr_decay"]["milestones"],
        gamma=cfg["milestone_lr_decay"]["gamma"]
    )

    permute_surface_scheduler = None
    if cfg.use_permute_surface_v:
        permute_surface_scheduler = LinearInterpolateScheduler(
            **cfg.permute_surface_v_param)
    
    # main loop
    best_loss = 1e10
    best_loss_iter = 0
    best_opt_imgs = None
    best_v = None

    use_residue_correspondence = cfg.get("use_residue_correspondence", False)

    if use_residue_correspondence:

        # ------------------------------------------------------------------
        # 1) Load residue ids for TARGET surface vertices from .npy
        #    target_residue_ids_np[i] = residue id of target vertex i
        # ------------------------------------------------------------------
        target_residue_ids_np = np.load(
            os.path.join(cfg.get("output_path"), f"{cfg.protein_id}_residue.npy")
        ).astype(np.int64)  # shape [Nt]

        # Print how many vertices each residue has (for sanity check / debugging)
        unique_ids, counts = np.unique(target_residue_ids_np, return_counts=True)
        print("unique_ids =", unique_ids)
        print("Residue id -> vertex count (TARGET)")
        for rid, cnt in zip(unique_ids, counts):
            print(f"{rid:4d} : {cnt}")

        # Convert target residue ids to torch tensor (group id per target vertex)
        target_group_ids = torch.from_numpy(target_residue_ids_np).to(
            device=device, dtype=torch.long
        )  # shape [Nt]


        # ------------------------------------------------------------------
        # 2) Build group ids for PREDICTED surface vertices
        #    geometry.all_spheres_vtx_idx is assumed to be:
        #        list of length = num_residues,
        #        each entry is a list/array of vertex indices belonging to that residue
        #
        #    We want a 1D array pred_group_ids_np of length Np (number of predicted
        #    vertices) such that:
        #        pred_group_ids_np[i] = residue id of predicted vertex i
        # ------------------------------------------------------------------
        # Select only the surface vertices
        current_surface_vertices = geometry.tet_v[geometry.surface_vid]
        Np = current_surface_vertices.shape[0]  # number of surface vertices

        # Initialize all surface vertices with group id -1 (meaning "unassigned")
        pred_group_ids_np = np.full(Np, -1, dtype=np.int64)

        # all_spheres[k] -> indices of vertices (in the full tet mesh) belonging to residue unique_ids[k]
        all_spheres = geometry.all_spheres_vtx_idx

        assert len(all_spheres) == len(unique_ids), (
            f"len(all_spheres)={len(all_spheres)} != len(unique_ids)={len(unique_ids)}; "
            "they should both correspond to the same residue set."
        )

        # Build a mapping from global vertex index (in tet_v) to local surface index (0..Np-1)
        global2local = {
            int(global_idx): local_idx
            for local_idx, global_idx in enumerate(geometry.surface_vid)
        }

        # Optional: store per-residue lists of surface vertex indices (in surface-local indexing)
        surface_all_spheres = []

        for res_id, vert_idx_list in zip(unique_ids, all_spheres):
            print("res_id =", res_id)
            vert_idx_arr = np.asarray(vert_idx_list, dtype=np.int64)

            # Keep only vertices that are on the surface, and remap them to surface-local indices
            local_idx_list = []
            for idx in vert_idx_arr:
                g = int(idx)
                if g in global2local:
                    local_idx_list.append(global2local[g])
                # If g is not in global2local, this vertex is not on the surface and is discarded

            if not local_idx_list:
                # This residue has no vertices on the surface
                surface_all_spheres.append([])
                continue

            local_idx_arr = np.asarray(local_idx_list, dtype=np.int64)
            surface_all_spheres.append(local_idx_arr)

            # Assign the residue id to all surface vertices in this group
            pred_group_ids_np[local_idx_arr] = res_id


        # Warn if any predicted vertex did not get a group id (still -1)
        if (pred_group_ids_np < 0).any():
            bad_idx = np.where(pred_group_ids_np < 0)[0]
            print(
                f"[WARN] {bad_idx.size} predicted vertices have no residue/group id. "
                f"Example indices: {bad_idx[:10]}"
            )

        # Convert to torch tensor used by the loss
        pred_group_ids = torch.from_numpy(pred_group_ids_np).to(
            device=device, dtype=torch.long
        )  # shape [Np]

        print("pred_group_ids.shape  =", pred_group_ids.shape)
        print("target_group_ids.shape =", target_group_ids.shape)
        print("pred_group_ids  =", pred_group_ids)
        print("target_group_ids =", target_group_ids)

        current_surface_vertices = geometry.tet_v[geometry.surface_vid]
        save_group_colored_pointcloud_pair(
            pred_vertices=current_surface_vertices,      # torch.Tensor [Np, 3]
            pred_group_ids=pred_group_ids,           # torch.Tensor [Np]
            target_vertices=target_vertices, # torch.Tensor [Nt, 3]
            target_group_ids=target_group_ids,       # torch.Tensor [Nt]
            out_dir=cfg.get("output_path"),
        )

    for it in trange(cfg.total_num_iter+1):
        for forw_id in range(num_forward_per_iter):

            fit_depth = cfg.get("fit_depth", False)
            if fit_depth:
                fit_depth = cfg.get("fit_depth_starting_iter", 0) < it
            
            renderer_input = {
                "iter_num": it,
                "permute_surface_scheduler": permute_surface_scheduler,
            }

            # forward
            out = renderer.compute_geometry_forward(**renderer_input)

            # Compute loss
            current_surface_vertices = geometry.tet_v[geometry.surface_vid]
            current_surface_faces = geometry.surface_fid
            current_surface_normal = geometry.compute_vertex_normal()
            
            if forward_scheduler is not None:
                forward_facing_loss_weight = forward_scheduler.get_value(it)
            else:
                forward_facing_loss_weight = cfg.get("forward_facing_loss_weight", 0.5)

            if backward_scheduler is not None:
                backward_facing_loss_weight = backward_scheduler.get_value(it)
            else:
                backward_facing_loss_weight = cfg.get("backward_facing_loss_weight", 0.5)
            num_surface_samples = point_scheduler.get_num_surface_samples(it)

            if use_residue_correspondence:
                raw_mesh_loss, loss_forward, loss_backward, nearest_points, nearest_pred, dists = groupwise_knn_surface_loss(
                    pred_vertices=current_surface_vertices,      # [Np, 3]
                    pred_group_ids=pred_group_ids,            # [Np]
                    target_vertices=target_vertices,    # [Nt, 3]
                    target_group_ids=target_group_ids,        # [Nt]
                    pred_normals=current_surface_normal,      # [Nt, 3] or None
                    target_normals=None,      # [Nt, 3] or None
                    use_normal_projection=False,                
                    squared=True,
                    debug=False,
                    bi_loss=True,
                    forward_facing_loss_weight=forward_facing_loss_weight,
                    backward_facing_loss_weight=backward_facing_loss_weight,
                )

            else:
                raw_mesh_loss, loss_forward, loss_backward, nearest_points, nearest_pred, dists = compute_mesh_surface_distance_loss(
                    current_surface_vertices,
                    current_surface_faces,
                    current_surface_normal,
                    target_vertices,
                    target_surfaces,
                    sample_ratio=1.0,
                    num_surface_samples=num_surface_samples,
                    squared=True,
                    debug=False,
                    bi_loss=True,
                    forward_facing_loss_weight=forward_facing_loss_weight,
                    backward_facing_loss_weight=backward_facing_loss_weight,
                    use_euclidean_only=cfg.get("use_euclidean_only", False),
                    normal_weight=cfg.get("normal_weight", 0.0),
                )

            reg_loss = 0.0
            if cfg.get("fitting_stage", None) == "geometry":
                reg_loss = out["geo_regularization"].abs()

            loss = raw_mesh_loss * cfg.get("mesh_loss_weight") + reg_loss * cfg.get("reg_loss_weight")

            if it % 1000 == 0:  # it % 100 == 0:
                tqdm.write(
                    "iter=%4d, loss_forward=%.10f, loss_backward=%.10f, reg_loss=%.10f"
                    % (
                        it,
                        loss_forward,
                        loss_backward,
                        reg_loss,
                    )
                )

            # backward
            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            optimizer.step()
            scheduler.step()

            # Log losses and learning rate to Weights & Biases (wandb)
            wandb.log({
                "iter": it,
                "num_surface_samples": num_surface_samples,
                "forward_facing_loss_weight": forward_facing_loss_weight,
                "backward_facing_loss_weight": backward_facing_loss_weight,
                "log_loss_forward": math.log10(loss_forward.item()) if isinstance(loss_forward, torch.Tensor) else math.log10(loss_forward),
                "log_loss_backward": math.log10(loss_backward.item()) if isinstance(loss_backward, torch.Tensor) else math.log10(loss_backward),
                "log_reg_loss": math.log10(reg_loss.item()) if isinstance(reg_loss, torch.Tensor) else math.log10(reg_loss),
                "log_total_loss": math.log10(loss.item()) if isinstance(loss, torch.Tensor) else math.log10(loss),
                "lr": optimizer.param_groups[0]['lr']
            })


            if it % 1000 == 0:
                os.makedirs(f"{cfg.output_path}/mesh{it:05d}", exist_ok=True)
                geometry.export(f"{cfg.output_path}/mesh{it:05d}", f"{it:05d}")
                export_correspondences_to_obj(current_surface_vertices, nearest_points, f"{cfg.output_path}/mesh{it:05d}/{it:05d}_correspondences_forward.obj")
                export_correspondences_to_obj(target_vertices, nearest_pred, f"{cfg.output_path}/mesh{it:05d}/{it:05d}_correspondences_backward.obj")
                grad_obj_path = f"{cfg.output_path}/mesh{it:05d}/{it:05d}_gradients.obj"
                export_vertex_gradients_to_obj(geometry, grad_obj_path)

    print(f"Best rendering loss: {best_loss} at iteration {best_loss_iter}")
    geometry.export(f"{cfg.output_path}/final", "final", save_npy=True)

    if material is not None:
        material.export(f"{cfg.output_path}/final", "material")
        renderer.export(f"{cfg.output_path}/final", "material")

    wandb.save(f"{cfg.output_path}/final/*")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    args, extras = parser.parse_known_args()
    
    cfg_file = args.config
    cfg = load_config(cfg_file, cli_args=extras)
    
    train(cfg)
