#!/usr/bin/env python3
"""
Use MASt3R to estimate camera poses from multi-view images.
Uses /project3 for model cache to avoid disk space issues.
"""

import os
# Set HuggingFace cache to /project3 before importing anything
os.environ['HF_HOME'] = '/project3/oceanic/.cache/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/project3/oceanic/.cache/huggingface/hub'

import sys
sys.path.insert(0, '/project3/oceanic/exp/mast3r')

import numpy as np
import torch
import pickle
import json
import tempfile
from pathlib import Path
import argparse

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.image_pairs import make_pairs


def calibrate_cameras_mast3r(data_dir, frame_idx=0, device='cuda'):
    """Estimate camera extrinsics using MASt3R."""
    
    print(f"Model cache directory: {os.environ['HF_HOME']}")
    print(f"Loading MASt3R model...")
    
    if torch.cuda.is_available() and device == 'cuda':
        device = 'cuda'
    else:
        device = 'cpu'
    
    model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    ).to(device)
    
    print(f"✓ Model loaded on {device}")
    print(f"Loading images from frame {frame_idx}...")
    
    # Prepare image paths
    img_paths = [f"{data_dir}/color/{i}/{frame_idx}.png" for i in [0, 1]]
    
    # Load images in dust3r format
    imgs = load_images(img_paths, size=512, verbose=True)
    print(f"Loaded {len(imgs)} images")
    
    # Create pairs for matching (bidirectional)
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    print(f"Created {len(pairs)} image pairs for matching")
    
    # Create temporary cache directory
    cache_dir = tempfile.mkdtemp(suffix='_mast3r_cache', dir='/project3/oceanic/exp')
    
    print("\nRunning sparse global alignment...")
    print("This will jointly optimize camera poses and 3D structure...")
    
    # Run sparse global alignment
    scene = sparse_global_alignment(
        img_paths,
        pairs,
        cache_dir,
        model,
        lr1=0.07,
        niter1=500,
        lr2=0.014,
        niter2=200,
        device=device,
        opt_depth=True,
        shared_intrinsics=False,
        matching_conf_thr=5.0,
    )
    
    print("\n✓ Alignment completed!")
    
    # Extract camera poses (camera-to-world matrices)
    c2w_matrices = scene.get_im_poses().cpu().numpy()
    focals = scene.get_focals().cpu().numpy()
    
    print("\nEstimated camera parameters:")
    for i in range(len(imgs)):
        c2w = c2w_matrices[i]
        focal = focals[i]
        pos = c2w[:3, 3]
        print(f"  Camera {i}:")
        print(f"    Position: {pos}")
        print(f"    Focal: {focal:.2f}px")
    
    # Normalize: Set camera 0 as world origin
    c2w_normalized = c2w_matrices.copy()
    cam0_w2c = np.linalg.inv(c2w_matrices[0])
    for i in range(len(c2w_normalized)):
        c2w_normalized[i] = cam0_w2c @ c2w_matrices[i]
    
    print("\nNormalized camera poses (camera 0 at origin):")
    for i, c2w in enumerate(c2w_normalized):
        pos = c2w[:3, 3]
        print(f"  Camera {i}: {pos}")
    
    baseline = np.linalg.norm(c2w_normalized[1][:3, 3])
    print(f"\nBaseline distance: {baseline:.4f}m ({baseline*1000:.1f}mm)")

    # Extract sparse 3D points and colors
    print("\nExtracting sparse 3D point cloud...")
    pts3d_sparse = scene.get_sparse_pts3d()  # List of tensors
    pts3d_colors = scene.pts3d_colors  # List of numpy arrays

    # Convert to numpy arrays
    pts3d_list = []
    colors_list = []
    for i in range(len(pts3d_sparse)):
        pts = pts3d_sparse[i].cpu().numpy()  # (N, 3)
        colors = pts3d_colors[i]  # (N, 3)
        pts3d_list.append(pts)
        colors_list.append(colors)
        print(f"  Camera {i}: {len(pts)} points")

    # Cleanup cache
    import shutil
    shutil.rmtree(cache_dir)

    return c2w_normalized, focals, pts3d_list, colors_list


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate camera extrinsics using MASt3R"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Data directory (e.g., data_ours/different_types/Bread_tearing)")
    parser.add_argument("--frame", type=int, default=0,
                       help="Frame index (default: 0)")
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'],
                       help="Device (default: cuda)")
    
    args = parser.parse_args()

    result = calibrate_cameras_mast3r(args.data_dir, args.frame, args.device)

    if result is None:
        return

    c2w_matrices, focals, pts3d_list, colors_list = result

    # Save calibration results
    calib_file = f"{args.data_dir}/calibrate.pkl"
    backup_file = f"{args.data_dir}/calibrate_backup.pkl"

    import shutil
    if Path(calib_file).exists():
        shutil.copy(calib_file, backup_file)
        print(f"\nBacked up original to: {backup_file}")

    with open(calib_file, 'wb') as f:
        pickle.dump(c2w_matrices, f)

    print(f"\n✓ Saved camera extrinsics to: {calib_file}")

    # Save point cloud data
    pcd_file = f"{args.data_dir}/mast3r_pcd_{args.frame}.pkl"
    pcd_data = {
        'c2w': c2w_matrices,
        'focals': focals,
        'points': pts3d_list,
        'colors': colors_list,
        'frame': args.frame
    }
    with open(pcd_file, 'wb') as f:
        pickle.dump(pcd_data, f)

    print(f"✓ Saved MASt3R point cloud to: {pcd_file}")

    # Also save as PLY files for visualization
    try:
        import open3d as o3d
        for i, (pts, colors) in enumerate(zip(pts3d_list, colors_list)):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            ply_file = f"{args.data_dir}/mast3r_cam{i}_frame{args.frame}.ply"
            o3d.io.write_point_cloud(ply_file, pcd)
            print(f"✓ Saved camera {i} point cloud to: {ply_file}")
    except:
        print("  (Open3D not available, skipping PLY export)")

    print("\nNext: Run PhysTwin with MASt3R calibration to generate aligned point clouds")


if __name__ == "__main__":
    main()
