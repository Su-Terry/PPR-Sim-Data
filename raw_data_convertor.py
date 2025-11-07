#!/usr/bin/env python3
"""
Raw Data Converter: Convert RealSense MOV files to data_ours format

This script converts raw RealSense camera data (MOV files) to the processed
data_ours format used by the SGA project.

Expected raw data structure:
raw_data/
├── case_name/
│   ├── Case_D435f_Master_RGB.MOV
│   ├── Case_D435f_Master_Depth_filter.MOV  
│   ├── Case_D435f_Master_Depth_origin.MOV
│   ├── Case_D435_slave_RGB.MOV
│   ├── Case_D435_slave_Depth_filter.MOV
│   └── Case_D435_slave_Depth_origin.MOV

Output structure:
data_ours/different_types/case_name/
├── calibrate.pkl
├── metadata.json
├── split.json
├── color/
│   ├── 0/     # Master camera
│   ├── 1/     # Slave camera 
│   ├── 0.mp4
│   └── 1.mp4 
├── depth/
│   ├── 0/     # Master camera
│   ├── 1/     # Slave camera
│   ├── 0.mp4
└── └── 1.mp4 
"""

import os
import sys
import json
import pickle
import argparse
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

class RawDataConverter:
    def __init__(self, cameras_dir: str = "/project3/oceanic/exp/data_ours/cameras"):
        """
        Initialize converter with camera calibration directory.

        Args:
            cameras_dir: Directory containing scene-specific camera calibration files
        """
        self.cameras_dir = cameras_dir

        # Actual calibrated intrinsics for RealSense D435 (default fallback)
        self.default_intrinsics = {
            'master': [
                [609.213254, 0.0, 307.25896107],
                [0.0, 605.07951874, 256.26322148],
                [0.0, 0.0, 1.0]
            ],
            'slave': [
                [599.29822085, 0.0, 314.58617871],
                [0.0, 599.98596418, 254.39472826],
                [0.0, 0.0, 1.0]
            ]
        }

        # Distortion coefficients for both cameras (default fallback)
        self.default_dist_coeffs = {
            'master': [0.06220872, 0.33776926, 0.00522679, -0.01301685, -1.21256597],
            'slave': [1.57391866e-02, 7.12997452e-01, 1.17358377e-03, 6.43605530e-04, -2.67972926e+00]
        }

        # Actual calibrated extrinsics (default fallback)
        # Master camera extrinsics (identity - reference frame)
        # NOTE: Translation values are converted from mm to meters (divided by 1000)
        # The default values are slave-to-master, so we invert to get slave c2w
        slave_to_master_default = np.array([
            [0.94953572, 0.09413182, 0.2992008, 0.32768354],  # 327.6835361 mm -> 0.327 m
            [-0.18014379, 0.94455272, 0.27453302, 0.00054586],  # 0.54586192 mm -> 0.0005 m
            [-0.25676863, -0.31457807, 0.91384381, 0.07381379],  # 73.81378764 mm -> 0.074 m
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.default_extrinsics = {
            'master': np.eye(4),
            'slave': np.linalg.inv(slave_to_master_default)
        }

        # Current scene-specific parameters (will be loaded per scene)
        self.current_intrinsics = None
        self.current_dist_coeffs = None
        self.current_extrinsics = None

        self.fps = 30
        # Image size will be determined from actual frames
        self.image_size = None

    def extract_material_from_scene_name(self, scene_name: str) -> str:
        """
        Extract material type from scene name.

        Examples:
            Bread_squeezing -> bread
            Clay_pull_squeeze -> clay
            Pudding_tapping -> pudding

        Args:
            scene_name: Full scene name

        Returns:
            Material name in lowercase
        """
        # Split by underscore and take first part
        material = scene_name.split('_')[0].lower()
        return material

    def parse_camera_parameters(self, camera_file: str) -> Tuple[Dict, Dict, Dict]:
        """
        Parse camera parameters from scene-specific text file.

        Args:
            camera_file: Path to camera parameters file

        Returns:
            Tuple of (intrinsics, dist_coeffs, extrinsics) dictionaries
        """
        if not os.path.exists(camera_file):
            print(f"Camera file not found: {camera_file}, using defaults")
            return None, None, None

        print(f"Loading camera parameters from: {camera_file}")

        intrinsics = {}
        dist_coeffs = {}
        extrinsics = {}

        try:
            with open(camera_file, 'r') as f:
                content = f.read()

            # Parse intrinsics
            # Look for "master" section
            if 'master' in content.lower():
                import re
                # Find master intrinsics matrix
                master_match = re.search(r'master\s*\n\s*\[\[([^\]]+)\]\s*\n\s*\[([^\]]+)\]\s*\n\s*\[([^\]]+)\]\]', content)
                if master_match:
                    row1 = [float(x.strip()) for x in master_match.group(1).split()]
                    row2 = [float(x.strip()) for x in master_match.group(2).split()]
                    row3 = [float(x.strip()) for x in master_match.group(3).split()]
                    intrinsics['master'] = [row1, row2, row3]
                    print(f"  ✓ Loaded master intrinsics")

            # Look for slave D435 intrinsics
            slave_d435_match = re.search(r'D435 slave\s*\n\s*\[\[([^\]]+)\]\s*\n\s*\[([^\]]+)\]\s*\n\s*\[([^\]]+)\]\]', content)
            if slave_d435_match:
                row1 = [float(x.strip()) for x in slave_d435_match.group(1).split()]
                row2 = [float(x.strip()) for x in slave_d435_match.group(2).split()]
                row3 = [float(x.strip()) for x in slave_d435_match.group(3).split()]
                intrinsics['slave'] = [row1, row2, row3]
                print(f"  ✓ Loaded slave D435 intrinsics")

            # Parse extrinsics (masterD435F / slaveD435)
            # Look for rotation matrix and translation vector
            master_slave_match = re.search(r'masterD435F / slaveD435\s*\n\s*Rotation matrix \(R\):\s*\n\s*\[\[([^\]]+)\]\s*\n\s*\[([^\]]+)\]\s*\n\s*\[([^\]]+)\]\]\s*\n\s*T=np\.array\(\[\s*\n\s*\[([^\]]+)\],\s*\n\s*\[([^\]]+)\],\s*\n\s*\[([^\]]+)\]', content)

            if master_slave_match:
                # Parse rotation matrix
                R = np.array([
                    [float(x.strip()) for x in master_slave_match.group(1).split()],
                    [float(x.strip()) for x in master_slave_match.group(2).split()],
                    [float(x.strip()) for x in master_slave_match.group(3).split()]
                ])
                # Parse translation vector (in mm, convert to meters)
                T = np.array([
                    float(master_slave_match.group(4).strip()),
                    float(master_slave_match.group(5).strip()),
                    float(master_slave_match.group(6).strip())
                ]) / 1000.0  # Convert mm to meters

                # Build 4x4 transformation matrix
                # "masterD435F / slaveD435" with R,T typically means transformation from slave to master
                # i.e., point_master = R * point_slave + T
                # To get c2w (camera-to-world) where master is world origin:
                # - Master c2w = Identity
                # - Slave c2w = inverse of [R|T]
                extrinsics['master'] = np.eye(4)

                # Build slave-to-master transform
                slave_to_master = np.eye(4)
                slave_to_master[:3, :3] = R
                slave_to_master[:3, 3] = T

                # Invert to get slave c2w
                extrinsics['slave'] = np.linalg.inv(slave_to_master)

                print(f"  ✓ Loaded extrinsics (slave-to-master inverted to c2w, translation converted mm->m)")

            # Fallback to defaults if parsing failed
            if not intrinsics:
                print("  ⚠ Failed to parse intrinsics, using defaults")
                return None, None, None
            if not extrinsics:
                print("  ⚠ Failed to parse extrinsics, using defaults")
                return intrinsics, dist_coeffs, None

            return intrinsics, dist_coeffs, extrinsics

        except Exception as e:
            print(f"  ⚠ Error parsing camera file: {e}")
            return None, None, None

    def load_scene_camera_parameters(self, scene_name: str) -> bool:
        """
        Load camera parameters for a specific scene.

        Args:
            scene_name: Scene name (e.g., "Bread_squeezing")

        Returns:
            True if parameters loaded successfully, False if using hardcoded defaults
        """
        # Extract material type from scene name
        material = self.extract_material_from_scene_name(scene_name)

        # Build camera file path
        camera_file = os.path.join(self.cameras_dir, f"{material}.txt")

        # Try loading scene-specific parameters
        intrinsics, dist_coeffs, extrinsics = self.parse_camera_parameters(camera_file)

        if intrinsics is not None:
            self.current_intrinsics = intrinsics
            self.current_dist_coeffs = dist_coeffs if dist_coeffs else self.default_dist_coeffs
            self.current_extrinsics = extrinsics if extrinsics else self.default_extrinsics
            print(f"✓ Using scene-specific camera parameters for material: {material}")
            return True
        else:
            # Try loading from default.txt
            default_file = os.path.join(self.cameras_dir, "default.txt")
            intrinsics, dist_coeffs, extrinsics = self.parse_camera_parameters(default_file)

            if intrinsics is not None:
                self.current_intrinsics = intrinsics
                self.current_dist_coeffs = dist_coeffs if dist_coeffs else self.default_dist_coeffs
                self.current_extrinsics = extrinsics if extrinsics else self.default_extrinsics
                print(f"✓ Using default camera parameters from default.txt (no {material}.txt found)")
                return True
            else:
                # Final fallback to hardcoded defaults
                self.current_intrinsics = self.default_intrinsics
                self.current_dist_coeffs = self.default_dist_coeffs
                self.current_extrinsics = self.default_extrinsics
                print(f"⚠ Using hardcoded default camera parameters (no txt files found)")
                return False

    def find_mov_files(self, raw_case_dir: str) -> Dict[str, str]:
        """Find and categorize MOV files in the raw case directory."""
        mov_files = {}
        
        for file in os.listdir(raw_case_dir):
            if not file.endswith('.MOV'):
                continue
                
            file_path = os.path.join(raw_case_dir, file)
            file_lower = file.lower()
            
            # Categorize files based on naming convention
            if 'd435f_master' in file_lower:
                if 'rgb' in file_lower:
                    mov_files['master_rgb'] = file_path
                elif 'depth_filter' in file_lower:
                    mov_files['master_depth'] = file_path
            elif 'd435_slave' in file_lower:
                if 'rgb' in file_lower:
                    mov_files['slave_rgb'] = file_path
                elif 'depth_filter' in file_lower:
                    mov_files['slave_depth'] = file_path
                    
        return mov_files

    def load_calibration_matrices(self, calibration_file: str) -> Optional[np.ndarray]:
        """Load calibration matrices from external file.
        
        Args:
            calibration_file: Path to calibration file (.pkl, .npy, or .json)
            
        Returns:
            Loaded calibration matrices or None if file doesn't exist/can't be loaded
        """
        if not os.path.exists(calibration_file):
            print(f"Calibration file not found: {calibration_file}")
            return None
            
        try:
            if calibration_file.endswith('.pkl'):
                with open(calibration_file, 'rb') as f:
                    matrices = pickle.load(f)
            elif calibration_file.endswith('.npy'):
                matrices = np.load(calibration_file)
            elif calibration_file.endswith('.json'):
                with open(calibration_file, 'r') as f:
                    data = json.load(f)
                    # Assume the matrices are stored under 'poses' or 'calibration' key
                    matrices = np.array(data.get('poses', data.get('calibration', data)))
            else:
                print(f"Unsupported calibration file format: {calibration_file}")
                return None
                
            print(f"Loaded calibration matrices from {calibration_file}, shape: {matrices.shape}")
            return np.array(matrices)
            
        except Exception as e:
            print(f"Error loading calibration file {calibration_file}: {str(e)}")
            return None

    def extract_frames(self, video_path: str, output_dir: str, is_depth: bool = False) -> Tuple[int, Tuple[int, int]]:
        """Extract frames from a MOV file and return frame count and dimensions."""
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_dims = None
        
        print(f"Extracting {total_frames} frames from {os.path.basename(video_path)}")
        
        with tqdm(total=total_frames, desc=f"Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get dimensions from first frame
                if frame_dims is None:
                    height, width = frame.shape[:2]
                    frame_dims = (width, height)  # Store as (W, H)
                    print(f"Frame dimensions: {width}x{height}")
                    
                if is_depth:
                    # RealSense depth MOV encodes 16-bit depth in color channels
                    # Typically: Red channel = high byte, Green channel = low byte
                    if len(frame.shape) == 3:
                        # Extract depth from BGR channels
                        # RealSense format: depth = (R << 8) | G
                        b, g, r = cv2.split(frame)
                        frame = (r.astype(np.uint16) << 8) | g.astype(np.uint16)
                    else:
                        # Already grayscale, just convert to uint16
                        frame = frame.astype(np.uint16)

                    # Save depth as NPY (in millimeters, uint16)
                    frame_path = os.path.join(output_dir, f"{frame_count}.npy")
                    np.save(frame_path, frame)
                else:
                    # For RGB images, save directly (cv2.imwrite expects BGR format)
                    frame_path = os.path.join(output_dir, f"{frame_count}.png")
                    cv2.imwrite(frame_path, frame)
                
                frame_count += 1
                pbar.update(1)
                
        cap.release()
        return frame_count, frame_dims

    def copy_original_videos(self, mov_files: Dict[str, str], output_case_dir: str):
        """Copy original MOV files with standardized naming."""
        print("Copying original MOV files...")
        
        # Copy RGB videos to color directory
        if 'master_rgb' in mov_files:
            dest_path = os.path.join(output_case_dir, "color", "0.mp4")
            shutil.copy2(mov_files['master_rgb'], dest_path)
            print(f"  ✓ Copied master RGB as color/0.mp4")
            
        if 'slave_rgb' in mov_files:
            dest_path = os.path.join(output_case_dir, "color", "1.mp4")
            shutil.copy2(mov_files['slave_rgb'], dest_path)
            print(f"  ✓ Copied slave RGB as color/1.mp4")
        
        # Copy Depth videos to depth directory  
        if 'master_depth' in mov_files:
            dest_path = os.path.join(output_case_dir, "depth", "0.mp4")
            shutil.copy2(mov_files['master_depth'], dest_path)
            print(f"  ✓ Copied master Depth as depth/0.mp4")
            
        if 'slave_depth' in mov_files:
            dest_path = os.path.join(output_case_dir, "depth", "1.mp4")
            shutil.copy2(mov_files['slave_depth'], dest_path)
            print(f"  ✓ Copied slave Depth as depth/1.mp4")

    def create_calibration_data(self, frame_count: int, pose_matrices: Optional[np.ndarray] = None, pose_format: str = 'w2c') -> np.ndarray:
        """Create camera calibration matrices.

        Args:
            frame_count: Number of frames (unused - kept for compatibility)
            pose_matrices: Optional pose matrices to use
            pose_format: Format of input matrices ('w2c' or 'c2w')

        Returns:
            Camera-to-world transformation matrices for each camera (shape: [num_cameras, 4, 4])
            NOT per-frame matrices! PhysTwin expects per-camera extrinsics.
        """
        # Use current scene-specific extrinsics if loaded, otherwise use defaults
        extrinsics_to_use = self.current_extrinsics if self.current_extrinsics else self.default_extrinsics

        # Create per-camera matrices (not per-frame!)
        # PhysTwin expects shape (num_cameras, 4, 4)
        c2w_matrices = [
            extrinsics_to_use['master'].copy(),
            extrinsics_to_use['slave'].copy()
        ]

        # NOTE: data_process_pcd.py's getPcdFromDepth applies complex coordinate flips.
        # The extrinsics from calibration might already account for this.
        # Return c2w matrices as-is for now.
        return np.array(c2w_matrices)

    def create_metadata(self, frame_count: int, has_slave: bool = True) -> Dict:
        """Create metadata JSON using current scene-specific parameters."""
        # Use current scene-specific parameters if loaded, otherwise use defaults
        intrinsics_to_use = self.current_intrinsics if self.current_intrinsics else self.default_intrinsics
        dist_coeffs_to_use = self.current_dist_coeffs if self.current_dist_coeffs else self.default_dist_coeffs

        intrinsics = [intrinsics_to_use['master']]
        serial_numbers = ["239222300433"]  # Default serial number

        if has_slave:
            intrinsics.append(intrinsics_to_use['slave'])
            serial_numbers.append("239222300781")

        # Include distortion coefficients in metadata
        dist_coeffs = [dist_coeffs_to_use['master']]
        if has_slave:
            dist_coeffs.append(dist_coeffs_to_use['slave'])

        metadata = {
            "intrinsics": intrinsics,
            "distortion_coefficients": dist_coeffs,
            "serial_numbers": serial_numbers,
            "fps": self.fps,
            "WH": self.image_size,
            "frame_num": frame_count,
            "start_step": 0,
            "end_step": frame_count - 1
        }

        return metadata

    def create_split_data(self, frame_count: int) -> Dict:
        """Create train/test split."""
        train_split = int(frame_count * 0.7)
        
        split_data = {
            "frame_len": frame_count,
            "train": [0, train_split],
            "test": [train_split, frame_count]
        }
        
        return split_data

    def convert_case(self, raw_case_dir: str, output_case_dir: str, case_name: str, calibration_file: str = None, pose_format: str = 'w2c'):
        """Convert a single case from raw format to data_ours format."""
        print(f"\nConverting case: {case_name}")
        print("="*60)

        # Load scene-specific camera parameters first
        self.load_scene_camera_parameters(case_name)

        # Load calibration matrices if provided
        pose_matrices = None
        if calibration_file:
            pose_matrices = self.load_calibration_matrices(calibration_file)
            if pose_matrices is not None:
                print(f"Loaded {len(pose_matrices)} {pose_format} calibration matrices")
        
        # Find MOV files
        mov_files = self.find_mov_files(raw_case_dir)
        
        if not mov_files:
            print(f"No MOV files found in {raw_case_dir}")
            return
            
        print(f"Found files: {list(mov_files.keys())}")
        
        # Create output directories
        os.makedirs(output_case_dir, exist_ok=True)
        os.makedirs(os.path.join(output_case_dir, "color", "0"), exist_ok=True)
        os.makedirs(os.path.join(output_case_dir, "depth", "0"), exist_ok=True)
        
        has_slave = 'slave_rgb' in mov_files
        if has_slave:
            os.makedirs(os.path.join(output_case_dir, "color", "1"), exist_ok=True)
            os.makedirs(os.path.join(output_case_dir, "depth", "1"), exist_ok=True)
        
        # Extract frames and detect dimensions
        frame_count = 0
        actual_image_size = None
        
        # Master camera
        if 'master_rgb' in mov_files:
            frame_count, actual_image_size = self.extract_frames(
                mov_files['master_rgb'],
                os.path.join(output_case_dir, "color", "0"),
                is_depth=False
            )
            
        if 'master_depth' in mov_files:
            depth_count, depth_size = self.extract_frames(
                mov_files['master_depth'],
                os.path.join(output_case_dir, "depth", "0"),
                is_depth=True
            )
            # Use depth size if RGB size not available
            if actual_image_size is None:
                actual_image_size = depth_size
            
        # Slave camera
        if has_slave:
            if 'slave_rgb' in mov_files:
                slave_count, slave_size = self.extract_frames(
                    mov_files['slave_rgb'],
                    os.path.join(output_case_dir, "color", "1"),
                    is_depth=False
                )
                # Use slave size if master size not available
                if actual_image_size is None:
                    actual_image_size = slave_size
                
            if 'slave_depth' in mov_files:
                self.extract_frames(
                    mov_files['slave_depth'],
                    os.path.join(output_case_dir, "depth", "1"),
                    is_depth=True
                )
        
        # Set the detected image size
        if actual_image_size:
            self.image_size = list(actual_image_size)
            print(f"Detected image size: {self.image_size} (WxH)")
        else:
            # Fallback to default
            self.image_size = [640, 480]
            print("Warning: Could not detect image size, using default [640, 480]")
        
        # Copy original MOV files
        self.copy_original_videos(mov_files, output_case_dir)
        
        # Create calibration data
        c2w_matrices = self.create_calibration_data(frame_count, pose_matrices, pose_format)
        with open(os.path.join(output_case_dir, "calibrate.pkl"), "wb") as f:
            pickle.dump(c2w_matrices, f)
            
        # Log the conversion
        if pose_matrices is not None:
            if pose_format == 'w2c':
                print(f"  ✓ Converted {len(pose_matrices)} w2c matrices to c2w format")
            else:
                print(f"  ✓ Used {len(pose_matrices)} c2w matrices directly")
        else:
            # Note: Even without external calibration file, scene-specific parameters may be loaded
            if self.current_extrinsics is not None:
                print(f"  ✓ Using scene-specific camera calibration (from cameras/{self.extract_material_from_scene_name(case_name)}.txt)")
            else:
                print(f"  ✓ Using hardcoded default calibration matrices")
            
        # Create metadata
        metadata = self.create_metadata(frame_count, has_slave)
        with open(os.path.join(output_case_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Create split data
        split_data = self.create_split_data(frame_count)
        with open(os.path.join(output_case_dir, "split.json"), "w") as f:
            json.dump(split_data, f, indent=2)
            
        # Create placeholder tracking data (you may need actual tracking)
        placeholder_tracking = {
            "frame_count": frame_count,
            "placeholder": True,
            "note": "This is placeholder tracking data. Replace with actual tracking results."
        }
        with open(os.path.join(output_case_dir, "track_process_data.pkl"), "wb") as f:
            pickle.dump(placeholder_tracking, f)
            
        print(f"✓ Successfully converted {case_name}")
        print(f"  - Frames: {frame_count}")
        print(f"  - Cameras: {'2 (master + slave)' if has_slave else '1 (master only)'}")

def main():
    parser = argparse.ArgumentParser(description="Convert raw RealSense data to data_ours format")
    parser.add_argument("--raw_base_path", type=str, default="/project3/oceanic/exp/data_ours/raw",
                       help="Base path containing raw data")
    parser.add_argument("--output_base_path", type=str, default="/project3/oceanic/exp/data_ours/different_types",
                       help="Output base path")
    parser.add_argument("--cameras_dir", type=str, default="/project3/oceanic/exp/data_ours/cameras",
                       help="Directory containing scene-specific camera calibration files")
    parser.add_argument("--case_name", type=str, help="Specific case to convert (optional)")
    parser.add_argument("--intrinsics_file", type=str,
                       help="Path to camera intrinsics file (optional)")
    parser.add_argument("--w2c_calibration_file", type=str,
                       help="Path to world-to-camera calibration matrices file (.pkl, .npy, or .json)")
    parser.add_argument("--pose_format", type=str, choices=['w2c', 'c2w'], default='c2w',
                       help="Format of input pose matrices (default: c2w)")

    args = parser.parse_args()

    converter = RawDataConverter(cameras_dir=args.cameras_dir)
    
    # Load custom intrinsics if provided
    if args.intrinsics_file and os.path.exists(args.intrinsics_file):
        print(f"Loading intrinsics from: {args.intrinsics_file}")
        # Add code to load intrinsics from file (depends on format)
    
    if args.case_name:
        # Convert specific case
        raw_case_dir = os.path.join(args.raw_base_path, args.case_name)
        output_case_dir = os.path.join(args.output_base_path, args.case_name)
        
        if not os.path.exists(raw_case_dir):
            print(f"Raw case directory not found: {raw_case_dir}")
            return
            
        converter.convert_case(raw_case_dir, output_case_dir, args.case_name, args.w2c_calibration_file, args.pose_format)
    else:
        # Convert all cases
        if not os.path.exists(args.raw_base_path):
            print(f"Raw base path not found: {args.raw_base_path}")
            return
            
        for case_name in os.listdir(args.raw_base_path):
            raw_case_dir = os.path.join(args.raw_base_path, case_name)
            
            if not os.path.isdir(raw_case_dir):
                continue
                
            output_case_dir = os.path.join(args.output_base_path, case_name)
            
            try:
                converter.convert_case(raw_case_dir, output_case_dir, case_name, args.w2c_calibration_file, args.pose_format)
            except Exception as e:
                print(f"Error converting {case_name}: {str(e)}")
                continue
    
    print("\nConversion completed!")

if __name__ == "__main__":
    main()