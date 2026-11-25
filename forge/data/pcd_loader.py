"""
PCD File Loader for IITH LiDAR Ground Dataset & Custom Point Clouds

Supports ASCII and binary PCD files with XYZ and RGB / intensity fields.
Ground truth color encoding in IITH dataset:
  - Red (r=255, g=0, b=0): Ground / Drivable Surface (Class 0)
  - Green (r=0, g=255, b=0): Non-ground / Obstacles
"""

import os
import re
import numpy as np
from typing import Tuple, Dict, List, Optional


class PCDLoader:
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            # Look for data/ at project root
            # __file__ is in forge/data/pcd_loader.py -> dirname is forge/data -> dirname is forge -> dirname is project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.data_dir = os.path.join(project_root, "data")
        else:
            self.data_dir = data_dir

        self.labelled_dir = os.path.join(self.data_dir, "labelled_data")
        self.raw_dir = os.path.join(self.data_dir, "raw_data")

    def list_available_scans(self) -> List[Dict[str, str]]:
        """List all available labelled and raw PCD files in the data directory."""
        scans = []
        if os.path.exists(self.labelled_dir):
            for fname in sorted(os.listdir(self.labelled_dir)):
                if fname.endswith(".pcd"):
                    scans.append({
                        "name": fname,
                        "type": "labelled",
                        "path": os.path.join(self.labelled_dir, fname),
                    })
        if os.path.exists(self.raw_dir):
            for fname in sorted(os.listdir(self.raw_dir)):
                if fname.endswith(".pcd"):
                    scans.append({
                        "name": fname,
                        "type": "raw",
                        "path": os.path.join(self.raw_dir, fname),
                    })
        return scans

    def get_categorized_scans(self) -> Dict[str, List[Dict[str, str]]]:
        """Return scans grouped cleanly by category: 'labelled', 'raw', 'synthetic'."""
        categorized = {
            "labelled": [],
            "raw": [],
            "synthetic": [{"name": "synthetic", "label": "Synthetic 64-Beam Urban Scene", "type": "synthetic"}]
        }
        if os.path.exists(self.labelled_dir):
            for fname in sorted(os.listdir(self.labelled_dir)):
                if fname.endswith(".pcd"):
                    categorized["labelled"].append({
                        "name": fname,
                        "label": f"IITH Labelled: {fname}",
                        "type": "labelled",
                        "path": os.path.join(self.labelled_dir, fname),
                    })
        if os.path.exists(self.raw_dir):
            for fname in sorted(os.listdir(self.raw_dir)):
                if fname.endswith(".pcd"):
                    categorized["raw"].append({
                        "name": fname,
                        "label": f"IITH Raw Scan: {fname}",
                        "type": "raw",
                        "path": os.path.join(self.raw_dir, fname),
                    })
        return categorized

    def load_pcd(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
        """
        Load a PCD file and extract coordinates and semantic labels.

        Returns:
            points: (N, 4) array [x, y, z, intensity/rgb_packed].
            labels: (N,) semantic labels in {0, 1, 2, 3}.
            header: Header metadata dictionary.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PCD file not found at: {file_path}")

        header = {}
        data_format = "ascii"
        header_lines = 0

        with open(file_path, "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                header_lines += 1
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                key = parts[0].upper()
                if key == "DATA":
                    data_format = parts[1].lower()
                    break
                elif key == "FIELDS":
                    header["fields"] = [p.lower() for p in parts[1:]]
                elif key == "POINTS":
                    header["num_points"] = int(parts[1])

        if data_format != "ascii":
            raise NotImplementedError(f"Binary PCD loading not yet supported for format: {data_format}")

        # Fast loading of ASCII data using numpy
        raw_data = np.loadtxt(file_path, skiprows=header_lines, dtype=np.float64)
        if len(raw_data.shape) == 1:
            raw_data = raw_data.reshape(1, -1)

        xyz = raw_data[:, :3].astype(np.float32)
        n_points = len(xyz)

        # Check fields
        fields = header.get("fields", ["x", "y", "z"])
        labels = np.zeros(n_points, dtype=np.uint8)
        intensity = np.full((n_points, 1), 0.5, dtype=np.float32)

        if "rgb" in fields:
            rgb_idx = fields.index("rgb")
            rgb_packed = raw_data[:, rgb_idx].astype(np.uint64)

            # Unpack RGB: 0xRRGGBB
            r = ((rgb_packed >> 16) & 0xFF).astype(np.uint8)
            g = ((rgb_packed >> 8) & 0xFF).astype(np.uint8)
            b = (rgb_packed & 0xFF).astype(np.uint8)

            # In IITH ground truth:
            # Red (r=255, g=0, b=0) -> Ground / Drivable Surface (Class 0)
            # Green (r=0, g=255, b=0) -> Non-ground (Obstacles)
            # For non-ground, we refine into Static vs Dynamic using height & cluster rules:
            is_ground = (r > 150) & (g < 100)
            is_non_ground = ~is_ground

            labels[is_ground] = 0  # DRIVABLE_SURFACE

            # Non-ground points:
            # Low non-ground (rough shoulder / curbs / small dips): rel_z <= 0.35m
            # High non-ground (poles / trees / vehicles): rel_z > 0.35m
            rel_z = xyz[:, 2] - (-1.65)
            non_ground_rough = is_non_ground & (rel_z <= 0.35)
            non_ground_static = is_non_ground & (rel_z > 0.35) & (np.abs(xyz[:, 1]) > 3.0)
            non_ground_dynamic = is_non_ground & (rel_z > 0.35) & (~non_ground_static)

            labels[non_ground_rough] = 1   # NON_DRIVABLE_TERRAIN
            labels[non_ground_static] = 2  # STATIC_OBSTACLE
            labels[non_ground_dynamic] = 3 # DYNAMIC_OBJECT

            intensity = (r.astype(np.float32) / 255.0)[:, None]
        else:
            # Raw point cloud without RGB: apply height heuristic
            rel_z = xyz[:, 2] - (-1.65)
            ground = np.abs(rel_z) <= 0.20
            labels[ground] = 0
            labels[~ground] = 2

        points_with_feat = np.hstack([xyz, intensity]).astype(np.float32)

        header["counts"] = {
            "total": n_points,
            "drivable": int(np.sum(labels == 0)),
            "nondrivable": int(np.sum(labels == 1)),
            "static": int(np.sum(labels == 2)),
            "dynamic": int(np.sum(labels == 3)),
        }
        return points_with_feat, labels, header
