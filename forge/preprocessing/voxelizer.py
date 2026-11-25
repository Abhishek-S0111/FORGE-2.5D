"""
Range-Adaptive Voxelizer for FORGE-2.5D (2-Ring Configuration)

Performs distance-dependent point cloud downsampling before feeding points into 
downstream segmentation and mapping networks, slashing input compute at the source.

Ring Configuration:
  - Ring 0: 0 <= d <= 10m  -> Voxel size (0.05m, 0.05m, 0.05m) [Fovea High Precision]
  - Ring 1: 10m < d <= 50m -> Voxel size (0.25m, 0.25m, 0.20m) [Far-Field Coarsened]
  - Beyond 50m             -> Pruned / Excluded
"""

import time
import numpy as np
from typing import Dict, Tuple, Optional


class RangeAdaptiveVoxelizer:
    def __init__(
        self,
        r0_max_dist: float = 10.0,
        r1_max_dist: float = 50.0,
        r0_voxel_size: Tuple[float, float, float] = (0.05, 0.05, 0.05),
        r1_voxel_size: Tuple[float, float, float] = (0.25, 0.25, 0.20),
        min_z: float = -3.0,
        max_z: float = 5.0,
    ):
        """
        Args:
            r0_max_dist: Outer radius of Ring 0 in meters (default: 10.0m)
            r1_max_dist: Outer radius of Ring 1 in meters (default: 50.0m)
            r0_voxel_size: (dx, dy, dz) voxel size for Ring 0 in meters (default: 5cm)
            r1_voxel_size: (dx, dy, dz) voxel size for Ring 1 in meters (default: 25cm x 25cm x 20cm)
            min_z: Minimum height bound relative to sensor origin
            max_z: Maximum height bound relative to sensor origin
        """
        self.r0_max_dist = r0_max_dist
        self.r1_max_dist = r1_max_dist
        self.r0_voxel_size = np.array(r0_voxel_size, dtype=np.float32)
        self.r1_voxel_size = np.array(r1_voxel_size, dtype=np.float32)
        self.min_z = min_z
        self.max_z = max_z

    def voxelize(
        self, points: np.ndarray, extra_features: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
        """
        Voxelize point cloud adaptively based on distance from sensor origin.

        Args:
            points: (N, 3) array of (x, y, z) coordinates in sensor frame.
            extra_features: Optional (N, D) array of additional attributes (intensity, timestamp, etc.)

        Returns:
            downsampled_points: (M, 3) decimated point coordinates.
            downsampled_features: (M, D) aggregated features or None.
            stats: Dictionary with timing, point counts, and reduction ratio.
        """
        t0 = time.perf_counter()
        n_raw = len(points)
        if n_raw == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                None,
                {"latency_ms": 0.0, "reduction_ratio": 0.0, "n_raw": 0, "n_out": 0},
            )

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        d_xy = np.sqrt(x * x + y * y)

        # 1. Height and distance bounds filter
        valid_mask = (z >= self.min_z) & (z <= self.max_z) & (d_xy <= self.r1_max_dist)
        filtered_pts = points[valid_mask]
        filtered_d = d_xy[valid_mask]
        filtered_feat = extra_features[valid_mask] if extra_features is not None else None

        # 2. Split into Ring 0 and Ring 1 masks
        mask_r0 = filtered_d <= self.r0_max_dist
        mask_r1 = ~mask_r0

        pts_r0 = filtered_pts[mask_r0]
        pts_r1 = filtered_pts[mask_r1]

        feat_r0 = filtered_feat[mask_r0] if filtered_feat is not None else None
        feat_r1 = filtered_feat[mask_r1] if filtered_feat is not None else None

        # 3. Voxel downsample Ring 0 (High resolution - 5cm)
        out_r0_pts, out_r0_feat = self._voxel_grid_filter(pts_r0, self.r0_voxel_size, feat_r0)

        # 4. Voxel downsample Ring 1 (Coarsened - 25cm)
        out_r1_pts, out_r1_feat = self._voxel_grid_filter(pts_r1, self.r1_voxel_size, feat_r1)

        # 5. Concatenate Ring 0 and Ring 1 points
        if len(out_r0_pts) > 0 and len(out_r1_pts) > 0:
            out_points = np.vstack([out_r0_pts, out_r1_pts])
            out_features = (
                np.vstack([out_r0_feat, out_r1_feat]) if filtered_feat is not None else None
            )
        elif len(out_r0_pts) > 0:
            out_points = out_r0_pts
            out_features = out_r0_feat
        else:
            out_points = out_r1_pts
            out_features = out_r1_feat

        latency_ms = (time.perf_counter() - t0) * 1000.0
        n_out = len(out_points)
        reduction_ratio = (1.0 - (n_out / max(1, n_raw))) * 100.0

        stats = {
            "latency_ms": latency_ms,
            "n_raw": n_raw,
            "n_filtered": len(filtered_pts),
            "n_r0_in": len(pts_r0),
            "n_r0_out": len(out_r0_pts),
            "n_r1_in": len(pts_r1),
            "n_r1_out": len(out_r1_pts),
            "n_out": n_out,
            "reduction_percent": reduction_ratio,
        }
        return out_points, out_features, stats

    def _voxel_grid_filter(
        self,
        pts: np.ndarray,
        voxel_size: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fast vectorized voxel grid centroid filtering using NumPy."""
        if len(pts) == 0:
            empty_feat = np.empty((0, features.shape[1]), dtype=np.float32) if features is not None else None
            return np.empty((0, 3), dtype=np.float32), empty_feat

        # Compute voxel integer coordinates
        voxel_coords = np.floor(pts / voxel_size).astype(np.int64)

        # Unique voxels with inverse mapping
        _, unique_indices, inverse_indices, counts = np.unique(
            voxel_coords, axis=0, return_index=True, return_inverse=True, return_counts=True
        )

        n_voxels = len(unique_indices)

        # Centroid accumulation
        sum_pts = np.zeros((n_voxels, 3), dtype=np.float64)
        np.add.at(sum_pts, inverse_indices, pts)
        centroid_pts = (sum_pts / counts[:, None]).astype(np.float32)

        out_features = None
        if features is not None:
            sum_feat = np.zeros((n_voxels, features.shape[1]), dtype=np.float64)
            np.add.at(sum_feat, inverse_indices, features)
            out_features = (sum_feat / counts[:, None]).astype(np.float32)

        return centroid_pts, out_features
