"""
Unit tests for FORGE-2.5D Rapid Prototype Components
"""

import unittest
import numpy as np
import sys
import os

# Add root directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.preprocessing.voxelizer import RangeAdaptiveVoxelizer
from forge.mapping.kalman_grid import FoveatedGrid25D
from forge.models.segmenter import SemanticSegmenter
from forge.data.synthetic_generator import SyntheticLidarGenerator


class TestForgePrototype(unittest.TestCase):
    def setUp(self):
        self.generator = SyntheticLidarGenerator()
        self.voxelizer = RangeAdaptiveVoxelizer(
            r0_max_dist=10.0,
            r1_max_dist=50.0,
            r0_voxel_size=(0.05, 0.05, 0.05),
            r1_voxel_size=(0.25, 0.25, 0.20),
        )
        self.segmenter = SemanticSegmenter(device="cpu")
        self.grid = FoveatedGrid25D(
            r0_radius=10.0,
            r1_radius=50.0,
            r0_cell_size=0.05,
            r1_cell_size=0.25,
        )

    def test_synthetic_generator(self):
        pts, labels, meta = self.generator.generate_scene()
        self.assertGreater(len(pts), 5000)
        self.assertEqual(len(pts), len(labels))
        self.assertEqual(pts.shape[1], 4)
        self.assertTrue(set(labels).issubset({0, 1, 2, 3}))

    def test_range_adaptive_voxelizer(self):
        pts, _, _ = self.generator.generate_scene()
        downsampled, _, stats = self.voxelizer.voxelize(pts[:, :3])

        self.assertLess(len(downsampled), len(pts))
        self.assertGreater(stats["reduction_percent"], 20.0)
        self.assertLessEqual(np.max(np.sqrt(downsampled[:, 0]**2 + downsampled[:, 1]**2)), 50.1)

    def test_semantic_segmenter(self):
        pts, _, _ = self.generator.generate_scene()
        pred_labels, stats = self.segmenter.predict(pts[:, :3])

        self.assertEqual(len(pred_labels), len(pts))
        self.assertTrue(set(pred_labels).issubset({0, 1, 2, 3}))
        self.assertGreater(stats["count_drivable"], 0)

    def test_kalman_grid_engine(self):
        pts, _, _ = self.generator.generate_scene()
        downsampled, _, _ = self.voxelizer.voxelize(pts[:, :3])
        labels, _ = self.segmenter.predict(downsampled)

        grid_stats = self.grid.update(downsampled, labels)
        self.assertGreater(grid_stats["active_cells_r0"], 0)
        self.assertGreater(grid_stats["active_cells_r1"], 0)

        # Check Kalman ground elevation is valid where points exist
        valid_h = ~np.isnan(self.grid.ring0.ground_elevation)
        self.assertTrue(np.any(valid_h))

    def test_memory_telemetry(self):
        mem = self.grid.compute_memory_telemetry()
        self.assertGreater(mem["reduction_vs_uniform_25d_percent"], 85.0)
        self.assertGreater(mem["reduction_vs_uniform_3d_percent"], 99.0)
        self.assertLess(mem["forge_mem_mb"], 10.0)


if __name__ == "__main__":
    unittest.main()
