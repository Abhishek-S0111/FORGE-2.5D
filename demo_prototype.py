"""
FORGE-2.5D: Rapid Prototype End-to-End Demonstration
Executes complete pipeline:
  Synthetic LiDAR Scan -> Range-Adaptive Voxelizer -> 4-Class Segmenter 
  -> 2-Ring Foveated 2.5D Kalman Grid -> Performance Telemetry & Visualization
"""

import sys
import os
import time
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forge.data.synthetic_generator import SyntheticLidarGenerator
from forge.preprocessing.voxelizer import RangeAdaptiveVoxelizer
from forge.models.segmenter import SemanticSegmenter
from forge.mapping.kalman_grid import FoveatedGrid25D
from forge.viz.visualizer import LidarVisualizer


def run_pipeline():
    print("=" * 75)
    print("      FORGE-2.5D: ADAPTIVE VARIABLE-RESOLUTION 2.5D LIDAR MAPPING      ")
    print("          SIH 2026 (SIH26053) - RAPID PROTOTYPE DEMONSTRATION           ")
    print("=" * 75)

    # 1. Initialize Pipeline Modules
    print("\n[Stage 0] Initializing FORGE-2.5D 2-Ring Perception Engine...")
    generator = SyntheticLidarGenerator()
    voxelizer = RangeAdaptiveVoxelizer(
        r0_max_dist=10.0,
        r1_max_dist=50.0,
        r0_voxel_size=(0.05, 0.05, 0.05),
        r1_voxel_size=(0.25, 0.25, 0.20),
    )
    segmenter = SemanticSegmenter(device="cpu")
    grid_engine = FoveatedGrid25D(
        r0_radius=10.0,
        r1_radius=50.0,
        r0_cell_size=0.05,
        r1_cell_size=0.25,
    )
    visualizer = LidarVisualizer()
    print("  -> Configuration:")
    print("     - Ring 0 (Near-Field Fovea): 0.0m - 10.0m @ 5cm resolution")
    print("     - Ring 1 (Far-Field Merged): 10.0m - 50.0m @ 25cm resolution")
    print("     - Ring Boundary Seam: 10.0m (+/- 0.5m transition blending)")
    print("     - Out of Bounds (>50m): Excluded")

    # 2. Ingest / Synthesize LiDAR Scan
    print("\n[Stage 1] Ingesting Synthetic 64-Beam Urban LiDAR Scan...")
    t_gen_start = time.perf_counter()
    raw_points, gt_labels, scene_meta = generator.generate_scene()
    t_gen_ms = (time.perf_counter() - t_gen_start) * 1000.0
    print(f"  -> Generated {len(raw_points):,} raw points in {t_gen_ms:.2f} ms")
    print(f"     Ground truth: {scene_meta['num_drivable']:,} Drivable, "
          f"{scene_meta['num_nondrivable']:,} Rough/Pothole, "
          f"{scene_meta['num_static']:,} Static, "
          f"{scene_meta['num_dynamic']:,} Dynamic")

    # 3. Range-Adaptive Voxelization
    print("\n[Stage 2] Executing Range-Adaptive Voxelization (< 15 ms target)...")
    decimated_pts, decimated_feat, vox_stats = voxelizer.voxelize(raw_points[:, :3], raw_points[:, 3:])
    print(f"  -> Raw points:            {vox_stats['n_raw']:,}")
    print(f"  -> Decimated points:       {vox_stats['n_out']:,}")
    print(f"  -> Input compute reduction: {vox_stats['reduction_percent']:.1f}%")
    print(f"  -> Voxelizer Latency:      {vox_stats['latency_ms']:.2f} ms")

    # 4. 4-Class Deep Semantic Segmentation
    print("\n[Stage 3] Executing 4-Class Semantic Segmentation (< 40 ms target)...")
    pred_labels, seg_stats = segmenter.predict(decimated_pts)
    print(f"  -> Semantic Distribution:")
    print(f"     - Class 0 (Drivable Road):      {seg_stats['count_drivable']:,} pts")
    print(f"     - Class 1 (Non-Drivable/Pothole):{seg_stats['count_nondrivable']:,} pts")
    print(f"     - Class 2 (Static Obstacles):   {seg_stats['count_static']:,} pts")
    print(f"     - Class 3 (Dynamic Objects):    {seg_stats['count_dynamic']:,} pts")
    print(f"  -> Segmentation Latency:           {seg_stats['latency_ms']:.2f} ms")

    # 5. 2-Ring Foveated 2.5D Grid Engine & Kalman Height Update
    print("\n[Stage 4] Assembling 2-Ring 2.5D Grid & Running Kalman Elevation (< 10 ms target)...")
    grid_stats = grid_engine.update(decimated_pts, pred_labels)
    print(f"  -> Active cells Ring 0 (5cm):  {grid_stats['active_cells_r0']:,}")
    print(f"  -> Active cells Ring 1 (25cm): {grid_stats['active_cells_r1']:,}")
    print(f"  -> Total active 2.5D cells:    {grid_stats['total_active_cells']:,}")
    print(f"  -> Grid Assembly Latency:      {grid_stats['latency_ms']:.2f} ms")

    # Check pothole detection
    num_potholes = np.sum(grid_engine.ring0.is_pothole)
    num_overhangs = np.sum(grid_engine.ring0.is_overhang) + np.sum(grid_engine.ring1.is_overhang)
    print(f"  -> Detected Hazards:")
    print(f"     - Negative Obstacles (Potholes): {num_potholes} cells")
    print(f"     - Overhanging Clearances:        {num_overhangs} cells")

    # 6. Memory & Latency Telemetry
    total_pipeline_ms = vox_stats['latency_ms'] + seg_stats['latency_ms'] + grid_stats['latency_ms']
    fps = 1000.0 / max(0.1, total_pipeline_ms)
    mem_stats = grid_engine.compute_memory_telemetry()

    print("\n" + "=" * 75)
    print("                    PERFORMANCE & MEMORY TELEMETRY                     ")
    print("=" * 75)
    print(f"  [End-to-End Latency Waterfall]")
    print(f"    1. Range-Adaptive Voxelizer: {vox_stats['latency_ms']:6.2f} ms")
    print(f"    2. Semantic Segmentation:    {seg_stats['latency_ms']:6.2f} ms")
    print(f"    3. 2.5D Grid & Kalman Filter:{grid_stats['latency_ms']:6.2f} ms")
    print(f"    ----------------------------------------")
    print(f"    TOTAL TURN-AROUND:           {total_pipeline_ms:6.2f} ms  (Target < 65.0 ms)")
    print(f"    THROUGHPUT:                  {fps:6.1f} FPS (Target >= 15.0 FPS)")

    print(f"\n  [Memory Footprint Analysis (100m x 100m Area)]")
    print(f"    - Baseline Uniform 3D Grid (5cm):  {mem_stats['uniform_3d_mb']:>9.1f} MB  ({int(mem_stats['uniform_3d_voxels']):,} voxels)")
    print(f"    - Baseline Uniform 2.5D Grid (5cm):{mem_stats['uniform_25d_mb']:>9.1f} MB  ({int(mem_stats['uniform_25d_cells']):,} cells)")
    print(f"    - FORGE-2.5D (2-Ring Adaptive):    {mem_stats['forge_mem_mb']:>9.1f} MB  ({int(mem_stats['forge_total_cells']):,} cells)")
    print(f"    ----------------------------------------")
    print(f"    MEMORY REDUCTION vs UNIFORM 2.5D:  {mem_stats['reduction_vs_uniform_25d_percent']:>8.1f}%")
    print(f"    MEMORY REDUCTION vs UNIFORM 3D:    {mem_stats['reduction_vs_uniform_3d_percent']:>8.1f}%")

    # 7. Render Visualization Dashboard
    print("\n[Stage 5] Generating Multi-Panel Visual Dashboard...")
    output_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_dashboard.png")
    timings = {
        "voxelizer_ms": vox_stats['latency_ms'],
        "segmenter_ms": seg_stats['latency_ms'],
        "grid_ms": grid_stats['latency_ms'],
        "total_ms": total_pipeline_ms,
    }
    visualizer.render_dashboard(
        raw_pts=raw_points,
        voxelized_pts=decimated_pts,
        labels=pred_labels,
        grid_engine=grid_engine,
        pipeline_timings=timings,
        memory_stats=mem_stats,
        output_filepath=output_png,
    )
    print(f"  -> Dashboard figure saved to: {output_png}")
    print("\n[SUCCESS] FORGE-2.5D Rapid Prototype Pipeline Execution Complete!")


if __name__ == "__main__":
    run_pipeline()
