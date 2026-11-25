"""
Visualization Engine and Dashboard for FORGE-2.5D

Produces multi-panel Bird's Eye View (BEV) plots, semantic classification maps,
2.5D elevation heatmaps, and memory/latency telemetry charts.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, Optional


class LidarVisualizer:
    def __init__(self):
        # Semantic color map
        self.color_map = {
            0: "#2ECC71",  # DRIVABLE_SURFACE (Green)
            1: "#F39C12",  # NON_DRIVABLE_TERRAIN (Amber/Brown)
            2: "#E74C3C",  # STATIC_OBSTACLE (Red)
            3: "#3498DB",  # DYNAMIC_OBJECT (Blue)
        }
        self.label_names = {
            0: "Drivable Surface",
            1: "Non-Drivable / Pothole",
            2: "Static Obstacle",
            3: "Dynamic Object",
        }

    def render_dashboard(
        self,
        raw_pts: np.ndarray,
        voxelized_pts: np.ndarray,
        labels: np.ndarray,
        grid_engine,
        pipeline_timings: Dict[str, float],
        memory_stats: Dict[str, float],
        output_filepath: str = "demo_dashboard.png",
    ):
        """Render and save a complete 4-panel dashboard figure."""
        fig = plt.figure(figsize=(18, 11), facecolor="#1E1E2F")
        plt.rcParams["text.color"] = "#E0E0E0"
        plt.rcParams["axes.labelcolor"] = "#CCCCCC"
        plt.rcParams["xtick.color"] = "#999999"
        plt.rcParams["ytick.color"] = "#999999"

        gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

        # -------------------------------------------------------------
        # Panel 1: Raw vs Range-Adaptive Voxelization (BEV)
        # -------------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor("#12121A")
        ax1.set_title(
            f"1. Range-Adaptive Voxelization (Raw: {len(raw_pts):,} -> Decimated: {len(voxelized_pts):,})",
            fontsize=12, fontweight="bold", color="#00E5FF", pad=10
        )

        # Plot decimated points
        d_xy = np.sqrt(voxelized_pts[:, 0]**2 + voxelized_pts[:, 1]**2)
        scatter1 = ax1.scatter(
            voxelized_pts[:, 0], voxelized_pts[:, 1],
            c=voxelized_pts[:, 2], cmap="viridis", s=4, alpha=0.85
        )
        cbar1 = plt.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label("Elevation z (m)", color="#CCCCCC")
        cbar1.ax.yaxis.set_tick_params(color="#CCCCCC")

        # Overlay concentric ring boundaries
        circle_r0 = plt.Circle((0, 0), 10.0, color="#00E5FF", fill=False, linestyle="--", linewidth=1.8, label="Ring 0 (10m @ 5cm)")
        circle_r1 = plt.Circle((0, 0), 50.0, color="#FF007F", fill=False, linestyle=":", linewidth=1.5, label="Ring 1 (50m @ 25cm)")
        ax1.add_patch(circle_r0)
        ax1.add_patch(circle_r1)
        ax1.plot(0, 0, marker="^", color="lime", markersize=9, label="Ego LiDAR Origin")

        ax1.set_xlim(-15, 55)
        ax1.set_ylim(-15, 15)
        ax1.set_xlabel("Longitudinal X (m)")
        ax1.set_ylabel("Lateral Y (m)")
        ax1.legend(loc="upper left", facecolor="#222233", edgecolor="none", fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.25)

        # -------------------------------------------------------------
        # Panel 2: 4-Class Semantic Segmentation Map (BEV)
        # -------------------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor("#12121A")
        ax2.set_title("2. 4-Class Deep Semantic Segmentation", fontsize=12, fontweight="bold", color="#00E5FF", pad=10)

        for class_id in range(4):
            c_mask = labels == class_id
            if np.any(c_mask):
                ax2.scatter(
                    voxelized_pts[c_mask, 0], voxelized_pts[c_mask, 1],
                    c=self.color_map[class_id], s=6, alpha=0.9,
                    label=f"{self.label_names[class_id]} ({np.sum(c_mask):,})"
                )

        ax2.add_patch(plt.Circle((0, 0), 10.0, color="#00E5FF", fill=False, linestyle="--", linewidth=1.5))
        ax2.add_patch(plt.Circle((0, 0), 50.0, color="#FF007F", fill=False, linestyle=":", linewidth=1.2))
        ax2.plot(0, 0, marker="^", color="lime", markersize=9)

        ax2.set_xlim(-15, 55)
        ax2.set_ylim(-15, 15)
        ax2.set_xlabel("Longitudinal X (m)")
        ax2.set_ylabel("Lateral Y (m)")
        ax2.legend(loc="upper left", facecolor="#222233", edgecolor="none", fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.25)

        # -------------------------------------------------------------
        # Panel 3: Foveated 2.5D Elevation Grid with Kalman Ground
        # -------------------------------------------------------------
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor("#12121A")
        ax3.set_title("3. Foveated 2.5D Elevation Map (Kalman Filtered Ground)", fontsize=12, fontweight="bold", color="#00E5FF", pad=10)

        # Visualizing Ring 0 elevation patch
        r0 = grid_engine.ring0
        extent_r0 = [-r0.extent, r0.extent, -r0.extent, r0.extent]
        # Mask out empty cells for clean display
        r0_elev = np.ma.masked_invalid(r0.ground_elevation)
        im3 = ax3.imshow(
            r0_elev.T, origin="lower", extent=extent_r0, cmap="terrain",
            vmin=-2.0, vmax=0.5, interpolation="nearest"
        )
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label("Kalman Ground Elevation (m)", color="#CCCCCC")
        cbar3.ax.yaxis.set_tick_params(color="#CCCCCC")

        # Mark detected potholes and dynamic objects in Ring 0
        pothole_y, pothole_x = np.where(r0.is_pothole)
        if len(pothole_x) > 0:
            px, py = r0.grid_to_world(pothole_y, pothole_x)
            ax3.scatter(px, py, color="#FF0055", marker="x", s=35, linewidth=2, label=f"Pothole ({len(pothole_x)})")

        ax3.set_xlim(-12, 12)
        ax3.set_ylim(-12, 12)
        ax3.set_xlabel("X (m) [Ring 0 Fovea: 5cm cells]")
        ax3.set_ylabel("Y (m)")
        handles, labels_list = ax3.get_legend_handles_labels()
        if handles:
            ax3.legend(loc="upper left", facecolor="#222233", edgecolor="none", fontsize=8)
        ax3.grid(True, linestyle=":", alpha=0.25)

        # -------------------------------------------------------------
        # Panel 4: Performance Telemetry (Memory Reduction & Latency Waterfall)
        # -------------------------------------------------------------
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor("#12121A")
        ax4.set_title("4. Real-Time Telemetry: Memory Savings & Latency", fontsize=12, fontweight="bold", color="#00E5FF", pad=10)

        # Subplot splits: Left side Memory comparison, Right side Latency waterfall
        categories = ["Uniform 3D\n(5cm)", "Uniform 2.5D\n(5cm)", "FORGE-2.5D\n(2-Ring)"]
        mem_values = [
            memory_stats["uniform_3d_mb"],
            memory_stats["uniform_25d_mb"],
            memory_stats["forge_mem_mb"],
        ]
        bar_colors = ["#E74C3C", "#F39C12", "#2ECC71"]

        bars = ax4.bar(categories, mem_values, color=bar_colors, width=0.55, edgecolor="none")
        ax4.set_ylabel("Memory Footprint (MB)")
        ax4.set_yscale("log")
        ax4.set_ylim(1, 10000)

        # Annotate memory numbers on bars
        for bar, val in zip(bars, mem_values):
            y_pos = bar.get_height() * 1.25
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0, y_pos,
                f"{val:.1f} MB", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#FFFFFF"
            )

        # Add Latency and Savings Callout Box
        fps = 1000.0 / max(0.1, pipeline_timings["total_ms"])
        callout_text = (
            f"--- TIMING PROFILE ---\n"
            f"Voxelization: {pipeline_timings.get('voxelizer_ms', 0):.2f} ms\n"
            f"Segmentation: {pipeline_timings.get('segmenter_ms', 0):.2f} ms\n"
            f"2.5D Assembly: {pipeline_timings.get('grid_ms', 0):.2f} ms\n"
            f"Total Turn-Around: {pipeline_timings.get('total_ms', 0):.2f} ms\n"
            f"Throughput: {fps:.1f} FPS (Target >= 15)\n\n"
            f"--- MEMORY REDUCTION ---\n"
            f"vs Uniform 2.5D: {memory_stats['reduction_vs_uniform_25d_percent']:.1f}%\n"
            f"vs Uniform 3D:   {memory_stats['reduction_vs_uniform_3d_percent']:.1f}%"
        )
        ax4.text(
            0.62, 0.48, callout_text,
            transform=ax4.transAxes,
            fontsize=8.5, fontfamily="monospace",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#1F1F35", edgecolor="#00E5FF", alpha=0.9)
        )

        plt.savefig(output_filepath, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return output_filepath
