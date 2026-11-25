"""
Standalone Interactive Web Server for FORGE-2.5D Dashboard
Runs an interactive Google Maps-style web application:
  - Pan & Zoom (Drag to pan, mousewheel / +/- to zoom)
  - Fullscreen toggle, Reset view, Layer toggles (Ring 0, Ring 1, Semantics, Elevation, Grid)
  - Tooltip inspection showing exact X, Y, Elevation, Variance, and Classification
  - Scan Selector (switching dynamically across all 22 real IITH dataset scans and synthetic scenes)
  - Real-time Performance & Memory Telemetry Gauges
"""

import os
import sys
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import numpy as np

# Ensure project root is in sys.path
# __file__ is in forge/dashboard/server.py -> dirname is forge/dashboard -> dirname is forge -> dirname is project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from forge.data.pcd_loader import PCDLoader
from forge.data.synthetic_generator import SyntheticLidarGenerator
from forge.preprocessing.voxelizer import RangeAdaptiveVoxelizer
from forge.models.segmenter import SemanticSegmenter
from forge.mapping.kalman_grid import FoveatedGrid25D

loader = PCDLoader(os.path.join(PROJECT_ROOT, "data"))
generator = SyntheticLidarGenerator()
voxelizer = RangeAdaptiveVoxelizer()
segmenter = SemanticSegmenter()
grid_engine = FoveatedGrid25D()

categorized_scans = loader.get_categorized_scans()


def process_scan(scan_id: str, category: str = "labelled"):
    """Run FORGE-2.5D pipeline on selected scan and return JSON payload."""
    t0 = time.perf_counter()
    if category == "synthetic" or scan_id == "synthetic":
        raw_pts, gt_labels, meta = generator.generate_scene()
        scan_name = "Synthetic Urban LiDAR Scene (64-beam)"
        data_mode = "SYNTHETIC SIMULATION"
        gt_for_seg = gt_labels
    elif category == "raw":
        # Search in raw scans
        matched = None
        for s in categorized_scans["raw"]:
            if s["name"] == scan_id or os.path.basename(s["path"]) == scan_id:
                matched = s["path"]
                break
        if matched is None and len(categorized_scans["raw"]) > 0:
            matched = categorized_scans["raw"][0]["path"]

        scan_name = f"IITH Raw Scan: {os.path.basename(matched)}"
        data_mode = "RAW SENSOR SCAN (PURE MODEL INFERENCE)"
        raw_pts, _, meta = loader.load_pcd(matched)
        gt_for_seg = None  # Force model inference on raw data!
    else:
        # Default to labelled scans
        matched = None
        for s in categorized_scans["labelled"]:
            if s["name"] == scan_id or os.path.basename(s["path"]) == scan_id:
                matched = s["path"]
                break
        if matched is None and len(categorized_scans["labelled"]) > 0:
            matched = categorized_scans["labelled"][0]["path"]

        scan_name = f"IITH Labelled: {os.path.basename(matched)}"
        data_mode = "LABELLED GROUND TRUTH (IITH DATASET)"
        raw_pts, gt_labels, meta = loader.load_pcd(matched)
        gt_for_seg = gt_labels

    t_load_ms = (time.perf_counter() - t0) * 1000.0

    # 1. Range-Adaptive Voxelization
    t_vox_0 = time.perf_counter()
    decimated_pts, _, vox_stats = voxelizer.voxelize(raw_pts[:, :3], raw_pts[:, 3:] if raw_pts.shape[1] > 3 else None)
    t_vox_ms = (time.perf_counter() - t_vox_0) * 1000.0

    # 2. Semantic Segmentation (Uses model inference for raw data, or GT evaluation for labelled data)
    t_seg_0 = time.perf_counter()
    pred_labels, seg_stats = segmenter.predict(
        decimated_pts, 
        gt_for_seg if (gt_for_seg is not None and len(gt_for_seg) == len(decimated_pts)) else None
    )
    t_seg_ms = (time.perf_counter() - t_seg_0) * 1000.0

    # 3. 2.5D Grid Engine & Kalman Height Filter
    t_grid_0 = time.perf_counter()
    grid_stats = grid_engine.update(decimated_pts, pred_labels)
    t_grid_ms = (time.perf_counter() - t_grid_0) * 1000.0

    total_pipeline_ms = t_vox_ms + t_seg_ms + t_grid_ms
    fps = 1000.0 / max(0.1, total_pipeline_ms)
    mem_stats = grid_engine.compute_memory_telemetry()

    # Extract Ring 0 active cells
    r0 = grid_engine.ring0
    r0_valid = r0.valid_cell_mask & (~np.isnan(r0.ground_elevation))
    ix0, iy0 = np.where(r0_valid)
    wx0, wy0 = r0.grid_to_world(ix0, iy0)
    elev0 = r0.ground_elevation[ix0, iy0]
    var0 = r0.elevation_variance[ix0, iy0]
    lbl0 = r0.semantic_label[ix0, iy0]
    pothole0 = r0.is_pothole[ix0, iy0]
    overhang0 = r0.is_overhang[ix0, iy0]

    # Extract Ring 1 active cells
    r1 = grid_engine.ring1
    r1_valid = r1.valid_cell_mask & (~np.isnan(r1.ground_elevation))
    ix1, iy1 = np.where(r1_valid)
    wx1, wy1 = r1.grid_to_world(ix1, iy1)
    elev1 = r1.ground_elevation[ix1, iy1]
    var1 = r1.elevation_variance[ix1, iy1]
    lbl1 = r1.semantic_label[ix1, iy1]
    pothole1 = r1.is_pothole[ix1, iy1]
    overhang1 = r1.is_overhang[ix1, iy1]

    # Downsample points for fast web rendering if point count > 15,000
    stride = max(1, len(decimated_pts) // 12000)
    disp_pts = decimated_pts[::stride]
    disp_lbl = pred_labels[::stride]

    payload = {
        "scan_name": scan_name,
        "data_mode": data_mode,
        "category": category,
        "available_scans": categorized_scans,
        "timings": {
            "load_ms": round(t_load_ms, 2),
            "voxelizer_ms": round(t_vox_ms, 2),
            "segmenter_ms": round(t_seg_ms, 2),
            "grid_ms": round(t_grid_ms, 2),
            "total_ms": round(total_pipeline_ms, 2),
            "fps": round(fps, 1),
        },
        "stats": {
            "raw_points": len(raw_pts),
            "decimated_points": len(decimated_pts),
            "point_reduction_pct": round(vox_stats["reduction_percent"], 1),
            "ring0_active_cells": len(ix0),
            "ring1_active_cells": len(ix1),
            "total_active_cells": len(ix0) + len(ix1),
            "detected_potholes": int(np.sum(pothole0) + np.sum(pothole1)),
            "detected_overhangs": int(np.sum(overhang0) + np.sum(overhang1)),
        },
        "memory": {
            "uniform_3d_mb": round(mem_stats["uniform_3d_mb"], 1),
            "uniform_25d_mb": round(mem_stats["uniform_25d_mb"], 1),
            "forge_mem_mb": round(mem_stats["forge_mem_mb"], 1),
            "savings_vs_25d": round(mem_stats["reduction_vs_uniform_25d_percent"], 1),
            "savings_vs_3d": round(mem_stats["reduction_vs_uniform_3d_percent"], 1),
        },
        "ring0_cells": {
            "x": np.round(wx0, 2).tolist(),
            "y": np.round(wy0, 2).tolist(),
            "elevation": np.round(elev0, 2).tolist(),
            "variance": np.round(var0, 4).tolist(),
            "label": lbl0.tolist(),
            "pothole": pothole0.tolist(),
            "overhang": overhang0.tolist(),
            "cell_size": r0.cell_size,
        },
        "ring1_cells": {
            "x": np.round(wx1, 2).tolist(),
            "y": np.round(wy1, 2).tolist(),
            "elevation": np.round(elev1, 2).tolist(),
            "variance": np.round(var1, 4).tolist(),
            "label": lbl1.tolist(),
            "pothole": pothole1.tolist(),
            "overhang": overhang1.tolist(),
            "cell_size": r1.cell_size,
        },
        "points_sample": {
            "x": np.round(disp_pts[:, 0], 2).tolist(),
            "y": np.round(disp_pts[:, 1], 2).tolist(),
            "z": np.round(disp_pts[:, 2], 2).tolist(),
            "label": disp_lbl.tolist(),
        }
    }
    return payload


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
            with open(html_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif path == "/api/scans":
            body = json.dumps(categorized_scans).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/process":
            qs = urllib.parse.parse_qs(parsed.query)
            category = qs.get("category", ["labelled"])[0]
            scan_id = qs.get("scan", ["slope_0_labelled.pcd"])[0]
            try:
                data = process_scan(scan_id, category)
                body = json.dumps(data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                err_body = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(err_body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence default access logs to keep terminal clean
        pass


def start_server(port: int = 8080):
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"\n[DASHBOARD READY] FORGE-2.5D Interactive Map Server running at: http://localhost:{port}")
    print(f"  - Pan with Click & Drag (just like Google Maps)")
    print(f"  - Zoom with Mouse Wheel or +/- controls")
    print(f"  - Hover any cell to inspect elevation, variance, and semantic class")
    print(f"  - Use top-bar dropdown to switch between real IITH LiDAR scans and synthetic scenes\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server.")
        server.server_close()


if __name__ == "__main__":
    p = 8080
    if len(sys.argv) > 1:
        p = int(sys.argv[1])
    start_server(p)
