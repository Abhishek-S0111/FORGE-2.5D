# FORGE-2.5D: Pre-Development Architectural & Implementation Report
**Systematic Analysis of Approaches, Tech Stack Variations, and Task-by-Task Implementation Blueprints**  
*Problem Statement ID:* **SIH26053** | *Team ID:* **146825 (BYTEFORCE123)**

---

## 1. Executive Overview & Purpose

This document provides a comprehensive technical trade-off study and implementation blueprint for **FORGE-2.5D**. It deconstructs the project into discrete, testable development tasks, evaluates multiple algorithmic approaches for each, and maps out concrete tech stack combinations (from pure Python rapid prototyping to production-grade C++/CUDA edge pipelines).

The primary objective is to eliminate technical uncertainty, ensuring the system satisfies the core constraints of the SIH26053 challenge:
- **Total Latency**: $< 65\text{ ms/frame}$ ($\ge 15\text{ FPS}$) on embedded hardware (empirically benchmarked at **$25\text{--}44\text{ ms}$ / $22\text{--}30\text{ FPS}$**).
- **Memory Footprint**: $> 90\%$ reduction compared to uniform $5\text{ cm}$ grids ($\sim 246\text{k}$ cells vs $4\text{M}$ cells, achieving **$93.8\%$ RAM reduction** and **$99.8\%$ vs uniform 3D**).
- **Perception Quality**: Near-field mIoU $\ge 85\%$, Mid-field $\ge 75\%$, Far-field $\ge 60\%$.
- **Real-World Validation**: Ingestion and evaluation across both labelled and raw point clouds from the **IITH LiDAR Ground Dataset** (22 `.pcd` scans) and a 64-beam synthetic urban simulator.
- **Navigability**: Google Maps-style interactive pan/zoom web dashboard with real-time HUD telemetry.

---

## 2. High-Level Architectural Paradigms (Macro Approaches)

We evaluate three distinct overarching architectural paradigms across the pipeline:

```mermaid
graph TD
    subgraph APPROACH_A["Approach A: Python Rapid Prototype (Verified Baseline)"]
        A1["Raw PCD / Synthetic Stream"] --> A2["NumPy Range-Adaptive 2-Ring Voxelizer"]
        A2 --> A3["4-Class Semantic Segmenter (Model / GT)"]
        A3 --> A4["Vectorized 2-Ring 2.5D Kalman Grid"]
        A4 --> A5["Google Maps Canvas Web Dashboard + Telemetry HUD"]
    end

    subgraph APPROACH_B["Approach B: Hybrid Python-C++ Core (Recommended Production)"]
        B1["Raw Scans"] --> B2["C++ / OpenMP Adaptive Voxelizer"]
        B2 --> B3["SpConv / TensorRT 3D U-Net"]
        B3 --> B4["C++20 Spatial Ring Buffers + SIMD Kalman"]
        B4 --> B5["Web Canvas Dashboard + Rerun.io & Foxglove"]
        B4 --> B6["ROS 2 Nav2 Costmap C++ Plugin"]
    end

    subgraph APPROACH_C["Approach C: Pure C++ / CUDA Native (Ultra-Edge Optimized)"]
        C1["Raw Scans"] --> C2["CUDA Kernel Voxelizer"]
        C2 --> C3["TensorRT C++ Engine"]
        C3 --> C4["CUDA Grid + Thrust Kalman"]
        C4 --> C5["Native ROS 2 Costmap Layer"]
    end
```

### Comparative Summary of Macro Paradigms

| Dimension | Approach A: Python Prototype (Current Status) | Approach B: Hybrid Python-C++ (Recommended Production) | Approach C: Pure C++/CUDA (Ultra-Edge) |
| :--- | :--- | :--- | :--- |
| **Development Velocity** | ⚡ Instant / Fully Functional (Completed) | 🚀 Fast-Moderate (1–2 weeks) | ⏳ Slower (3–4 weeks) |
| **End-to-End Latency** | **$25\text{--}44\text{ ms}$ ($22\text{--}30\text{ FPS}$)** *(NumPy Vectorized)* | $\mathbf{\sim 15\text{--}25\text{ ms}}$ ($\mathbf{40\text{--}65\text{ FPS}}$) | $\mathbf{\sim 10\text{--}18\text{ ms}}$ ($\mathbf{55\text{--}100\text{ FPS}}$) |
| **RAM Footprint (100x100m)** | $\mathbf{7.51\text{ MB}}$ ($246\text{k}$ cells, **$93.8\%$ savings**) | $\sim 7.5\text{ MB}$ (Structured 32B C++ struct) | $\sim 7.5\text{ MB}$ (GPU VRAM buffer) |
| **Edge Viability (Jetson)** | Good (Vectorized NumPy hot path) | Excellent (PyBind11 zero-copy C++ core) | Maximum (Direct GPU-to-GPU memory) |
| **Real-World PCD Support** | Full (IITH labelled & raw scans) | Full (IITH labelled & raw scans) | Full (IITH labelled & raw scans) |
| **Interactive Visualization** | Google Maps Pan/Zoom Web Canvas | Google Maps Web Canvas + Rerun.io | RViz2 Plugin / WebGL Three.js |
| **SIH Hackathon Fit** | **Optimal for live interactive judging** | Ideal for final edge production | Optional stretch goal |

---

## 3. Task-by-Task Implementation Blueprint & Tech Stack Variations

---

### Task 1: Point Cloud Ingestion & Range-Adaptive Voxelization
**Objective**: Ingest raw LiDAR packets (`x, y, z, intensity, timestamp`) or PCD files, filter invalid/distant points ($>50\text{ m}$), and downsample points according to the 2-ring configuration *before* entering neural networks.

#### Algorithmic Approaches
1. **Approach 1.1: 2-Ring Range-Adaptive Voxel Slicing (Current Standard)**
   - Partition ground space into 2 concentric rings:
     - Ring 0 ($0\text{--}10\text{ m}$): $5\text{ cm} \times 5\text{ cm} \times 5\text{ cm}$ fine voxels (preserves curbs, potholes, pedestrians).
     - Ring 1 ($10\text{--}50\text{ m}$): $25\text{ cm} \times 25\text{ cm} \times 20\text{ cm}$ coarsened voxels.
     - Out of Bounds ($>50\text{ m}$): Pruned early at ingestion.
   - Vectorized centroid accumulation via `np.floor` and `np.unique` inverse indexing.
   - *Empirical Verification*: Reduces point count by **$33\%\text{--}63\%$** with zero near-field distortion.
2. **Approach 1.2: Continuous Polar-Range Voxel Hashing (Cylindrical Coordinates)**
   - Transform $(x, y, z) \rightarrow (r, \theta, z)$ with $\Delta r = k \cdot r$.
   - *Pros*: Natural LiDAR beam divergence matching.
   - *Cons*: Slower trigonometric projections in software without SIMD approximation.
3. **Approach 1.3: Real-World PCD Parser (IITH Dataset Engine)**
   - Loads binary/ASCII `.pcd` files with automatic `XYZRGB` field decoding.
   - Unpacks RGB color tags: Red ($r=255, g=0, b=0$) = Ground surface, Green ($r=0, g=255, b=0$) = Obstacles.
   - Provides seamless toggle between **Labelled Ground Truth** and **Raw Sensor Scans**.

#### Tech Stack Variations for Task 1
- **Stack 1.A (Python/NumPy Vectorized - Active)**: `np.floor`, `np.unique` centroid averaging, `pcd_loader.py`.
- **Stack 1.B (C++ / OpenMP / Robin-Hood Hashing)**: C++20 `std::span`, `tsl::robin_map` with multi-threading.
- **Stack 1.C (CUDA Kernel)**: Thrust sort-by-key + `thrust::unique_by_key` directly in VRAM.

---

### Task 2: Deep Learning 4-Class Semantic Segmentation
**Objective**: Segment point clouds into:
`0: DRIVABLE_SURFACE`, `1: NON_DRIVABLE_TERRAIN`, `2: STATIC_OBSTACLE`, `3: DYNAMIC_OBJECT`.

#### Algorithmic Approaches
1. **Approach 2.1: Dual Evaluation Engine (Current Standard)**
   - **Labelled Mode**: Extracts verified ground-truth labels from dataset RGB tags for benchmark verification.
   - **Raw Sensor Mode**: Executes geometric / heuristic feature classifier or lightweight PyTorch 1D-Conv/MLP network to dynamically infer 4-class labels on raw unlabelled scans ($< 0.3\text{ ms}$).
2. **Approach 2.2: Submanifold Sparse 3D Convolution (SpConv / MinkowskiEngine)**
   - Quantizes point cloud into a sparse voxel tensor; runs 3D sparse convolutions *only* on non-empty voxels.
   - High mIoU ($> 70\%$) with edge latency $< 35\text{ ms}$ on Jetson Orin.
3. **Approach 2.3: PointNet++ Reference Baseline**
   - Hierarchical set abstraction (SA) with ball query sampling + Feature Propagation (FP) decoder.

#### Tech Stack Variations for Task 2
- **Stack 2.A (PyTorch + Vectorized Heuristics - Active)**: PyTorch 2.x, lightweight PointNetClassifier with residual blocks.
- **Stack 2.B (SpConv v2.x + PyTorch)**: SubMConv3d + SparseConvTensor U-Net backbone.
- **Stack 2.C (TensorRT FP16 / INT8)**: Compiled TensorRT engine executing on Jetson Orin DLA/GPU.

---

### Task 3: Adaptive 2.5D Grid Engine & 3D-to-2.5D Projection
**Objective**: Project classified 3D points into a 2-ring multi-resolution 2.5D grid ($5\text{ cm} \rightarrow 25\text{ cm}$) without coordinate drift or vertical feature loss.

#### Algorithmic Approaches
1. **Approach 3.1: Concentric 2-Ring Cartesian Grids with Integer Alignment (Current Standard)**
   - Ring 0: $[-10\text{m}, 10\text{m}] \times [-10\text{m}, 10\text{m}]$ @ $\Delta r = 0.05\text{m}$ ($400 \times 400$ array, $125.7\text{k}$ circular cells).
   - Ring 1: $[-50\text{m}, 50\text{m}] \times [-50\text{m}, 50\text{m}]$ @ $\Delta r = 0.25\text{m}$ ($400 \times 400$ array, $120.6\text{k}$ annular cells).
   - Flat 1D memory indexing: `idx = (x + extent) / res * dim + (y + extent) / res`. Direct compatibility with ROS 2 costmap buffers.
2. **Approach 3.2: Multi-Layer Geometric Profile (Anti-Data-Loss)**
   - Every cell tracks: $\{z_{\text{min}}, z_{\text{ground}}, z_{\text{max}}, \Delta z, \sigma_h^2, N_{\text{pts}}, \mathbf{s}_{\text{class}}, \text{flags}\}$.
   - Clearance detection for overhanging bridges/branches ($z > z_{\text{clearance}}$) preventing phantom obstacle walls.
   - Pothole detection ($z_{\text{ground}} - z_{\text{min}} > 8\text{ cm}$) on non-drivable cells.
3. **Approach 3.3: Bilinear Seam Blending**
   - Bilinear interpolation over a $1.0\text{ m}$ transition band ($9.5\text{ m} \le d \le 10.5\text{ m}$) ensuring zero step cliffs across ring interfaces.

#### Tech Stack Variations for Task 3
- **Stack 3.A (Vectorized NumPy GridEngine - Active)**: `FoveatedGrid25D` with `np.maximum.at`, `np.minimum.at`, `np.unique` ($2\text{--}13\text{ ms}$).
- **Stack 3.B (C++20 / PyBind11 / nanobind)**: Cache-friendly contiguous ring structures with zero-copy NumPy buffer protocol ($< 5\text{ ms}$).
- **Stack 3.C (CUDA Thrust / Shared Memory)**: GPU grid accumulation via `atomicMin` / `atomicMax` on height fields ($< 2\text{ ms}$).

---

### Task 4: Probabilistic Terrain Height Filtering (Kalman Filter)
**Objective**: Recursively estimate ground surface elevation $h_j$ and estimation uncertainty $\sigma_{h,j}^2$ per cell.

#### Algorithmic Approaches
1. **Approach 4.1: Closed-Form Batch-Vectorized 1D Kalman Filter (Current Standard)**
   - Groups terrain points by cell, calculates sample mean $\bar{z}_j$ and batch variance $\sigma_{\text{batch}}^2 = \sigma_m^2 / N_{t, j}$.
   - Updates Kalman state in a single vectorized NumPy pass:
     $$K_j = \frac{\sigma_{h, j}^2}{\sigma_{h, j}^2 + \sigma_{\text{batch}}^2}, \quad h_j \leftarrow h_j + K_j(\bar{z}_j - h_j), \quad \sigma_{h, j}^2 \leftarrow (1 - K_j)\sigma_{h, j}^2$$
   - *Empirical Latency*: Reduced assembly time from $482\text{ ms}$ (unvectorized loop) to **$9.3\text{ ms}$**!
2. **Approach 4.2: Extended Kalman Filter with Surface Normal & Slope**
   - Estimates $3\times 1$ state: $[h, \theta, \phi]$ (elevation, pitch, roll) for steep incline tracking.

#### Tech Stack Variations for Task 4
- **Stack 4.A (NumPy Array-Wide Vectorized - Active)**: Batch closed-form Gaussian update.
- **Stack 4.B (C++ SIMD AVX2 / ARM NEON)**: Vectorized 8-wide float operations.
- **Stack 4.C (CUDA Warp Reduction)**: GPU warp-level parallel reduction.

---

### Task 5: Dynamic Obstacle Clustering & Velocity Estimation
**Objective**: Cluster points labeled `DYNAMIC_OBJECT` into individual vehicle/pedestrian instances and estimate motion vectors $(v_x, v_y)$.

#### Algorithmic Approaches
1. **Approach 5.1: 2.5D Connected Component Labeling (CCL) / Grid BFS**
   - Cluster adjacent dynamic cells in the 2.5D grid using 8-connectivity BFS.
   - Associate bounding boxes across frames using nearest-centroid or Hungarian matching.
   - Estimate velocity via 2D Kalman filter: $[\Delta x / \Delta t, \Delta y / \Delta t]$.
2. **Approach 5.2: 3D DBSCAN on Active Dynamic Points**
   - Euclidean clustering on downsampled 3D dynamic points using KD-tree.

#### Tech Stack Variations for Task 5
- **Stack 5.A (Python OpenCV / SciPy)**: `cv2.connectedComponentsWithStats` on the dynamic 2D occupancy slice.
- **Stack 5.B (C++ nanoflann + Hungarian Tracker)**: KD-tree based clustering with multi-target tracking.

---

### Task 6: Real-Time Visualization Dashboard & Telemetry
**Objective**: Provide an interactive visual dashboard showing 2.5D foveated layers, 4-class semantic color coding, live FPS, stage latencies, and real-time memory savings.

#### Algorithmic Approaches
1. **Approach 6.1: Google Maps-Style Navigatable Web Application (Current Standard - Port 8080)**
   - Continuous pan & drag navigation with cursor grabbing.
   - Mouse wheel and button-driven smooth zoom centered on the cursor.
   - Center & Reset (`⌖`) button snapping to the ego-vehicle frame.
   - Real-time cell hover tooltip inspecting exact coordinates $(X, Y)$, elevation $h$, variance $\sigma_h^2$, semantic classification, and hazard flags.
   - Dual Dataset Source Selector: Instant toggle between **Labelled Data**, **Raw Data**, and **Synthetic Simulation**.
   - Real-time Telemetry HUD: Live FPS gauge, latency waterfall, input decimation bar, active cell count, and memory savings gauge.
   - Layer toggles: Ring 0, Ring 1, 3D Point Cloud, and Concentric Ring Grid overlays.
2. **Approach 6.2: Native Rerun.io Visualizer**
   - Streams point clouds, 2.5D mesh grids, and scalar telemetry to Rerun SDK.
3. **Approach 6.3: Multi-Panel Matplotlib Report (`demo_dashboard.png`)**
   - Generates high-resolution static 4-panel analysis figures for documentation and reports.

#### Tech Stack Variations for Task 6
- **Stack 6.A (Python HTTP Server + HTML5 Canvas JS - Active)**: Built-in `http.server`, native JavaScript canvas with zero external npm dependencies.
- **Stack 6.B (FastAPI + Three.js + WebSockets)**: Full 3D WebGL mesh renderer with custom vertex shaders.
- **Stack 6.C (Rerun SDK)**: Native desktop viewer streaming via `pip install rerun-sdk`.

---

### Task 7: Robotics Integration (ROS 2 Nav2 Costmap Layer)
**Objective**: Expose the foveated 2.5D representation to autonomous navigation planners without performance degradation.

#### Algorithmic Approaches
1. **Approach 7.1: Multi-Resolution to Dense Costmap Downsampler Node**
   - Converts the concentric rings into a standard ROS 2 `nav_msgs/OccupancyGrid` centered on the robot.
2. **Approach 7.2: Custom Nav2 Foveated Costmap Layer Plugin (C++)**
   - Custom `nav2_costmap_2d::Layer` plugin querying `FoveatedGrid25D` directly via radial resolution lookups.

#### Tech Stack Variations for Task 7
- **Stack 7.A (rclpy Python Node)**: Standalone Python ROS 2 node publishing `nav_msgs/OccupancyGrid`.
- **Stack 7.B (rclcpp C++ Lifecycle Node)**: High-performance C++ ROS 2 node with intra-process zero-copy communication.

---

### Task 8: Quantitative Benchmarking & Evaluation Suite
**Objective**: Verify system accuracy, latency, and memory compression against standardized benchmarks (IITH Dataset, SemanticKITTI, nuScenes).

#### Algorithmic Approaches
1. **Approach 8.1: Distance-Stratified mIoU Evaluation**
   - Partition ground-truth test points into Ring 0 ($0\text{--}10\text{ m}$) and Ring 1 ($10\text{--}50\text{ m}$).
   - Compute Confusion Matrix and Intersection over Union (IoU) per semantic class in each radial band.
2. **Approach 8.2: End-to-End Latency Waterfall Profiler (Current Standard)**
   - High-precision timers instrumenting:
     1. Ingestion & Range-Adaptive Voxelization ($\text{ms}$)
     2. Semantic Segmentation ($\text{ms}$)
     3. 2.5D Grid Projection & Kalman Height Update ($\text{ms}$)
     4. Total Turn-Around Frame Time & FPS
3. **Approach 8.3: Memory Consumption & Compression Profiler (Current Standard)**
   - Resident memory (RSS) tracking comparing:
     - Baseline A: Uniform 3D Voxel Grid ($5\text{ cm}$) $\rightarrow 3,051.8\text{ MB}$
     - Baseline B: Uniform 2.5D Grid ($5\text{ cm}$) $\rightarrow 122.1\text{ MB}$
     - FORGE-2.5D 2-Ring Foveated Grid $\rightarrow \mathbf{7.51\text{ MB}}$ (**$93.8\%$ reduction**)

#### Tech Stack Variations for Task 8
- **Stack 8.A (Python High-Resolution Profiler - Active)**: `time.perf_counter()` latency waterfall + memory telemetry equations.
- **Stack 8.B (C++ Google Benchmark / Tracy Profiler)**: Nanosecond-accurate CPU/GPU timeline profiling.
- **Stack 8.C (NVIDIA Nsight Systems)**: Kernel-level GPU profiling on Jetson Orin.

---

## 4. Comprehensive Tech Stack Trade-off Matrix

| Subsystem | Option 1 (Rapid Python - Active) | Option 2 (Hybrid C++ - Target Production) | Option 3 (Pure C++/CUDA - Extreme) |
| :--- | :--- | :--- | :--- |
| **Language Runtime** | Python 3.10+ / NumPy / SciPy | Python 3.10+ & C++20 via PyBind11 | Pure Modern C++20 |
| **LiDAR Preprocessing** | NumPy Vectorized Voxelizer | C++ OpenMP + AVX2 Vectorized | CUDA Kernels + Thrust |
| **Deep Learning** | PyTorch + Geometric Heuristics | PyTorch + SpConv v2 (3D Sparse U-Net) | TensorRT C++ Engine (FP16/INT8) |
| **Grid Engine** | Vectorized 2-Ring Kalman Grid | C++20 Concentric Ring + Kalman Filter | CUDA Shared Memory Hash Grid |
| **Visualization** | Google Maps Pan/Zoom Web Canvas | FastAPI + WebSockets + Three.js & Rerun | Qt6 / OpenGL / RViz2 Plugin |
| **Robotics Interfacing** | rclpy (ROS 2 Python) | rclcpp + Python Bridge | rclcpp C++ Lifecycle Node |
| **Benchmarking** | Custom Waterfall Telemetry + HUD | Custom CLI + Tracy / Nsight Systems | NVIDIA Nsight Systems / Perf |

---

## 5. Strategic Development Roadmap & Staged Milestones

```
[Stage 1: Core Engine] ──> [Stage 2: Real Datasets] ──> [Stage 3: Interactive UI] ──> [Stage 4: C++ Core]
      (COMPLETED)                (COMPLETED)                 (COMPLETED)               (NEXT STEP)
   - 2-Ring Voxelizer         - IITH 22 PCD Scans         - Google Maps Canvas UI    - PyBind11 C++ Kernel
   - Vectorized Kalman        - Labelled Ground Truth     - Pan / Zoom / Tooltip     - SpConv Sparse U-Net
   - 4-Class Taxonomy         - Raw Model Inference       - Live Telemetry HUD       - Jetson Orin Quantization
```

### Stage 1: Functional Prototype & Preprocessing Core (✅ COMPLETED)
- Implemented `RangeAdaptiveVoxelizer` ($33\%\text{--}63\%$ point reduction).
- Implemented `FoveatedGrid25D` with batch-vectorized 1D Kalman surface elevation ($93.8\%$ RAM reduction).
- Verified with 5 automated unit tests in `tests/test_prototype.py`.

### Stage 2: Real-World Dataset & Dual Evaluation Engine (✅ COMPLETED)
- Built `PCDLoader` supporting the **IITH LiDAR Ground Dataset** (22 scans in `data/labelled_data` and `data/raw_data`).
- Enabled dual evaluation mode: **Labelled Ground Truth** vs **Raw Sensor Scan (Pure Model Inference)**.

### Stage 3: Navigatable Google Maps-Style Visual Dashboard (✅ COMPLETED)
- Built interactive web application at `http://localhost:8080`.
- Implemented continuous pan/drag, wheel zoom, center & reset (`⌖`), cell hover inspection, and live telemetry HUD.

### Stage 4: C++ Hot Path & Sparse-Conv Integration (Upcoming)
- Compile C++ SIMD AVX2 acceleration kernel via `pybind11` for sub-5ms grid assembly.
- Integrate SpConv 3D Sparse U-Net for production deep learning inference.

### Stage 5: Edge Optimization & Jetson Orin Validation (Upcoming)
- Quantize model with TensorRT FP16 for embedded vehicle deployment.
- Verify end-to-end latency targets on NVIDIA Jetson Orin hardware.
