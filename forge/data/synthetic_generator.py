"""
Realistic Synthetic Urban LiDAR Scene Generator for FORGE-2.5D

Simulates a 64-beam automotive LiDAR scanning a complex urban road environment:
  - Asphalt Roadway (Drivable Surface)
  - Curbs, Potholes, and Berms (Non-Drivable Terrain)
  - Utility Poles, Guardrails, and Overpass (Static Obstacles)
  - Moving Cars and Crossing Pedestrians (Dynamic Objects)
"""

import numpy as np
from typing import Tuple, Dict


class SyntheticLidarGenerator:
    def __init__(
        self,
        num_rings: int = 64,
        azimuth_resolution_deg: float = 0.25,
        max_range: float = 60.0,
        sensor_height: float = 1.7,
        measurement_noise: float = 0.02,
    ):
        self.num_rings = num_rings
        self.azimuth_res = np.radians(azimuth_resolution_deg)
        self.max_range = max_range
        self.sensor_height = sensor_height
        self.measurement_noise = measurement_noise

    def generate_scene(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Generate a complete point cloud frame with 4 ground-truth semantic classes.

        Returns:
            points: (N, 4) array [x, y, z, intensity].
            labels: (N,) semantic labels in {0, 1, 2, 3}.
            metadata: Counts of objects and points.
        """
        pts_list = []
        lbl_list = []

        # 1. Roadway Ground Plane (z = -sensor_height = -1.7m)
        xs = np.linspace(-15.0, 50.0, 150)
        ys = np.linspace(-3.5, 3.5, 50)
        grid_x, grid_y = np.meshgrid(xs, ys)
        gx = grid_x.ravel()
        gy = grid_y.ravel()
        gz = np.full_like(gx, -self.sensor_height)

        # Inject pothole at x = 7.0m, y = 0.5m (radius 0.6m, depth 12cm)
        pothole_dist = np.sqrt((gx - 7.0) ** 2 + (gy - 0.5) ** 2)
        pothole_mask = pothole_dist < 0.6
        gz[pothole_mask] -= 0.12

        glabels = np.zeros(len(gx), dtype=np.uint8)  # Class 0: DRIVABLE
        glabels[pothole_mask] = 1                     # Class 1: NON_DRIVABLE (pothole)

        pts_list.append(np.column_stack([gx, gy, gz]))
        lbl_list.append(glabels)

        # 2. Road Shoulders / Rough Terrain & Curbs (y in [-6, -3.5] and [3.5, 6])
        for y_range in [(-6.5, -3.5), (3.5, 6.5)]:
            ys_rough = np.linspace(y_range[0], y_range[1], 25)
            rx, ry = np.meshgrid(xs, ys_rough)
            rx = rx.ravel()
            ry = ry.ravel()
            rz = np.full_like(rx, -self.sensor_height + 0.15)  # Curb elevation
            rz += np.random.uniform(-0.04, 0.04, size=len(rx)) # Roughness
            
            pts_list.append(np.column_stack([rx, ry, rz]))
            lbl_list.append(np.full(len(rx), 1, dtype=np.uint8))  # Class 1: NON_DRIVABLE

        # 3. Static Obstacles: Utility Poles and Wall
        # Utility poles along y = -4.5m at x = 4m, 14m, 28m, 42m
        pole_xs = [4.0, 14.0, 28.0, 42.0]
        for px in pole_xs:
            pz = np.linspace(-self.sensor_height, 2.5, 40)
            theta = np.linspace(0, 2 * np.pi, 12)
            for z_val in pz:
                pole_pts = np.column_stack([
                    px + 0.15 * np.cos(theta),
                    -4.5 + 0.15 * np.sin(theta),
                    np.full_like(theta, z_val),
                ])
                pts_list.append(pole_pts)
                lbl_list.append(np.full(len(theta), 2, dtype=np.uint8))  # Class 2: STATIC

        # Roadside Building Wall along y = 5.5m, x in [10m, 35m]
        wall_x, wall_z = np.meshgrid(np.linspace(10, 35, 60), np.linspace(-self.sensor_height, 3.0, 30))
        wall_pts = np.column_stack([
            wall_x.ravel(),
            np.full(wall_x.size, 5.5),
            wall_z.ravel(),
        ])
        pts_list.append(wall_pts)
        lbl_list.append(np.full(len(wall_pts), 2, dtype=np.uint8))

        # Overhang: Overhead Highway Signboard / Bridge at x = 20m, z = +1.5m, y in [-4, 4]
        sign_y, sign_z = np.meshgrid(np.linspace(-4, 4, 30), np.linspace(1.2, 2.2, 15))
        sign_pts = np.column_stack([
            np.full(sign_y.size, 20.0),
            sign_y.ravel(),
            sign_z.ravel(),
        ])
        pts_list.append(sign_pts)
        lbl_list.append(np.full(len(sign_pts), 2, dtype=np.uint8))

        # 4. Dynamic Objects
        # Dynamic 1: Crossing Pedestrian at x = 6.0m, y = -0.5m (Near field!)
        ped_z = np.linspace(-self.sensor_height, -0.1, 30)
        ped_pts = []
        for pzv in ped_z:
            th = np.linspace(0, 2 * np.pi, 8)
            ped_pts.append(np.column_stack([
                6.0 + 0.25 * np.cos(th),
                -0.5 + 0.25 * np.sin(th),
                np.full_like(th, pzv)
            ]))
        ped_pts = np.vstack(ped_pts)
        pts_list.append(ped_pts)
        lbl_list.append(np.full(len(ped_pts), 3, dtype=np.uint8))  # Class 3: DYNAMIC

        # Dynamic 2: Vehicle ahead at x = 16.0m, y = 1.0m (Mid field)
        # Box: 4.2m long, 1.8m wide, 1.4m high
        car_xs = np.linspace(14.0, 18.2, 25)
        car_ys = np.linspace(0.1, 1.9, 15)
        car_zs = np.linspace(-self.sensor_height + 0.25, -self.sensor_height + 1.65, 15)
        
        # Shell of the car box
        car_pts = []
        # Roof
        cx, cy = np.meshgrid(car_xs, car_ys)
        car_pts.append(np.column_stack([cx.ravel(), cy.ravel(), np.full(cx.size, -self.sensor_height + 1.65)]))
        # Sides
        for side_y in [0.1, 1.9]:
            cx, cz = np.meshgrid(car_xs, car_zs)
            car_pts.append(np.column_stack([cx.ravel(), np.full(cx.size, side_y), cz.ravel()]))
        # Front / back
        for end_x in [14.0, 18.2]:
            cy, cz = np.meshgrid(car_ys, car_zs)
            car_pts.append(np.column_stack([np.full(cy.size, end_x), cy.ravel(), cz.ravel()]))
            
        car_pts = np.vstack(car_pts)
        pts_list.append(car_pts)
        lbl_list.append(np.full(len(car_pts), 3, dtype=np.uint8))  # Class 3: DYNAMIC

        # Combine all components
        all_pts = np.vstack(pts_list)
        all_lbls = np.concatenate(lbl_list)

        # Add Gaussian sensor measurement noise
        noise = np.random.normal(0, self.measurement_noise, size=all_pts.shape)
        all_pts += noise

        # Simulate LiDAR intensity (reflectivity): high on road/obstacles, lower on terrain
        intensity = np.random.uniform(0.3, 0.8, size=(len(all_pts), 1))
        intensity[all_lbls == 0] = 0.7  # Paved road
        intensity[all_lbls == 1] = 0.2  # Rough grass / dirt
        intensity[all_lbls == 2] = 0.9  # Retroreflective signs / metal poles

        points_with_feat = np.hstack([all_pts, intensity]).astype(np.float32)

        metadata = {
            "total_points": len(all_pts),
            "num_drivable": int(np.sum(all_lbls == 0)),
            "num_nondrivable": int(np.sum(all_lbls == 1)),
            "num_static": int(np.sum(all_lbls == 2)),
            "num_dynamic": int(np.sum(all_lbls == 3)),
        }
        return points_with_feat, all_lbls, metadata
