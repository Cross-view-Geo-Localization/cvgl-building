"""
preprocess_uav1.py
==================
Preprocessing pipeline for UAV-VisLoc-style datasets.

Usage (CLI):
    python preprocess_uav1.py \
        --root /path/to/dataset \
        --save-root /path/to/output \
        --split cross-area \
        --satellite-key HoaLac \
        --satellite-lt-lat 21.0152911 --satellite-lt-lon 105.5161285 \
        --satellite-rb-lat 20.9909321 --satellite-rb-lon 105.5518341 \
        --satellite-h 13312 --satellite-w 9728 \
        --train-ids 1 3 --test-ids 4 \
        --steps tile csv label

All parameters can also be set programmatically via DatasetConfig.
"""

import os
import re
import argparse
import math
import csv
import json
import pickle
import random
import shutil
import concurrent.futures
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from PIL import Image
import rasterio

# Disable Pillow's size bomb protection for massive map conversions
Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """
    Central configuration object.  Pass one of these to every pipeline
    function so no values are ever hardcoded inside a function body.
    """
    # --- Paths ---
    root: str = ""                   # Dataset root directory
    save_root: str = ""              # Output directory for processed data

    # --- Split ---
    split_type: str = "cross-area"   # "cross-area" or "same-area"
    train_ids: List[int] = field(default_factory=lambda: [1, 3])
    test_ids: List[int] = field(default_factory=lambda: [4])

    # --- Satellite tile parameters ---
    tile_size: int = 384
    # Number of zoom levels to keep (counted from the *end* of the sorted list).
    # E.g. zoom_keep_slice = (-3, -1) keeps zoom levels [-3:-1] (2 levels).
    zoom_keep_slice: Tuple[int, int] = (-3, -1)

    # --- Matching thresholds ---
    threshold: float = 0.39       # IoU threshold for positive pairs
    semi_threshold: float = 0.14  # IoU threshold for semi-positive pairs

    # --- Camera / sensor defaults (DJI Mavic 3) ---
    fov_h: float = 84.0   # Horizontal FOV in degrees
    fov_v: float = 56.0   # Vertical FOV in degrees
    sensor_width_mm: float = 36.0
    sensor_height_mm: float = 27.0

    # --- Satellite georeferencing ---
    # Key used to look up the single "overview" satellite image that covers
    # ALL trajectories (e.g. "HoaLac").
    satellite_key: str = "HoaLac"
    # Bounding box of that overview satellite: [lt_lat, lt_lon, rb_lat, rb_lon]
    satellite_latlon: List[float] = field(
        default_factory=lambda: [21.0152911, 105.5161285, 20.9909321, 105.5518341]
    )
    # Pixel size of that overview satellite: [height, width]
    satellite_size: List[int] = field(default_factory=lambda: [13312, 9728])

    # --- Per-trajectory satellite metadata (used by tile_satellite) ---
    # Maps str(trajectory_id) -> [height, width]
    traj_sate_size: Dict[str, List[int]] = field(default_factory=dict)
    # Maps str(trajectory_id) -> [lt_lat, lt_lon, rb_lat, rb_lon]
    traj_sate_latlon: Dict[str, List[float]] = field(default_factory=dict)

    # --- Derived helpers ---
    @property
    def all_ids(self) -> List[int]:
        return sorted(set(self.train_ids) | set(self.test_ids))

    @property
    def satellite_tile_dir(self) -> str:
        """Directory where tiles for the overview satellite are stored."""
        return os.path.join(self.root, "tile", self.satellite_key)


# ---------------------------------------------------------------------------
# Utility / Math helpers
# ---------------------------------------------------------------------------

def calculate_phi(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the compass bearing between two GPS points.
    Returns an angle in degrees from 0 to 360.
    """
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    delta_lon = lon2_rad - lon1_rad

    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = (math.cos(lat1_rad) * math.sin(lat2_rad)
         - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))

    bearing = math.degrees(math.atan2(x, y))
    return round((bearing + 360) % 360, 2)


def calculate_fov(focal_len: Optional[float],
                  sensor_width_mm: float,
                  sensor_height_mm: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate horizontal and vertical FOV in degrees.

    Args:
        focal_len: focal length in mm
        sensor_width_mm: sensor width in mm
        sensor_height_mm: sensor height in mm

    Returns:
        (fov_h, fov_v) in degrees, or (None, None) if focal_len is invalid.
    """
    if focal_len is None or focal_len <= 0:
        return None, None
    fov_h = 2 * math.degrees(math.atan(sensor_width_mm / (2 * focal_len)))
    fov_v = 2 * math.degrees(math.atan(sensor_height_mm / (2 * focal_len)))
    return round(fov_h, 2), round(fov_v, 2)


def offset_to_latlon(latitude: float, longitude: float, dx: float, dy: float):
    """Convert a metric offset (dx, dy) from a given GPS point to a new lat/lon."""
    R = 6_378_137  # Earth radius in metres
    dlat = dy / R
    dlon = dx / (R * math.cos(math.pi * latitude / 180))
    return latitude + math.degrees(dlat), longitude + math.degrees(dlon)


def geo_to_image_coords(lat: float, lon: float,
                        lat1: float, lon1: float,
                        lat2: float, lon2: float,
                        H: int, W: int) -> Tuple[int, int]:
    """Map a GPS coordinate to pixel (x, y) within a georeferenced image."""
    R = 6_378_137
    center_lat = (lat1 + lat2) / 2
    x_range = R * (lon2 - lon1) * math.cos(math.radians(center_lat))
    y_range = R * (lat2 - lat1)
    x_offset = R * (lon - lon1) * math.cos(math.radians((lat1 + lat) / 2))
    y_offset = R * (lat - lat1)
    return int((x_offset / x_range) * W), int((y_offset / y_range) * H)


def calculate_coverage_endpoints(heading_angle: float, height: float,
                                 cur_lat: float, cur_lon: float,
                                 fov_horizontal: float, fov_vertical: float,
                                 debug: bool = False) -> dict:
    """
    Compute the four ground-footprint corners (lat/lon) of a nadir camera
    given the drone's heading, altitude, location, and FOV.
    """
    heading_rad = math.radians(heading_angle)
    fov_h_rad = math.radians(fov_horizontal)
    fov_v_rad = math.radians(fov_vertical)

    half_h = height * math.tan(fov_h_rad / 2)
    half_v = height * math.tan(fov_v_rad / 2)

    adj_rad = math.radians((90 - heading_angle) % 360)
    cos_a, sin_a = math.cos(adj_rad), math.sin(adj_rad)

    tl_x = -half_h * cos_a - half_v * sin_a
    tl_y = -half_h * sin_a + half_v * cos_a
    tr_x =  half_h * cos_a - half_v * sin_a
    tr_y =  half_h * sin_a + half_v * cos_a
    bl_x = -half_h * cos_a + half_v * sin_a
    bl_y = -half_h * sin_a - half_v * cos_a
    br_x =  half_h * cos_a + half_v * sin_a
    br_y =  half_h * sin_a - half_v * cos_a

    if debug:
        print("offsets:", tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y)

    return {
        "top_left":     offset_to_latlon(cur_lat, cur_lon, tl_x, tl_y),
        "top_right":    offset_to_latlon(cur_lat, cur_lon, tr_x, tr_y),
        "bottom_left":  offset_to_latlon(cur_lat, cur_lon, bl_x, bl_y),
        "bottom_right": offset_to_latlon(cur_lat, cur_lon, br_x, br_y),
    }


def order_points(points):
    hull = ConvexHull(points)
    return [points[i] for i in hull.vertices]


def calc_intersect_area(poly1: Polygon, poly2: Polygon) -> float:
    return poly1.intersection(poly2).area


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_tif(tif_path: str) -> Optional[Image.Image]:
    """
    Read a GeoTIFF / BigTIFF with Rasterio and return an RGB PIL Image.
    Returns None on failure.
    """
    print(f"Reading with Rasterio: {tif_path}")
    try:
        with rasterio.open(tif_path) as dataset:
            img_array = np.transpose(dataset.read(), (1, 2, 0))
            if img_array.dtype != np.uint8:
                img_array = (img_array / np.max(img_array) * 255).astype(np.uint8)
            bands = img_array.shape[2] if len(img_array.shape) == 3 else 1
            if bands >= 4:
                return Image.fromarray(img_array[:, :, :3], "RGB")
            elif bands == 3:
                return Image.fromarray(img_array, "RGB")
            else:
                return Image.fromarray(np.squeeze(img_array)).convert("RGB")
    except Exception as e:
        print(f"ERROR: Rasterio failed to read {tif_path}\nDetails: {e}")
        return None


def parse_dji_srt(srt_file: str) -> dict:
    """
    Read a DJI SRT subtitle file and return {frame_id: telemetry_dict}.
    """
    with open(srt_file, "r", encoding="utf-8") as f:
        content = f.read()

    registry = {}
    for block in content.strip().split("\n\n"):
        frame_m  = re.search(r"FrameCnt:\s*(\d+)", block)
        date_m   = re.search(r"(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d+)", block)
        lat_m    = re.search(r"\[latitude:\s*([\d.-]+)\]", block)
        lon_m    = re.search(r"\[longitude:\s*([\d.-]+)\]", block)
        alt_m    = re.search(r"\[rel_alt:\s*([\d.-]+)\s*abs_alt:\s*([\d.-]+)\]", block)
        focal_m  = re.search(r"\[focal_len:\s*([\d.]+)\]", block)

        if all([frame_m, date_m, lat_m, lon_m, alt_m]):
            fid = int(frame_m.group(1))
            registry[fid] = {
                "date":     date_m.group(1),
                "lat":      float(lat_m.group(1)),
                "lon":      float(lon_m.group(1)),
                "rel_alt":  float(alt_m.group(1)),
                "abs_alt":  float(alt_m.group(2)),
                "focal_len": float(focal_m.group(1)) if focal_m else None,
            }
    return registry


# ---------------------------------------------------------------------------
# Step 1: Build per-trajectory CSVs from DJI SRT files
# ---------------------------------------------------------------------------

def create_csv_from_srt(srt_file: str, csv_file: str, drone_folder: str,
                        cfg: DatasetConfig) -> None:
    """
    Parse a DJI SRT file and write a telemetry CSV for all JPEG frames in
    *drone_folder* that have matching SRT entries.

    Args:
        srt_file:     Path to the .SRT telemetry file.
        csv_file:     Destination CSV path.
        drone_folder: Folder containing the drone JPEG images.
        cfg:          DatasetConfig supplying sensor dimensions and FOV defaults.
    """
    srt_data = parse_dji_srt(srt_file)

    valid_frames: List[int] = []
    file_mapping: Dict[int, str] = {}

    for fname in os.listdir(drone_folder):
        if fname.lower().endswith(".jpg"):
            try:
                frame_id = int(fname.split("_")[-1].split(".")[0])
                if frame_id in srt_data:
                    valid_frames.append(frame_id)
                    file_mapping[frame_id] = fname
            except ValueError:
                continue

    valid_frames.sort()

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["num", "filename", "date", "lat", "lon",
                         "rel_alt", "abs_alt", "phi", "FOV_H", "FOV_V"])

        last_phi = 0.0
        for idx, cur_frame in enumerate(valid_frames):
            d_cur = srt_data[cur_frame]
            if idx + 1 < len(valid_frames):
                d_next = srt_data[valid_frames[idx + 1]]
                last_phi = calculate_phi(d_cur["lat"], d_cur["lon"],
                                         d_next["lat"], d_next["lon"])
            phi = last_phi

            fov_h, fov_v = calculate_fov(d_cur["focal_len"],
                                          cfg.sensor_width_mm,
                                          cfg.sensor_height_mm)

            writer.writerow([
                cur_frame, file_mapping[cur_frame], d_cur["date"],
                d_cur["lat"], d_cur["lon"],
                d_cur["rel_alt"], d_cur["abs_alt"],
                phi, fov_h, fov_v,
            ])


def generate_all_csvs(cfg: DatasetConfig) -> None:
    """
    Iterate over every subdirectory in <root>/video_map_DSMAC that contains
    an SRT file, generate the matching CSV in <root>/drone_og/<name>/.
    """
    src_root = os.path.join(cfg.root, "video_map_DSMAC")
    for dir_name in sorted(os.listdir(src_root)):
        dir_path = os.path.join(src_root, dir_name)
        if not os.path.isdir(dir_path):
            continue

        srt_files = [f for f in os.listdir(dir_path) if f.upper().endswith(".SRT")]
        if not srt_files:
            print(f"Warning: No SRT file found in {dir_name}, skipping...")
            continue

        srt_path = os.path.join(dir_path, srt_files[0])
        csv_name = srt_files[0].replace(".SRT", ".csv").replace(".srt", ".csv")
        base = csv_name.replace(".csv", "")
        drone_path = os.path.join(cfg.root, "drone_og", base)
        csv_path   = os.path.join(drone_path, csv_name)

        os.makedirs(drone_path, exist_ok=True)
        create_csv_from_srt(srt_path, csv_path, drone_path, cfg)
        print(f"  CSV written: {csv_path}")


# ---------------------------------------------------------------------------
# Step 2: Tile satellite images
# ---------------------------------------------------------------------------

def _process_tile(args) -> None:
    """Worker function for multiprocessing tile generation."""
    scaled_image, identifier, zoom_dir, zoom, x, y, tile_size = args
    box = (x, y,
           min(x + tile_size, scaled_image.width),
           min(y + tile_size, scaled_image.height))
    tile = scaled_image.crop(box)
    canvas = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    canvas.paste(tile, (0, 0))
    fname = f"{identifier}_{zoom}_{x // tile_size:03}_{y // tile_size:03}.png"
    canvas.save(os.path.join(zoom_dir, fname))


def _tile_image(image: Image.Image, identifier: str, output_dir: str,
                tile_size: int) -> None:
    """
    Tile a single PIL image at all zoom levels and save into *output_dir*.

    Args:
        image:      PIL Image to tile.
        identifier: Prefix string for output filenames (e.g. "1" or "HoaLac").
        output_dir: Root directory; zoom-level subdirectories are created here.
        tile_size:  Tile edge length in pixels.
    """
    os.makedirs(output_dir, exist_ok=True)
    max_dim  = max(image.width, image.height)
    max_zoom = math.ceil(math.log(max_dim / tile_size, 2))

    for zoom in range(max_zoom + 1):
        zoom_dir = os.path.join(output_dir, str(zoom))
        os.makedirs(zoom_dir, exist_ok=True)

        scale  = 2 ** (max_zoom - zoom)
        sw = math.ceil(image.width  / scale)
        sh = math.ceil(image.height / scale)

        scaled = image.resize((sw, sh), Image.Resampling.LANCZOS)
        tasks = [
            (scaled, identifier, zoom_dir, zoom, x, y, tile_size)
            for x in range(0, sw, tile_size)
            for y in range(0, sh, tile_size)
        ]
        print(f"  zoom {zoom}/{max_zoom}: {len(tasks)} tiles …")
        with Pool(cpu_count()) as pool:
            pool.map(_process_tile, tasks)


def tile_satellite(cfg: DatasetConfig) -> None:
    """
    Tile the per-trajectory satellite TIF files found under
    <root>/video_map_DSMAC/Trajectory_<id>/flight_corridor_<id>.tif

    Each trajectory listed in cfg.all_ids is processed.
    """
    src_root = os.path.join(cfg.root, "video_map_DSMAC")
    for i in cfg.all_ids:
        file_dir   = os.path.join(src_root, f"Trajectory_{i}")
        tile_dir   = os.path.join(file_dir, "tile")
        image_path = os.path.join(file_dir, f"flight_corridor_{i}.tif")

        if not os.path.exists(image_path):
            print(f"WARNING: {image_path} not found, skipping trajectory {i}.")
            continue

        image = read_tif(image_path)
        if image is None:
            continue
        print(f"Tiling trajectory {i}: {image.width}×{image.height} px")
        _tile_image(image, str(i), tile_dir, cfg.tile_size)

    print("Tiling per-trajectory satellites done.")


def tile_large_satellite(tif_path: str, cfg: DatasetConfig) -> bool:
    """
    Tile the single large overview satellite GeoTIFF that covers all trajectories.

    Args:
        tif_path: Path to the large satellite TIF.
        cfg:      DatasetConfig (uses cfg.satellite_key, cfg.satellite_tile_dir,
                  cfg.tile_size).

    Returns:
        True on success, False on failure.
    """
    if not os.path.exists(tif_path):
        print(f"ERROR: Large satellite image not found: {tif_path}")
        return False

    image = read_tif(tif_path)
    if image is None:
        return False

    print(f"Large satellite: {image.width}×{image.height} px")
    _tile_image(image, cfg.satellite_key, cfg.satellite_tile_dir, cfg.tile_size)
    print(f"Tiling complete → {cfg.satellite_tile_dir}")
    return True


# ---------------------------------------------------------------------------
# Step 3: Copy data into unified drone/ and satellite/ directories
# ---------------------------------------------------------------------------

def copy_satellite(cfg: DatasetConfig) -> None:
    """
    Copy the selected zoom-level tiles from <root>/tile/<satellite_key>/
    into <root>/satellite/.
    """
    dst_dir  = os.path.join(cfg.root, "satellite")
    os.makedirs(dst_dir, exist_ok=True)

    tile_dir = cfg.satellite_tile_dir
    zoom_list = sorted(int(z) for z in os.listdir(tile_dir))
    sl = cfg.zoom_keep_slice
    selected = zoom_list[sl[0]:sl[1]]

    for zoom in selected:
        zoom_dir = os.path.join(tile_dir, str(zoom))
        for fname in os.listdir(zoom_dir):
            src = os.path.join(zoom_dir, fname)
            if os.path.isfile(src):
                shutil.copy(src, dst_dir)

    print(f"Copy satellite done ({len(selected)} zoom levels).")


def copy_drone(cfg: DatasetConfig) -> None:
    """
    Copy all drone images for every trajectory in cfg.all_ids from
    <root>/drone_og/<folder>/ into <root>/drone/images/.
    """
    dst_dir      = os.path.join(cfg.root, "drone", "images")
    os.makedirs(dst_dir, exist_ok=True)

    drone_og_path = os.path.join(cfg.root, "drone_og")
    dir_mapping   = {i: d for i, d in enumerate(sorted(os.listdir(drone_og_path)))}

    for i in cfg.all_ids:
        idx = i - 1
        if idx not in dir_mapping:
            print(f"WARNING: No drone directory for trajectory {i}, skipping.")
            continue
        folder = dir_mapping[idx]
        src_dir = os.path.join(drone_og_path, folder)
        for fname in os.listdir(src_dir):
            src = os.path.join(src_dir, fname)
            if os.path.isfile(src):
                shutil.copy(src, dst_dir)

    print("Copy drone done.")


# ---------------------------------------------------------------------------
# Step 4: Build drone↔satellite label pairs
# ---------------------------------------------------------------------------

def _tile_center_latlon(lt_lat: float, lt_lon: float,
                        rb_lat: float, rb_lon: float,
                        sate_h: int, sate_w: int,
                        zoom: int, tile_x: int, tile_y: int,
                        tile_size: int) -> Tuple[float, float]:
    """Return the geographic centre (lat, lon) of a tile at the given zoom."""
    max_dim  = max(sate_h, sate_w)
    max_zoom = math.ceil(math.log(max_dim / tile_size, 2))
    scale    = 2 ** (max_zoom - zoom)

    scaled_w = math.ceil(sate_w / scale)
    scaled_h = math.ceil(sate_h / scale)

    coe_lon = (tile_x + 0.5) * tile_size / scaled_w
    coe_lat = (tile_y + 0.5) * tile_size / scaled_h

    center_lat = lt_lat - coe_lat * (lt_lat - rb_lat)
    center_lon = lt_lon + coe_lon * (rb_lon - lt_lon)
    return center_lat, center_lon


def _tile_name_to_latlon(tile_name: str, cfg: DatasetConfig) -> Tuple[float, float]:
    """Parse a tile filename and return its centre lat/lon using *cfg*."""
    base = tile_name.replace(".png", "")
    parts = base.split("_")
    # Tile filenames: <identifier>_<zoom>_<tx:03>_<ty:03>.png
    # identifier may itself contain underscores, so parse from the right
    tile_y, tile_x, zoom = int(parts[-1]), int(parts[-2]), int(parts[-3])
    return _tile_center_latlon(
        cfg.satellite_latlon[0], cfg.satellite_latlon[1],
        cfg.satellite_latlon[2], cfg.satellite_latlon[3],
        cfg.satellite_size[0], cfg.satellite_size[1],
        zoom, tile_x, tile_y, cfg.tile_size,
    )


def _tile_expand(str_i: str, cur_tile_x: int, cur_tile_y: int,
                 p_img_xy_scale, zoom_level: int,
                 tile_x_max: int, tile_y_max: int,
                 cfg: DatasetConfig,
                 debug: bool = False):
    """
    Expand from the current tile position and return lists of overlapping tiles
    together with their IoU weights and centre lat/lon coordinates.

    Returns:
        (iou_tiles, iou_weights, iou_latlons,
         semi_tiles, semi_weights, semi_latlons)
    """
    tile_size = cfg.tile_size

    tile_u = max(0, cur_tile_y - 5)
    tile_d = min(cur_tile_y + 5, tile_y_max)
    tile_l = max(0, cur_tile_x - 5)
    tile_r = min(cur_tile_x + 5, tile_x_max)

    p_ordered = order_points(p_img_xy_scale)
    poly_p    = Polygon(p_ordered)
    poly_p_area = poly_p.area

    iou_list, iou_w, iou_ll   = [], [], []
    semi_list, semi_w, semi_ll = [], [], []

    for tx in range(tile_l, tile_r + 1):
        for ty in range(tile_u, tile_d + 1):
            corners = [
                (tx * tile_size,       ty * tile_size),
                ((tx + 1) * tile_size, ty * tile_size),
                (tx * tile_size,       (ty + 1) * tile_size),
                ((tx + 1) * tile_size, (ty + 1) * tile_size),
            ]
            poly_tile = Polygon(order_points(corners))
            poly_tile_area = poly_tile.area

            intersect = calc_intersect_area(poly_p, poly_tile)
            oc = intersect / min(poly_p_area, poly_tile_area)
            iou = intersect / (poly_p_area + poly_tile_area - intersect)

            if debug:
                print(f"  zoom={zoom_level} ({tx},{ty}) iou={iou:.3f}")

            tile_name = (f"{str_i}_{zoom_level}"
                         f"_{tx:03}_{ty:03}.png")

            if iou > cfg.threshold:
                iou_list.append(tile_name)
                iou_w.append(iou)
                iou_ll.append(_tile_name_to_latlon(tile_name, cfg))

            if iou > cfg.semi_threshold:
                semi_list.append(tile_name)
                semi_w.append(iou)
                semi_ll.append(_tile_name_to_latlon(tile_name, cfg))

    return iou_list, iou_w, iou_ll, semi_list, semi_w, semi_ll


def _process_per_image(args) -> Optional[dict]:
    """
    Worker function (called in a ProcessPoolExecutor).

    args is a tuple:
        (cfg, str_i, file_dir, drone_img,
         lat, lon, height, phi, fov_h, fov_v)
    """
    (cfg, str_i, file_dir, drone_img,
     lat, lon, height, phi, fov_h, fov_v) = args

    debug = False

    p_latlon = calculate_coverage_endpoints(
        heading_angle=phi, height=height,
        cur_lat=lat, cur_lon=lon,
        fov_horizontal=fov_h, fov_vertical=fov_v,
        debug=debug,
    )

    tile_dir = cfg.satellite_tile_dir
    zoom_list = sorted(int(z) for z in os.listdir(tile_dir))
    zoom_max  = zoom_list[-1]
    sl = cfg.zoom_keep_slice
    zoom_list = zoom_list[sl[0]:sl[1]]

    sate_lt_lat, sate_lt_lon = cfg.satellite_latlon[0], cfg.satellite_latlon[1]
    sate_rb_lat, sate_rb_lon = cfg.satellite_latlon[2], cfg.satellite_latlon[3]
    sate_h, sate_w = cfg.satellite_size[0], cfg.satellite_size[1]

    cur_x, cur_y = geo_to_image_coords(lat, lon,
                                        sate_lt_lat, sate_lt_lon,
                                        sate_rb_lat, sate_rb_lon,
                                        sate_h, sate_w)
    p_img_xy = [
        geo_to_image_coords(v[0], v[1],
                             sate_lt_lat, sate_lt_lon,
                             sate_rb_lat, sate_rb_lon,
                             sate_h, sate_w)
        for v in p_latlon.values()
    ]

    result = {
        "str_i":        str_i,
        "drone_img_dir": file_dir,
        "drone_img":    drone_img,
        "lat":          lat,
        "lon":          lon,
        "sate_img_dir": os.path.join(cfg.root, "satellite"),
        "pair_pos_sate_img_list":               [],
        "pair_pos_sate_weight_list":            [],
        "pair_pos_sate_loc_lat_lon_list":       [],
        "pair_pos_semipos_sate_img_list":       [],
        "pair_pos_semipos_sate_weight_list":    [],
        "pair_pos_semipos_sate_loc_lat_lon_list": [],
    }

    for zoom_level in zoom_list:
        scale     = 2 ** (zoom_max - zoom_level)
        sw_scaled = math.ceil(sate_w / scale)
        sh_scaled = math.ceil(sate_h / scale)

        tx_max = sw_scaled // cfg.tile_size
        ty_max = sh_scaled // cfg.tile_size

        cx_s = math.ceil(cur_x / scale)
        cy_s = math.ceil(cur_y / scale)
        pxy_s = [(math.ceil(v[0] / scale), math.ceil(v[1] / scale))
                 for v in p_img_xy]

        cur_tx = cx_s // cfg.tile_size
        cur_ty = cy_s // cfg.tile_size

        (iou_list, iou_w, iou_ll,
         semi_list, semi_w, semi_ll) = _tile_expand(
            cfg.satellite_key, cur_tx, cur_ty, pxy_s,
            zoom_level, tx_max, ty_max, cfg, debug,
        )

        result["pair_pos_sate_img_list"].extend(iou_list)
        result["pair_pos_sate_weight_list"].extend(iou_w)
        result["pair_pos_sate_loc_lat_lon_list"].extend(iou_ll)
        result["pair_pos_semipos_sate_img_list"].extend(semi_list)
        result["pair_pos_semipos_sate_weight_list"].extend(semi_w)
        result["pair_pos_semipos_sate_loc_lat_lon_list"].extend(semi_ll)

    if not result["pair_pos_semipos_sate_img_list"]:
        return None
    return result


def _save_pairs_meta_data(pairs_list: list, pkl_path: str, _pair_dir: str) -> None:
    """Serialise the drone↔satellite pair list to a pickle file."""
    sate_img_dir_key = "sate_img_dir"
    pairs_to_save = []
    for pair in pairs_list:
        sate_dir = pair[sate_img_dir_key]
        has_any = any(
            os.path.exists(os.path.join(sate_dir, s))
            for s in (pair["pair_pos_sate_img_list"]
                      + pair["pair_pos_semipos_sate_img_list"])
        )
        if has_any:
            pairs_to_save.append(pair)

    with open(pkl_path, "wb") as f:
        pickle.dump({
            "pairs_drone2sate_list":         pairs_to_save,
            "pairs_iou_sate2drone_dict":     {},
            "pairs_iou_drone2sate_dict":     {},
            "pairs_iou_match_set":           set(),
            "pairs_semi_iou_sate2drone_dict": {},
            "pairs_semi_iou_drone2sate_dict": {},
            "pairs_semi_iou_match_set":       set(),
        }, f)


def _write_json(pickle_root: str, root: str, split_type: str) -> None:
    """Convert a pair pickle to a JSON file expected by the dataloader."""
    for split in ("train", "test"):
        pkl_path = os.path.join(pickle_root, f"{split}_pair_meta.pkl")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        records = []
        for p in data["pairs_drone2sate_list"]:
            records.append({
                "drone_img_dir":   "drone/images",
                "drone_img_name":  p["drone_img"],
                "drone_loc_lat_lon": (p["lat"], p["lon"]),
                "sate_img_dir":    "satellite",
                "pair_pos_sate_img_list":               p["pair_pos_sate_img_list"],
                "pair_pos_sate_weight_list":            p["pair_pos_sate_weight_list"],
                "pair_pos_sate_loc_lat_lon_list":       p["pair_pos_sate_loc_lat_lon_list"],
                "pair_pos_semipos_sate_img_list":       p["pair_pos_semipos_sate_img_list"],
                "pair_pos_semipos_sate_weight_list":    p["pair_pos_semipos_sate_weight_list"],
                "pair_pos_semipos_sate_loc_lat_lon_list": p["pair_pos_semipos_sate_loc_lat_lon_list"],
                "drone_metadata": {
                    "height":     None,
                    "drone_roll": None, "drone_pitch": None, "drone_yaw": None,
                    "cam_roll":   None, "cam_pitch":   None, "cam_yaw":   None,
                },
            })

        save_path = os.path.join(root, f"{split_type}-drone2sate-{split}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4, ensure_ascii=False)
        print(f"  JSON written: {save_path} ({len(records)} pairs)")


def process_visloc_data(cfg: DatasetConfig) -> None:
    """
    Main label-building step: match every drone image to satellite tiles.

    All parameters (paths, split IDs, thresholds, satellite bounds) are
    taken from *cfg* 
    """
    os.makedirs(cfg.save_root, exist_ok=True)

    drone_og_path = os.path.join(cfg.root, "drone_og")
    dir_mapping   = {i: d for i, d in enumerate(sorted(os.listdir(drone_og_path)))}

    train_args: List[tuple] = []
    test_args:  List[tuple] = []
    all_args:   List[tuple] = []

    for i in cfg.all_ids:
        idx = i - 1
        if idx not in dir_mapping:
            print(f"ERROR: No trajectory directory for index {i}, skipping.")
            continue

        folder_name = dir_mapping[idx]
        file_dir    = os.path.join(drone_og_path, folder_name)
        csv_name    = os.path.basename(file_dir)
        drone_csv   = os.path.join(file_dir, f"{csv_name}.csv")

        if not os.path.exists(drone_csv):
            print(f"WARNING: CSV not found: {drone_csv}, skipping trajectory {i}.")
            continue

        str_i = str(i)
        with open(drone_csv, newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            for row in reader:
                args = (
                    cfg, str_i, file_dir,
                    row[1],            # filename
                    float(row[3]),     # lat
                    float(row[4]),     # lon
                    float(row[5]),     # rel_alt
                    float(row[7]),     # phi
                    float(row[8]),     # FOV_H
                    float(row[9]),     # FOV_V
                )
                if cfg.split_type == "cross-area":
                    if i in cfg.train_ids:
                        train_args.append(args)
                    else:
                        test_args.append(args)
                else:
                    all_args.append(args)

    def _run_parallel(arg_list, desc):
        results = []
        with concurrent.futures.ProcessPoolExecutor() as ex:
            for r in tqdm(ex.map(_process_per_image, arg_list),
                          total=len(arg_list), desc=desc):
                results.append(r)
        return [r for r in results if r is not None]

    if cfg.split_type == "same-area":
        print(f"same-area: {len(all_args)} images")
        random.shuffle(all_args)
        data = _run_parallel(all_args, "same-area")
        split = len(data) * 4 // 5
        train_data, test_data = data[:split], data[split:]
    else:
        print(f"cross-area: {len(train_args)} train / {len(test_args)} test images")
        train_data = _run_parallel(train_args, "train")
        test_data  = _run_parallel(test_args,  "test")

    train_pkl = os.path.join(cfg.save_root, "train_pair_meta.pkl")
    test_pkl  = os.path.join(cfg.save_root, "test_pair_meta.pkl")
    _save_pairs_meta_data(train_data, train_pkl, os.path.join(cfg.save_root, "train"))
    _save_pairs_meta_data(test_data,  test_pkl,  os.path.join(cfg.save_root, "test"))

    _write_json(cfg.save_root, cfg.root, cfg.split_type)
    print("process_visloc_data done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Paths
    p.add_argument("--root",      required=True,
                   help="Dataset root directory (e.g. /data/UAV1)")
    p.add_argument("--save-root", required=True,
                   help="Output directory for processed labels")

    # Split
    p.add_argument("--split", default="cross-area",
                   choices=["cross-area", "same-area"],
                   help="Dataset split strategy")
    p.add_argument("--train-ids", nargs="+", type=int, default=[1, 3],
                   help="Trajectory IDs used for training")
    p.add_argument("--test-ids",  nargs="+", type=int, default=[4],
                   help="Trajectory IDs used for testing")

    # Satellite overview bounds
    p.add_argument("--satellite-key", default="HoaLac",
                   help="Identifier for the overview satellite image")
    p.add_argument("--satellite-lt-lat", type=float, default=21.0152911)
    p.add_argument("--satellite-lt-lon", type=float, default=105.5161285)
    p.add_argument("--satellite-rb-lat", type=float, default=20.9909321)
    p.add_argument("--satellite-rb-lon", type=float, default=105.5518341)
    p.add_argument("--satellite-h", type=int,   default=13312,
                   help="Overview satellite pixel height")
    p.add_argument("--satellite-w", type=int,   default=9728,
                   help="Overview satellite pixel width")

    # Tile parameters
    p.add_argument("--tile-size",  type=int,   default=384)
    p.add_argument("--zoom-keep-start", type=int, default=-3,
                   help="Start index for zoom-level slice (negative = from end)")
    p.add_argument("--zoom-keep-end",   type=int, default=-1,
                   help="End index for zoom-level slice (negative = from end)")

    # Matching thresholds
    p.add_argument("--threshold",      type=float, default=0.39)
    p.add_argument("--semi-threshold", type=float, default=0.14)

    # Camera / sensor
    p.add_argument("--fov-h",           type=float, default=84.0)
    p.add_argument("--fov-v",           type=float, default=56.0)
    p.add_argument("--sensor-width-mm", type=float, default=36.0)
    p.add_argument("--sensor-height-mm",type=float, default=27.0)

    # Which pipeline steps to run
    p.add_argument(
        "--steps", nargs="+",
        choices=["csv", "tile", "tile-large", "copy", "label", "all"],
        default=["label"],
        help=(
            "Pipeline steps to execute:\n"
            "  csv        – build per-trajectory CSVs from SRT files\n"
            "  tile       – tile per-trajectory satellite TIFs\n"
            "  tile-large – tile the large overview satellite TIF\n"
            "  copy       – copy satellite tiles and drone images\n"
            "  label      – build drone↔satellite pair labels\n"
            "  all        – run all steps in order"
        ),
    )
    p.add_argument("--large-tif", default=None,
                   help="Path to the large overview satellite TIF (needed for tile-large)")

    return p


def _args_to_config(args: argparse.Namespace) -> DatasetConfig:
    return DatasetConfig(
        root=args.root,
        save_root=args.save_root,
        split_type=args.split,
        train_ids=args.train_ids,
        test_ids=args.test_ids,
        tile_size=args.tile_size,
        zoom_keep_slice=(args.zoom_keep_start, args.zoom_keep_end),
        threshold=args.threshold,
        semi_threshold=args.semi_threshold,
        fov_h=args.fov_h,
        fov_v=args.fov_v,
        sensor_width_mm=args.sensor_width_mm,
        sensor_height_mm=args.sensor_height_mm,
        satellite_key=args.satellite_key,
        satellite_latlon=[
            args.satellite_lt_lat, args.satellite_lt_lon,
            args.satellite_rb_lat, args.satellite_rb_lon,
        ],
        satellite_size=[args.satellite_h, args.satellite_w],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = _build_arg_parser()
    args   = parser.parse_args()
    cfg    = _args_to_config(args)

    steps = set(args.steps)
    run_all = "all" in steps

    print(f"\n=== DroneCVGL Preprocessing ===")
    print(f"  root       : {cfg.root}")
    print(f"  save_root  : {cfg.save_root}")
    print(f"  split      : {cfg.split_type}")
    print(f"  train_ids  : {cfg.train_ids}")
    print(f"  test_ids   : {cfg.test_ids}")
    print(f"  satellite  : {cfg.satellite_key} "
          f"({cfg.satellite_size[0]}×{cfg.satellite_size[1]})")
    print()

    if run_all or "csv" in steps:
        print("--- Step: generate CSVs from SRT ---")
        generate_all_csvs(cfg)

    if run_all or "tile" in steps:
        print("--- Step: tile per-trajectory satellites ---")
        tile_satellite(cfg)

    if run_all or "tile-large" in steps:
        print("--- Step: tile large overview satellite ---")
        large_tif = args.large_tif or os.path.join(
            cfg.root, "video_map_DSMAC",
            f"{cfg.satellite_key}_satellite_19.tif"
        )
        tile_large_satellite(large_tif, cfg)

    if run_all or "copy" in steps:
        print("--- Step: copy data ---")
        copy_satellite(cfg)
        copy_drone(cfg)

    if run_all or "label" in steps:
        print("--- Step: build drone↔satellite pair labels ---")
        process_visloc_data(cfg)

    print("\nAll requested steps complete.")