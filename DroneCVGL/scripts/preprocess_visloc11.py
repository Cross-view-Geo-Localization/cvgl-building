#!/usr/bin/env python3
"""
UAV-VisLoc – Trích xuất patch vệ tinh khớp từng ảnh drone
---------------------------------------------------------

Mục tiêu
- Với mỗi ảnh drone (tọa độ, cao độ, yaw trong xx.csv), cắt 1 patch từ
  ảnh vệ tinh satelliteXX.tif sao cho:
  * Tâm patch đúng vị trí lat/lon của ảnh drone
  * Kích thước pixel của patch = kích thước pixel của ảnh drone
  * Patch được xoay theo yaw (Phi1 ưu tiên; nếu thiếu dùng Phi2)
  * Diện tích (độ phủ mặt đất) của patch tính gần đúng từ cao độ + FOV ngang

Giới hạn & giả định
- Ảnh vệ tinh là orthophoto (north-up). Ảnh drone có thể nghiêng; ta chỉ có
  thể khớp vị trí + hướng quay (yaw), KHÔNG tái tạo phối cảnh pitch/roll.
- Không có thông số camera của drone → mặc định FOV ngang = 70° (có thể đổi
  qua tham số --fov-deg). Nếu có sensor/focal thì sửa tham số này để diện tích
  cắt sát hơn với footprint của ảnh drone.

Phụ thuộc: rasterio, numpy, pandas, pillow, pyproj
Cài đặt: pip install rasterio numpy pandas pillow pyproj

Cách dùng ví dụ:
python uav_visloc_extract_satellite_patches.py \
  --root /path/to/UAV-VisLoc \
  --out  /path/to/output_patches \
  --fov-deg 70 \
  --yaw-source phi1 \
  --yaw-clockwise-from-north

Tham số quan trọng khác:
- --yaw-offset-deg: bù trừ góc nếu quy ước yaw của dữ liệu khác mong đợi.
- --padding-m: cộng thêm biên (mét) quanh footprint (mặc định 0).

Cấu trúc thư mục kỳ vọng (ví dụ cho cảnh 01):
UAV-VisLoc/
  01/
    satellite01.tif
    01.csv            # có cột: filename, lat, lon, height, Omega, Kappa, Phi1, Phi2
    drone/
      01_0001.JPG ...

Kết quả:
- Mỗi ảnh drone tạo 1 file PNG trong OUT/XX/ có tên <filename>_satpatch.png

"""
from __future__ import annotations
import argparse
import math
import os
import re
from pathlib import Path
from typing import Tuple, Optional
import tqdm

import numpy as np
import pandas as pd
from PIL import Image

import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import Resampling, reproject
from rasterio.vrt import WarpedVRT
from affine import Affine
from pyproj import CRS, Transformer

# -------------------------------
# Tiện ích địa lý
# -------------------------------

def auto_utm_crs(lon: float, lat: float) -> CRS:
    """Chọn UTM zone theo kinh độ/vĩ độ (WGS84)."""
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

# -------------------------------
# Đọc CSV linh hoạt
# -------------------------------

def read_drone_csv(csv_path: Path) -> pd.DataFrame:
    """Đọc xx.csv với header có thể khác nhau (comma hoặc whitespace).
    Trả về DataFrame với các cột chuẩn hóa: filename, lat, lon, height, phi1, phi2.
    """
    # Thử comma trước
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        # Thử whitespace
        df = pd.read_csv(csv_path, sep=None, engine="python")

    # Chuẩn hóa tên cột về lower-case bỏ khoảng trắng
    df.columns = [re.sub(r"\s+", "", c.lower()) for c in df.columns]

    # Map các alias cột
    col_map = {}
    # filename
    for c in df.columns:
        if c in {"filename", "image", "img", "name"}:
            col_map[c] = "filename"
    # lat/lon
    for c in df.columns:
        if c in {"lat", "latitude", "y"}:
            col_map[c] = "lat"
        if c in {"lon", "longitude", "long", "x"}:
            col_map[c] = "lon"
    # height
    for c in df.columns:
        if c in {"height", "alt", "altitude", "h"}:
            col_map[c] = "height"
    # phi1/phi2
    for c in df.columns:
        if c in {"phi1", "yaw1"}:
            col_map[c] = "phi1"
        if c in {"phi2", "yaw2"}:
            col_map[c] = "phi2"

    df = df.rename(columns=col_map)

    required = {"filename", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc trong {csv_path.name}: {sorted(missing)}")

    # Bổ sung cột nếu thiếu
    if "height" not in df.columns:
        df["height"] = np.nan
    if "phi1" not in df.columns:
        df["phi1"] = np.nan
    if "phi2" not in df.columns:
        df["phi2"] = np.nan

    return df

# -------------------------------
# Ảnh – chuyển dtype và xoay
# -------------------------------

def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Chuyển HxWxC sang uint8 nếu cần, scale theo percentiles để hiển thị tốt."""
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    out = np.empty(arr.shape, dtype=np.uint8)
    C = 1 if arr.ndim == 2 else arr.shape[2]
    if arr.ndim == 2:
        p2, p98 = np.percentile(arr, (2, 98))
        if p98 <= p2:
            p2, p98 = float(np.min(arr)), float(np.max(arr))
        if p98 == p2:
            return np.clip(arr, 0, 255).astype(np.uint8)
        band = (arr - p2) / (p98 - p2)
        out = np.clip(band * 255, 0, 255).astype(np.uint8)
        return out
    else:
        for b in range(C):
            p2, p98 = np.percentile(arr[..., b], (2, 98))
            if p98 <= p2:
                p2, p98 = float(np.min(arr[..., b])), float(np.max(arr[..., b]))
            if p98 == p2:
                out[..., b] = np.clip(arr[..., b], 0, 255).astype(np.uint8)
            else:
                band = (arr[..., b] - p2) / (p98 - p2)
                out[..., b] = np.clip(band * 255, 0, 255).astype(np.uint8)
        return out


def build_patch_transform(x: float, y: float, gw: float, gh: float, width_px: int, height_px: int, theta_ccw_deg: float) -> Affine:
    """Tạo Affine transform ánh xạ (col,row) → (x,y) UTM cho 1 patch:
    - Tâm patch tại (x,y)
    - Kích thước ground (gw x gh)
    - Kích thước pixel (width_px x height_px)
    - patch quay CCW = theta_ccw_deg so với trục East-North
    """
    sx = gw / max(1, width_px)
    sy = gh / max(1, height_px)
    # Thứ tự áp dụng (phải → trái):
    # dịch tâm về (0,0) → scale pixel→m (y âm để hàng tăng xuống dưới) → rotate CCW → dịch đến (x,y)
    T = (
        Affine.translation(x, y)
        * Affine.rotation(theta_ccw_deg)
        * Affine.scale(sx, -sy)
        * Affine.translation(-width_px / 2.0, -height_px / 2.0)
    )
    return T

def rotate_keep_center(img: Image.Image, angle_deg: float) -> Image.Image:
    """Xoay ảnh quanh tâm, giữ nguyên kích thước."""
    return img.rotate(angle_deg, resample=Image.BILINEAR, expand=False, center=(img.width / 2, img.height / 2))

# -------------------------------
# Tính footprint từ H + FOV
# -------------------------------

def footprint_from_height(width_px: int, height_px: int, height_m: Optional[float], fov_h_deg: float) -> Tuple[float, float]:
    """Trả về (ground_width_m, ground_height_m) tương ứng với khung hình ảnh.
    Nếu height_m là NaN → mặc định footprint theo FOV và H = 120 m.
    """
    if height_m is None or not np.isfinite(height_m):
        height_m = 120.0  # giả định hợp lý cho UAV mapping, có thể chỉnh bằng --default-height-m
    hFOV = math.radians(fov_h_deg)
    aspect = width_px / max(1, height_px)
    vFOV = 2.0 * math.atan(math.tan(hFOV / 2.0) / aspect)
    ground_w = 2.0 * height_m * math.tan(hFOV / 2.0)
    ground_h = 2.0 * height_m * math.tan(vFOV / 2.0)
    return ground_w, ground_h

# -------------------------------
# Pipeline chính
# -------------------------------

def process_scene(scene_dir: Path, out_dir: Path, meta: pd.DataFrame, args) -> None:
    """
    Xử lý từng cảnh (scene) trong UAV-VisLoc bằng cách:
    1. Tìm satelliteXX.tif nếu không sẽ dùng bản ghi đầu tiên trong csv của scene.
    3. Lấy 1 cặp lon/lat trung bình để xác định UTM
    4. Tính footprint (m) từ H + FOV
    5. Cắt ảnh vệ tinh bằng VRT và tạo patch (m) theo UTM
    6. Tâm theo UTM (m)
    7. Cửa sổ bounds theo mét

    9. Tính footprint (m) và cắt ảnh vệ tinh theo VRT
    10. Apply T cho UTM và convert back to lat/lon
    11. Save GPS coordinates to CSV
    """
    scene_name = scene_dir.name
    # Tìm satelliteXX.tif
    tif_candidates = list(scene_dir.glob("satellite*.tif"))
    print(tif_candidates)
    if not tif_candidates:
        print(f"[WARN] Không tìm thấy satellite*.tif trong {scene_dir}")
        return
    pattern = re.compile(r"^satellite[0-9]+\.tif$")
    for tif_path in tif_candidates:
        if pattern.fullmatch(tif_path.name):
            sat_path = tif_path
            break
    else:
        print(f"[WARN] Không tìm thấy satellite*.tif trong {scene_dir}")
        return
    # Chọn một tâm (lon,lat) đại diện để dựng VRT → chọn từ satellite_coordinates_range.csv nếu có,
    # nếu không sẽ dùng bản ghi đầu tiên trong csv của scene.
    csv_candidates = list(scene_dir.glob(f"{scene_name}.csv"))
    if not csv_candidates:
        print(f"[WARN] Không tìm thấy {scene_name}.csv trong {scene_dir}")
        return
    csv_path = csv_candidates[0]
    df = read_drone_csv(csv_path)
    # Lấy 1 cặp lon/lat trung bình để xác định UTM
    if sat_path.name in meta['map_name'].values:
        row_meta = meta[meta['map_name'] == sat_path.name].iloc[0]
        lon0 = float((row_meta['LT_lon_map'] + row_meta['RB_lon_map']) / 2.0)
        lat0 = float((row_meta['LT_lat_map'] + row_meta['RB_lat_map']) / 2.0)
    else:
        lon0 = float(df["lon"].astype(float).mean())
        lat0 = float(df["lat"].astype(float).mean())
    utm_crs = auto_utm_crs(lon0, lat0)
    wgs84 = CRS.from_epsg(4326)
    to_utm = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    from_utm = Transformer.from_crs(utm_crs, wgs84, always_xy=True)
    # Mở ảnh vệ tinh và tạo VRT ở CRS mét để cắt theo mét
    with rasterio.open(sat_path) as src:
        with WarpedVRT(src, crs=utm_crs, resampling=Resampling.bilinear) as vrt:
            
            out_scene = out_dir / scene_name
            out_scene = out_scene / "sat_patches"
            out_scene.mkdir(parents=True, exist_ok=True)

            drone_dir = scene_dir / "drone"
            for idx, row in tqdm.tqdm(df.iterrows()):
                fname = str(row["filename"]).strip()
                if not fname:
                    continue
                out_name = f"{Path(fname).stem}.png"

                # if any(out_name == f.name for f in out_scene.iterdir() if f.is_file()):
                #     continue
                drone_img_path = drone_dir / fname
                if not drone_img_path.exists():
                    # Không bắt buộc phải có ảnh drone để lấy kích thước; nếu thiếu sẽ dùng 1024x768
                    width_px, height_px = 1024, 768
                else:
                    try:
                        with Image.open(drone_img_path) as dimg:
                            width_px, height_px = dimg.size
                    except Exception:
                        width_px, height_px = 1024, 768

                lat = float(row["lat"])  # trung tâm ảnh drone
                lon = float(row["lon"])  # trung tâm ảnh drone
                H = float(row["height"]) if np.isfinite(row["height"]) else None

                # Chọn yaw
                yaw = None
                if args.yaw_source.lower() == "phi1":
                    yaw = float(row.get("phi1", np.nan))
                    if not np.isfinite(yaw):
                        yaw = float(row.get("phi2", 0.0))
                else:
                    yaw = float(row.get("phi2", np.nan))
                    if not np.isfinite(yaw):
                        yaw = float(row.get("phi1", 0.0))
                if not np.isfinite(yaw):
                    yaw = 0.0

                # Tính footprint (m)
                gw, gh = footprint_from_height(width_px, height_px, H if args.use_height else None, args.fov_deg)
                # Thêm padding nếu cần
                gw += 2 * args.padding_m
                gh += 2 * args.padding_m

                # Tâm theo UTM (m)
                x, y = to_utm.transform(lon, lat)

                # Cửa sổ bounds theo mét
                half_w = gw / 2.0
                half_h = gh / 2.0


                T = build_patch_transform(x, y, gw, gh, int(round(width_px)), int(round(height_px)), -yaw)

                H_px = int(round(height_px))
                W_px = int(round(width_px))
                
                dst = np.zeros((int(round(height_px)), int(round(width_px)), vrt.count), dtype=np.float32)
                for b in range(1, vrt.count + 1):
                    reproject(
                        source=rasterio.band(vrt, b),
                        destination=dst[..., b - 1],
                        src_transform=vrt.transform,
                        src_crs=vrt.crs,
                        dst_transform=T,
                        dst_crs=vrt.crs,
                        resampling=Resampling.bilinear,
                        dst_nodata=0,
                    )
                rows = np.arange(H_px)
                cols = np.arange(W_px)
                c_grid, r_grid = np.meshgrid(cols + 0.5, rows + 0.5)
                
                # Apply T for UTM
                xs = T.a * c_grid + T.b * r_grid + T.c
                ys = T.d * c_grid + T.e * r_grid + T.f
                # Convert back to lat/lon
                lons, lats = from_utm.transform(xs, ys)

                #save gps coordinates to csv
                gps_dir = out_dir / scene_name / "sat_gps_coordinates"
                gps_dir.mkdir(parents=True, exist_ok=True)
                out_path = gps_dir / f"{Path(fname).stem}"
                gps_map = np.stack([lats, lons]).transpose(1,2,0).astype(np.float32)
                np.save(out_path, gps_map)
                                 
                img_arr = dst
                # Chuyển sang uint8 để xoay/lưu PNG
                img_u8 = to_uint8(img_arr)
                pil_img = Image.fromarray(img_u8)


                out_path = out_scene / out_name
                # pil_img.save("bla.png")
                # print(f"[OK] {scene_name}/{fname} → {out_path.relative_to(out_dir)}")


# -------------------------------
# main
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Crop & rotate satellite patches to match drone images (UAV-VisLoc)")
    parser.add_argument("--root", type=Path, required=True, help="Thư mục gốc UAV-VisLoc")
    parser.add_argument("--out", type=Path, required=True, help="Thư mục lưu patch đầu ra")
    parser.add_argument("--fov-deg", type=float, default=55.0, help="FOV ngang giả định của camera drone (độ)")
    parser.add_argument("--default-height-m", type=float, default=500.0, help="Cao độ mặc định nếu CSV thiếu (m)")
    parser.add_argument("--use-height", action="store_true", help="Dùng trường height từ CSV để tính footprint")
    parser.add_argument("--padding-m", type=float, default=0.0, help="Biên cộng thêm quanh footprint (m)")

    parser.add_argument("--yaw-source", type=str, choices=["phi1", "phi2"], default="phi1", help="Ưu tiên yaw lấy từ cột nào")
    parser.add_argument("--yaw-clockwise-from-north", dest="yaw_clockwise_from_north", action="store_true", help="Yaw dương = quay theo chiều kim đồng hồ từ hướng Bắc")
    parser.add_argument("-7-no-yaw-clockwise-from-north", dest="yaw_clockwise_from_north", action="store_false")
    parser.set_defaults(yaw_clockwise_from_north=True)
    parser.add_argument("--yaw-offset-deg", type=float, default=0.0, help="Bù trừ yaw (độ), dùng khi quy ước dấu/gốc khác nhau")

    args = parser.parse_args()

    # Ghi đè default height nếu người dùng cung cấp
    global DEFAULT_HEIGHT
    DEFAULT_HEIGHT = args.default_height_m
    # selected_scenes = ['01', '02', '03', '04', '05', '08', '11']
    selected_scenes = ['01']
    scenes = [p for p in sorted(args.root.iterdir()) if p.is_dir() and re.fullmatch(r"\d+", p.name) and p.name in selected_scenes]
    if not scenes:
        raise SystemExit("Không thấy thư mục cảnh (ví dụ 01, 02, 03, ...) trong --root")

    args.out.mkdir(parents=True, exist_ok=True)
    
    meta = pd.read_csv("satellite_ coordinates_range.csv")
    
    for scene_dir in scenes:
        process_scene(scene_dir, args.out, meta, args)

if __name__ == "__main__":
    main()
