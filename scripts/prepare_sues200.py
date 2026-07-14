import os
import random
import shutil
from pathlib import Path

def split_sues200_dataset(data_root, num_train=150, seed=42):
    data_root = Path(data_root)
    drone_dir = data_root / "drone_view_512"
    sate_dir = data_root / "satellite-view"

    if not drone_dir.exists() or not sate_dir.exists():
        raise FileNotFoundError("Không tìm thấy thư mục 'drone' hoặc 'satellite' trong thư mục gốc!")

    all_places = sorted([f.name for f in drone_dir.iterdir() if f.is_dir()])
    
    total_places = len(all_places)
    print(f"Tìm thấy tổng cộng {total_places} địa điểm.")
    
    random.seed(seed)
    random.shuffle(all_places)

    train_places = all_places[:num_train]
    test_places = all_places[num_train:]

    print(f"--> Chọn {len(train_places)} địa điểm để TRAIN")
    print(f"--> Chọn {len(test_places)} địa điểm để TEST")

    views = ["drone_view_512", "satellite-view"]
    splits = ["train", "test"]
    
    for split in splits:
        for view in views:
            (data_root / split / view).mkdir(parents=True, exist_ok=True)

    print("\nĐang tiến hành phân chia dữ liệu...")
    
    for place in train_places:
        src_drone = drone_dir / place
        dst_drone = data_root / splits[0] / views[0] / place
        if src_drone.exists():
            shutil.move(str(src_drone), str(dst_drone))
            
        src_sate = sate_dir / place
        dst_sate = data_root / splits[0] / views[1] / place
        if src_sate.exists():
            shutil.move(str(src_sate), str(dst_sate))

    for place in test_places:
        src_drone = drone_dir / place
        dst_drone = data_root / splits[1] / views[0] / place
        if src_drone.exists():
            shutil.move(str(src_drone), str(dst_drone))
            
        src_sate = sate_dir / place
        dst_sate = data_root / splits[1] / views[1] / place
        if src_sate.exists():
            shutil.move(str(src_sate), str(dst_sate))

    try:
        if not any(drone_dir.iterdir()):
            drone_dir.rmdir()
        if not any(sate_dir.iterdir()):
            sate_dir.rmdir()
        print("\nĐã dọn dẹp các thư mục gốc trống.")
    except Exception:
        pass

    print("🎉 Hoàn thành phân chia dữ liệu thành công!")

if __name__ == "__main__":
    DATASET_ROOT_PATH = "/home/tts26/sonh/data/SUES-200-512x512" 
    
    split_sues200_dataset(
        data_root=DATASET_ROOT_PATH, 
        num_train=150, 
        seed=42
    )