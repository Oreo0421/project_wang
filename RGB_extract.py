import os
import cv2
from tqdm import tqdm
from SMCReader import SMCReader

# === CONFIG ===
smc_path = "/home/zhiyw/Desktop/DNA-randering-part1/dna-rendering-part1-apose/dna_rendering_part1_apose/apose_main/0165_apose02.smc"
output_root = "output_RGB/subject00"
os.makedirs(output_root, exist_ok=True)

# === Init Reader ===
reader = SMCReader(smc_path)

# Default to 48 cameras (0–47), and assume 225 frames
num_cameras = 48
num_frames = 30

# === Extract RGB frames ===
for cam_id in range(num_cameras):
    cam_name = f"cam{cam_id:02d}"
    save_dir = os.path.join(output_root, cam_name)
    os.makedirs(save_dir, exist_ok=True)

    for frame_id in tqdm(range(num_frames), desc=f"{cam_name}"):
        try:
            img = reader.get_img(Camera_group='Camera_5mp', Camera_id=cam_id, Image_type='color', Frame_id=frame_id)
            if img is None:
                continue
            save_path = os.path.join(save_dir, f"{frame_id:03d}.jpg")
            cv2.imwrite(save_path, img)
        except Exception as e:
            print(f" Cam {cam_id}, Frame {frame_id}: {e}")

print(" Done: RGB extraction complete.")
