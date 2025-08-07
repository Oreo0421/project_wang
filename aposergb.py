import os
import h5py
import cv2
import numpy as np
from tqdm import tqdm

def extract_all_cameras_first_30_frames(smc_file_path, output_dir):
    """
    Extract the first 30 frames from each camera (0–47) in Camera_5mp using h5py.
    
    Args:
        smc_file_path (str): Path to the .smc file
        output_dir (str): Directory to save extracted images
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        with h5py.File(smc_file_path, 'r') as smc_file:
            print("Successfully opened the SMC file")

            if 'Camera_5mp' not in smc_file:
                print("Error: 'Camera_5mp' not found in SMC file.")
                return

            camera_group = smc_file['Camera_5mp']

            for cam_id in range(48):
                cam_str = str(cam_id)
                if cam_str not in camera_group:
                    print(f"[!] Camera {cam_id} not found.")
                    continue

                if 'color' not in camera_group[cam_str]:
                    print(f"[!] Color data missing for camera {cam_id}.")
                    continue

                color_group = camera_group[cam_str]['color']
                frame_ids = list(color_group.keys())
                frame_ids.sort(key=int)
                frame_ids = frame_ids[:30]

                # Make camera output folder like cam00, cam01, ...
                cam_folder = os.path.join(output_dir, f"{int(cam_id):02d}")
                os.makedirs(cam_folder, exist_ok=True)

                print(f"[✓] Extracting from camera {cam_id}, {len(frame_ids)} frames")

                for frame_id in tqdm(frame_ids, desc=f"{cam_id:02d}"):
                    compressed_data = color_group[frame_id][()]
                    img = cv2.imdecode(compressed_data, cv2.IMREAD_COLOR)

                    if img is None:
                        print(f"[x] Failed to decode frame {frame_id} from camera {cam_id}")
                        continue

                    out_path = os.path.join(cam_folder, f"{int(frame_id):08d}.jpg")
                    cv2.imwrite(out_path, img)

    except Exception as e:
        print(f"[!] Error: {str(e)}")


if __name__ == "__main__":
    smc_file_path = "/home/zhiyw/Desktop/DNA-randering-part1/dna-rendering-part1-apose/dna_rendering_part1_apose/apose_main/0165_apose02.smc"
    output_dir = "/home/zhiyw/Desktop/DNA-randering-part1/smc2obj/output_RGB"

    extract_all_cameras_first_30_frames(smc_file_path, output_dir)
