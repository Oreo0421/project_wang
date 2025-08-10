# Save this as RGBcamera_inf.py
from ModifiedSMCReader import SMCReader
import numpy as np
import os
import json

# Path to  .smc file
file_path = "/home/fzhi/fzt/dna/apose_main/0008_apose01.smc"

# Create a directory for saving calibration data
output_dir = "/home/fzhi/fzt/output"
os.makedirs(output_dir, exist_ok=True)

try:
    # Create a reader instance
    print("Loading .smc file...")
    reader = SMCReader(file_path)
    
    # Print available keys
    print("\nAvailable keys in .smc file:", reader.get_available_keys())
    
    # Check if Camera_Parameter exists
    if 'Camera_Parameter' not in reader.get_available_keys():
        print("Camera_Parameter not found in this file!")
    else:
        # Get calibration for all cameras
        print("Extracting calibration data...")
        all_calibration = reader.get_Calibration_all()
        
        if all_calibration:
            print(f"Found calibration data for {len(all_calibration)} cameras")
            
            # Create a summary dictionary
            summary = {}
            
            # Process each camera
            for camera_id in all_calibration:
                # Create directory for this camera
                cam_dir = os.path.join(output_dir, f"camera_{camera_id}")
                os.makedirs(cam_dir, exist_ok=True)
                
                camera_data = {}
                
                # Save each matrix type to a separate file
                for matrix_type in ['K', 'D', 'RT', 'Color_Calibration']:
                    matrix = all_calibration[camera_id][matrix_type]
                    
                    if matrix is not None:
                        if matrix_type == 'RT':
                            try:
                                # 转换为相机到世界坐标的变换矩阵
                                camera_to_world = reader.world_to_camera_to_camera_to_world(matrix)

                                # 保存世界坐标变换矩阵
                                np.save(os.path.join(cam_dir, "RT_world.npy"), camera_to_world)
                                np.savetxt(os.path.join(cam_dir, "RT_world.txt"), camera_to_world, fmt='%.10f')

                                # 添加到camera_data
                                camera_data['RT_world'] = camera_to_world.tolist()

                                # 提取相机位置和朝向信息
                                camera_position = camera_to_world[:3, 3]
                                camera_orientation = camera_to_world[:3, :3]

                                # 保存相机位置
                                np.save(os.path.join(cam_dir, "camera_position.npy"), camera_position)
                                np.savetxt(os.path.join(cam_dir, "camera_position.txt"), camera_position, fmt='%.10f')

                                # 保存相机朝向
                                np.save(os.path.join(cam_dir, "camera_orientation.npy"), camera_orientation)
                                np.savetxt(os.path.join(cam_dir, "camera_orientation.txt"), camera_orientation,
                                           fmt='%.10f')

                                # 添加到summary
                                camera_data['camera_position'] = camera_position.tolist()
                                camera_data['camera_orientation'] = camera_orientation.tolist()

                                print(
                                    f"  Camera {camera_id} - Position in world coordinates: [{camera_position[0]:.6f}, {camera_position[1]:.6f}, {camera_position[2]:.6f}]")

                            except Exception as e:
                                print(f"  Warning: Could not process RT matrix for camera {camera_id}: {str(e)}")
                        # Save as numpy array
                        np.save(os.path.join(cam_dir, f"{matrix_type}.npy"), matrix)
                        
                        # Save as text file for human readability
                        np.savetxt(os.path.join(cam_dir, f"{matrix_type}.txt"), matrix, fmt='%.10f')
                        
                        # Add to summary
                        camera_data[matrix_type] = matrix.tolist()
                    else:
                        print(f"  Skipping {matrix_type} for camera {camera_id} (not available)")
                
                # Add to summary
                summary[camera_id] = camera_data
                
                # Print the matrices for this camera (only if all are available)
                if all(all_calibration[camera_id][mt] is not None for mt in ['K', 'D', 'RT']):
                    print(f"\nCalibration for Camera {camera_id}:")
                    print(f"K (Intrinsic matrix):\n{all_calibration[camera_id]['K']}")
                    print(f"D (Distortion coefficients):\n{all_calibration[camera_id]['D']}")
                    print(f"RT (Extrinsic matrix):\n{all_calibration[camera_id]['RT']}")
                    if all_calibration[camera_id]['Color_Calibration'] is not None:
                        print(f"Color_Calibration:\n{all_calibration[camera_id]['Color_Calibration']}")
            
            # Save summary as JSON
            with open(os.path.join(output_dir, "calibration_summary.json"), 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nAll calibration data saved to '{output_dir}' directory")
            print(f"Summary saved to '{os.path.join(output_dir, 'calibration_summary.json')}'")
        else:
            print("No calibration data found")

    # Get camera information
    camera_info = {}
    
    camera_5mp_info = reader.get_Camera_5mp_info()
    if camera_5mp_info:
        camera_info["Camera_5mp"] = camera_5mp_info
        print("\nCamera 5MP Information:")
        print(f"Number of devices: {camera_5mp_info['num_device']}")
        print(f"Number of frames: {camera_5mp_info['num_frame']}")
        print(f"Resolution: {camera_5mp_info['resolution']}")
    
    camera_12mp_info = reader.get_Camera_12mp_info()
    if camera_12mp_info:
        camera_info["Camera_12mp"] = camera_12mp_info
        print("\nCamera 12MP Information:")
        print(f"Number of devices: {camera_12mp_info['num_device']}")
        print(f"Number of frames: {camera_12mp_info['num_frame']}")
        print(f"Resolution: {camera_12mp_info['resolution']}")

    # Save camera information
    if camera_info:
        with open(os.path.join(output_dir, "camera_info.json"), 'w') as f:
            json.dump(camera_info, f, indent=2)
        print(f"Camera information saved to '{os.path.join(output_dir, 'camera_info.json')}'")
    else:
        print("No camera information found")

except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

finally:
    # Make sure to release resources
    if 'reader' in locals():
        reader.release()
        print("Resources released")
