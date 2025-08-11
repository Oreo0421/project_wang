# Save this as enhanced_RGBcamera_inf.py
from ModifiedSMCReader import SMCReader
import numpy as np
import os
import json
import math

def getWorld2View(R, t):
    """
    Build basic world to view transformation matrix
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """
    Enhanced version with additional translation and scaling
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def view_to_world_transform(RT_view):
    """
    Convert view coordinates RT matrix to world coordinates
    RT_view: 4x4 view transformation matrix or 3x4 [R|t] matrix
    Returns: Camera to world transformation matrix (C2W)
    """
    if RT_view.shape == (3, 4):
        # Convert 3x4 [R|t] to 4x4 homogeneous matrix
        RT_homogeneous = np.zeros((4, 4))
        RT_homogeneous[:3, :4] = RT_view
        RT_homogeneous[3, 3] = 1.0
        RT_view = RT_homogeneous
    
    # RT_view is world-to-view, so C2W = inv(RT_view)
    C2W = np.linalg.inv(RT_view)
    return C2W

def extract_rotation_translation_from_RT(RT_matrix):
    """
    Extract R and t from RT matrix for world coordinates
    """
    if RT_matrix.shape == (3, 4):
        R = RT_matrix[:3, :3]
        t = RT_matrix[:3, 3]
    elif RT_matrix.shape == (4, 4):
        R = RT_matrix[:3, :3]
        t = RT_matrix[:3, 3]
    else:
        raise ValueError(f"Unsupported RT matrix shape: {RT_matrix.shape}")
    
    return R, t

def save_camera_parameters(camera_id, K, D, RT_view, output_dir):
    """
    Save camera parameters in both view and world coordinates
    """
    # Create directory for this camera
    cam_dir = os.path.join(output_dir, f"camera_{camera_id}")
    os.makedirs(cam_dir, exist_ok=True)
    
    # Save original view coordinate parameters
    view_dir = os.path.join(cam_dir, "view_coordinates")
    os.makedirs(view_dir, exist_ok=True)
    
    # Save world coordinate parameters
    world_dir = os.path.join(cam_dir, "world_coordinates")
    os.makedirs(world_dir, exist_ok=True)
    
    camera_data = {}
    
    # Save K (intrinsic matrix - same for both coordinate systems)
    if K is not None:
        np.save(os.path.join(view_dir, "K.npy"), K)
        np.save(os.path.join(world_dir, "K.npy"), K)
        np.savetxt(os.path.join(view_dir, "K.txt"), K, fmt='%.10f')
        np.savetxt(os.path.join(world_dir, "K.txt"), K, fmt='%.10f')
        camera_data['K'] = K.tolist()
    
    # Save D (distortion coefficients - same for both coordinate systems)
    if D is not None:
        np.save(os.path.join(view_dir, "D.npy"), D)
        np.save(os.path.join(world_dir, "D.npy"), D)
        np.savetxt(os.path.join(view_dir, "D.txt"), D, fmt='%.10f')
        np.savetxt(os.path.join(world_dir, "D.txt"), D, fmt='%.10f')
        camera_data['D'] = D.tolist()
    
    # Save RT in view coordinates (original)
    if RT_view is not None:
        np.save(os.path.join(view_dir, "RT.npy"), RT_view)
        np.savetxt(os.path.join(view_dir, "RT.txt"), RT_view, fmt='%.10f')
        camera_data['RT_view'] = RT_view.tolist()
        
        # Convert to world coordinates
        try:
            C2W = view_to_world_transform(RT_view)
            
            # Save C2W (camera to world transformation)
            np.save(os.path.join(world_dir, "C2W.npy"), C2W)
            np.savetxt(os.path.join(world_dir, "C2W.txt"), C2W, fmt='%.10f')
            camera_data['C2W'] = C2W.tolist()
            
            # Extract R and t for world coordinates
            R_world, t_world = extract_rotation_translation_from_RT(C2W)
            
            # Save world coordinate R and t separately
            np.save(os.path.join(world_dir, "R.npy"), R_world)
            np.save(os.path.join(world_dir, "t.npy"), t_world)
            np.savetxt(os.path.join(world_dir, "R.txt"), R_world, fmt='%.10f')
            np.savetxt(os.path.join(world_dir, "t.txt"), t_world, fmt='%.10f')
            
            camera_data['R_world'] = R_world.tolist()
            camera_data['t_world'] = t_world.tolist()
            
            # Create world-to-view transformation matrices using the provided functions
            W2V_basic = getWorld2View(R_world, t_world)
            np.save(os.path.join(world_dir, "W2V_basic.npy"), W2V_basic)
            np.savetxt(os.path.join(world_dir, "W2V_basic.txt"), W2V_basic, fmt='%.10f')
            camera_data['W2V_basic'] = W2V_basic.tolist()
            
            # Also save with optional translation and scale (default values)
            W2V_enhanced = getWorld2View2(R_world, t_world)
            np.save(os.path.join(world_dir, "W2V_enhanced.npy"), W2V_enhanced)
            np.savetxt(os.path.join(world_dir, "W2V_enhanced.txt"), W2V_enhanced, fmt='%.10f')
            camera_data['W2V_enhanced'] = W2V_enhanced.tolist()
            
        except Exception as e:
            print(f"Error converting RT to world coordinates for camera {camera_id}: {e}")
    
    return camera_data

# Path to .smc file
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
                print(f"\nProcessing Camera {camera_id}...")
                
                K = all_calibration[camera_id].get('K')
                D = all_calibration[camera_id].get('D')
                RT = all_calibration[camera_id].get('RT')
                Color_Calibration = all_calibration[camera_id].get('Color_Calibration')
                
                # Save camera parameters with coordinate transformation
                camera_data = save_camera_parameters(camera_id, K, D, RT, output_dir)
                
                # Handle Color_Calibration separately (if exists)
                if Color_Calibration is not None:
                    cam_dir = os.path.join(output_dir, f"camera_{camera_id}")
                    np.save(os.path.join(cam_dir, "Color_Calibration.npy"), Color_Calibration)
                    np.savetxt(os.path.join(cam_dir, "Color_Calibration.txt"), Color_Calibration, fmt='%.10f')
                    camera_data['Color_Calibration'] = Color_Calibration.tolist()
                
                # Add to summary
                summary[camera_id] = camera_data
                
                # Print the matrices for this camera
                if K is not None and D is not None and RT is not None:
                    print(f"Original View Coordinates:")
                    print(f"K (Intrinsic matrix):\n{K}")
                    print(f"D (Distortion coefficients):\n{D}")
                    print(f"RT (View coordinates):\n{RT}")
                    
                    if 'C2W' in camera_data:
                        C2W = np.array(camera_data['C2W'])
                        print(f"C2W (Camera to World):\n{C2W}")
                        print(f"Camera position in world coordinates: {C2W[:3, 3]}")
                else:
                    print(f"  Missing some calibration data for camera {camera_id}")
            
            # Save summary as JSON
            summary_path = os.path.join(output_dir, "calibration_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nAll calibration data saved to '{output_dir}' directory")
            print(f"Summary saved to '{summary_path}'")
            print("\nDirectory structure:")
            print("- view_coordinates/: Original SMC parameters")
            print("- world_coordinates/: Converted world coordinate parameters")
            print("  - C2W: Camera to world transformation matrix")
            print("  - R, t: Rotation matrix and translation vector in world coords")
            print("  - W2V_basic: World to view transformation (basic)")
            print("  - W2V_enhanced: World to view transformation (enhanced)")
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
        info_path = os.path.join(output_dir, "camera_info.json")
        with open(info_path, 'w') as f:
            json.dump(camera_info, f, indent=2)
        print(f"Camera information saved to '{info_path}'")
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