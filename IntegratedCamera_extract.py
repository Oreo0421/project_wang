# Enhanced RGBcamera_inf.py with world coordinate transformation
from ModifiedSMCReader import SMCReader
import numpy as np
import os
import json
import math

class CameraCalibrationProcessor:
    """处理相机标定数据并进行坐标系转换的类"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def world_to_camera_to_camera_to_world(self, RT_matrix):
        """
        将世界坐标到相机坐标的RT矩阵转换为相机坐标到世界坐标的变换矩阵
        
        参数:
        RT_matrix: 4x4 或 3x4 的变换矩阵，表示世界坐标到相机坐标的变换
        
        返回:
        camera_to_world: 4x4 变换矩阵，表示相机坐标到世界坐标的变换
        """
        # 确保RT_matrix是4x4格式
        if RT_matrix.shape == (3, 4):
            RT_4x4 = np.vstack([RT_matrix, [0, 0, 0, 1]])
        else:
            RT_4x4 = RT_matrix.copy()
        
        # 提取旋转矩阵R和平移向量t
        R = RT_4x4[:3, :3]  # 3x3 旋转矩阵
        t = RT_4x4[:3, 3]   # 3x1 平移向量
        
        # 计算逆变换：相机坐标到世界坐标
        R_inv = R.T  # 旋转矩阵的逆等于其转置
        t_inv = -R_inv @ t  # 新的平移向量
        
        # 构建相机到世界坐标的4x4变换矩阵
        camera_to_world = np.eye(4)
        camera_to_world[:3, :3] = R_inv
        camera_to_world[:3, 3] = t_inv
        
        return camera_to_world
    
    def getWorld2View(self, R, t):
        """
        构建世界坐标到视图坐标的变换矩阵（兼容3D Gaussian Splatting格式）
        """
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0
        return np.float32(Rt)
    
    def getWorld2View2(self, R, t, translate=np.array([.0, .0, .0]), scale=1.0):
        """
        构建世界坐标到视图坐标的变换矩阵（带缩放和平移）
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
    
    def focal2fov(self, focal, pixels):
        """焦距转视场角"""
        return 2 * math.atan(pixels / (2 * focal))
    
    def fov2focal(self, fov, pixels):
        """视场角转焦距"""
        return pixels / (2 * math.tan(fov / 2))
    
    def extract_camera_parameters(self, K_matrix, image_width, image_height):
        """
        从内参矩阵中提取相机参数
        """
        fx = K_matrix[0, 0]
        fy = K_matrix[1, 1]
        cx = K_matrix[0, 2]
        cy = K_matrix[1, 2]
        
        # 计算视场角
        fov_x = self.focal2fov(fx, image_width)
        fov_y = self.focal2fov(fy, image_height)
        
        return {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'fov_x': fov_x,
            'fov_y': fov_y,
            'width': image_width,
            'height': image_height
        }
    
    def process_camera_calibration(self, camera_id, calibration_data, camera_info=None):
        """
        处理单个相机的标定数据
        """
        # 创建相机目录
        cam_dir = os.path.join(self.output_dir, f"camera_{camera_id}")
        os.makedirs(cam_dir, exist_ok=True)
        
        camera_data = {}
        processed_data = {}
        
        # 处理基础矩阵
        for matrix_type in ['K', 'D', 'RT', 'Color_Calibration']:
            matrix = calibration_data.get(matrix_type)
            
            if matrix is not None:
                # 保存原始矩阵
                np.save(os.path.join(cam_dir, f"{matrix_type}.npy"), matrix)
                np.savetxt(os.path.join(cam_dir, f"{matrix_type}.txt"), matrix, fmt='%.10f')
                camera_data[matrix_type] = matrix.tolist()
                
                print(f"  ✓ Saved {matrix_type} matrix for camera {camera_id}")
            else:
                print(f"  ⚠ Skipping {matrix_type} for camera {camera_id} (not available)")
        
        # 特殊处理RT矩阵 - 生成世界坐标相关数据
        if 'RT' in calibration_data and calibration_data['RT'] is not None:
            try:
                RT_matrix = calibration_data['RT']
                
                # 转换为相机到世界坐标的变换矩阵
                camera_to_world = self.world_to_camera_to_camera_to_world(RT_matrix)
                
                # 保存世界坐标变换矩阵
                np.save(os.path.join(cam_dir, "RT_world.npy"), camera_to_world)
                np.savetxt(os.path.join(cam_dir, "RT_world.txt"), camera_to_world, fmt='%.10f')
                camera_data['RT_world'] = camera_to_world.tolist()
                
                # 提取相机位置和朝向
                camera_position = camera_to_world[:3, 3]
                camera_orientation = camera_to_world[:3, :3]
                
                # 保存相机位置和朝向
                np.save(os.path.join(cam_dir, "camera_position.npy"), camera_position)
                np.savetxt(os.path.join(cam_dir, "camera_position.txt"), camera_position, fmt='%.10f')
                np.save(os.path.join(cam_dir, "camera_orientation.npy"), camera_orientation)
                np.savetxt(os.path.join(cam_dir, "camera_orientation.txt"), camera_orientation, fmt='%.10f')
                
                camera_data['camera_position'] = camera_position.tolist()
                camera_data['camera_orientation'] = camera_orientation.tolist()
                
                # 生成兼容3D Gaussian Splatting的格式
                if RT_matrix.shape == (4, 4):
                    R = RT_matrix[:3, :3]
                    t = RT_matrix[:3, 3]
                elif RT_matrix.shape == (3, 4):
                    R = RT_matrix[:3, :3]
                    t = RT_matrix[:3, 3]
                
                world2view = self.getWorld2View(R, t)
                world2view2 = self.getWorld2View2(R, t)
                
                # 保存3DGS兼容格式
                np.save(os.path.join(cam_dir, "world2view.npy"), world2view)
                np.savetxt(os.path.join(cam_dir, "world2view.txt"), world2view, fmt='%.10f')
                np.save(os.path.join(cam_dir, "world2view2.npy"), world2view2)
                np.savetxt(os.path.join(cam_dir, "world2view2.txt"), world2view2, fmt='%.10f')
                
                camera_data['world2view'] = world2view.tolist()
                camera_data['world2view2'] = world2view2.tolist()
                
                print(f"  ✓ Camera {camera_id} world position: [{camera_position[0]:.6f}, {camera_position[1]:.6f}, {camera_position[2]:.6f}]")
                
            except Exception as e:
                print(f"  ❌ Error processing RT matrix for camera {camera_id}: {str(e)}")
        
        # 处理内参矩阵 - 提取相机参数
        if 'K' in calibration_data and calibration_data['K'] is not None and camera_info:
            try:
                K_matrix = calibration_data['K']
                
                # 获取图像分辨率
                resolution = None
                if camera_info.get('Camera_5mp') and int(camera_id) < 48:
                    resolution = camera_info['Camera_5mp']['resolution']
                elif camera_info.get('Camera_12mp') and int(camera_id) >= 48:
                    resolution = camera_info['Camera_12mp']['resolution']
                
                if resolution:
                    width, height = resolution
                    camera_params = self.extract_camera_parameters(K_matrix, width, height)
                    
                    # 保存相机参数
                    with open(os.path.join(cam_dir, "camera_params.json"), 'w') as f:
                        json.dump(camera_params, f, indent=2)
                    
                    camera_data['camera_params'] = camera_params
                    print(f"  ✓ Extracted camera parameters: FOV_X={math.degrees(camera_params['fov_x']):.1f}°, FOV_Y={math.degrees(camera_params['fov_y']):.1f}°")
                    
            except Exception as e:
                print(f"  ❌ Error extracting camera parameters for camera {camera_id}: {str(e)}")
        
        return camera_data
    
    def save_summary_data(self, summary, all_camera_positions):
        """保存汇总数据"""
        # 保存完整汇总
        with open(os.path.join(self.output_dir, "calibration_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 保存所有相机位置
        if all_camera_positions:
            with open(os.path.join(self.output_dir, "all_camera_positions.json"), 'w') as f:
                json.dump(all_camera_positions, f, indent=2)
            
            positions_array = np.array([pos for pos in all_camera_positions.values()])
            np.save(os.path.join(self.output_dir, "all_camera_positions.npy"), positions_array)
            np.savetxt(os.path.join(self.output_dir, "all_camera_positions.txt"), 
                      positions_array, fmt='%.10f', 
                      header='Camera positions in world coordinates (x, y, z)')
            
            print(f"\n✓ Saved positions for {len(all_camera_positions)} cameras")
        
        print(f"✓ All calibration data saved to '{self.output_dir}'")
        print(f"✓ Summary saved to 'calibration_summary.json'")


def main():
    # 配置参数
    file_path = "/home/fzhi/fzt/dna/apose_main/0008_apose01.smc"
    output_dir = "/home/fzhi/fzt/output"
    
    # 创建处理器
    processor = CameraCalibrationProcessor(output_dir)
    
    try:
        # 创建SMC读取器
        print("🔄 Loading .smc file...")
        reader = SMCReader(file_path)
        
        # 打印可用键
        print(f"\n📋 Available keys: {reader.get_available_keys()}")
        
        # 检查Camera_Parameter
        if 'Camera_Parameter' not in reader.get_available_keys():
            print("❌ Camera_Parameter not found!")
            return
        
        # 获取相机信息
        camera_info = {}
        camera_5mp_info = reader.get_Camera_5mp_info()
        if camera_5mp_info:
            camera_info["Camera_5mp"] = camera_5mp_info
            print(f"\n📷 Camera 5MP: {camera_5mp_info['num_device']} devices, {camera_5mp_info['num_frame']} frames, {camera_5mp_info['resolution']}")
        
        camera_12mp_info = reader.get_Camera_12mp_info()
        if camera_12mp_info:
            camera_info["Camera_12mp"] = camera_12mp_info
            print(f"📷 Camera 12MP: {camera_12mp_info['num_device']} devices, {camera_12mp_info['num_frame']} frames, {camera_12mp_info['resolution']}")
        
        # 获取标定数据
        print("\n🔄 Extracting calibration data...")
        all_calibration = reader.get_Calibration_all()
        
        if not all_calibration:
            print("❌ No calibration data found!")
            return
        
        print(f"✅ Found calibration data for {len(all_calibration)} cameras")
        
        # 处理每个相机
        summary = {}
        all_camera_positions = {}
        
        print("\n🔄 Processing cameras...")
        for camera_id in sorted(all_calibration.keys(), key=int):
            print(f"\n📷 Processing Camera {camera_id}:")
            
            camera_data = processor.process_camera_calibration(
                camera_id, 
                all_calibration[camera_id], 
                camera_info
            )
            
            summary[camera_id] = camera_data
            
            # 收集相机位置
            if 'camera_position' in camera_data:
                all_camera_positions[camera_id] = camera_data['camera_position']
            
            # 打印基础信息
            if all(all_calibration[camera_id].get(mt) is not None for mt in ['K', 'D', 'RT']):
                print(f"  ✓ All essential matrices available for Camera {camera_id}")
        
        # 保存汇总数据
        print(f"\n💾 Saving summary data...")
        processor.save_summary_data(summary, all_camera_positions)
        
        # 保存相机信息
        if camera_info:
            with open(os.path.join(output_dir, "camera_info.json"), 'w') as f:
                json.dump(camera_info, f, indent=2)
            print(f"✓ Camera info saved to 'camera_info.json'")
        
        print(f"\n🎉 Processing complete! Check '{output_dir}' for all files.")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 释放资源
        if 'reader' in locals():
            reader.release()
            print("🔄 Resources released")


if __name__ == "__main__":
    main()
