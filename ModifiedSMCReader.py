# Save this as ModifiedSMCReader.py
import h5py
import cv2
import numpy as np 
from tqdm import tqdm 

class SMCReader:

    def __init__(self, file_path):
        """Read SenseMocapFile endswith ".smc".

        Args:
            file_path (str):
                Path to an SMC file.
        """
        self.smc = h5py.File(file_path, 'r')
        self.__calibration_dict__ = None
        self.__kinect_calib_dict__ = None 
        self.__available_keys__ = list(self.smc.keys())
        
        self.actor_info = None 
        if hasattr(self.smc, 'attrs') and len(self.smc.attrs.keys()) > 0:
            # Create actor_info with safe attribute access
            self.actor_info = {}
            for attr in ['actor_id', 'performance_id', 'age', 'gender', 'height', 'weight']:
                if attr in self.smc.attrs:
                    self.actor_info[attr] = self.smc.attrs[attr]
                else:
                    self.actor_info[attr] = None
                    print(f"Warning: '{attr}' attribute not found in the .smc file")

        self.Camera_5mp_info = None 
        if 'Camera_5mp' in self.smc:
            self.Camera_5mp_info = dict(
                num_device=self.smc['Camera_5mp'].attrs['num_device'],
                num_frame=self.smc['Camera_5mp'].attrs['num_frame'],
                resolution=self.smc['Camera_5mp'].attrs['resolution'],
            )
        self.Camera_12mp_info = None 
        if 'Camera_12mp' in self.smc:
            self.Camera_12mp_info = dict(
                num_device=self.smc['Camera_12mp'].attrs['num_device'],
                num_frame=self.smc['Camera_12mp'].attrs['num_frame'],
                resolution=self.smc['Camera_12mp'].attrs['resolution'],
            )
        self.Kinect_info = None
        if 'Kinect' in self.smc:
            self.Kinect_info=dict(
                num_device=self.smc['Kinect'].attrs['num_device'],
                num_frame=self.smc['Kinect'].attrs['num_frame'],
                resolution=self.smc['Kinect'].attrs['resolution'],
            )
    ### Helper to decode RGB
    def __read_color_from_bytes__(self, color_array):
        return cv2.imdecode(color_array, cv2.IMREAD_COLOR)

    ### get_img() method you provided earlier
    def get_img(self, Camera_group, Camera_id, Image_type, Frame_id=None, disable_tqdm=False):
        if not Camera_group in self.smc:
            print("=== no key: %s.\nplease check available keys!" % Camera_group)
            return None

        assert(Camera_group in ['Camera_12mp', 'Camera_5mp','Kinect'])
        Camera_id = str(Camera_id)
        assert(Camera_id in self.smc[Camera_group].keys())
        assert(Image_type in self.smc[Camera_group][Camera_id].keys())
        assert(isinstance(Frame_id,(list,int, str, type(None))))

        if isinstance(Frame_id, (str,int)):
            Frame_id = str(Frame_id)
            assert(Frame_id in self.smc[Camera_group][Camera_id][Image_type].keys())
            if Image_type in ['color']:
                img_byte = self.smc[Camera_group][Camera_id][Image_type][Frame_id][()]
                img_color = self.__read_color_from_bytes__(img_byte)
            if Image_type == 'mask':
                img_byte = self.smc[Camera_group][Camera_id][Image_type][Frame_id][()]
                img_color = self.__read_color_from_bytes__(img_byte)
                img_color = np.max(img_color,2)
            if Image_type == 'depth':
                img_color = self.smc[Camera_group][Camera_id][Image_type][Frame_id][()]
            return img_color           
        else:
            if Frame_id is None:
                Frame_id_list = sorted([int(l) for l in self.smc[Camera_group][Camera_id][Image_type].keys()])
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in tqdm(Frame_id_list, disable=disable_tqdm):
                rs.append(self.get_img(Camera_group, Camera_id, Image_type,fi))
            return np.stack(rs, axis=0)

    def get_available_keys(self):
        return self.__available_keys__ 

    def get_actor_info(self):
        return self.actor_info
    
    def get_Camera_12mp_info(self):
        return self.Camera_12mp_info

    def get_Camera_5mp_info(self):
        return self.Camera_5mp_info
    
    def get_Kinect_info(self):
        return self.Kinect_info
    
    ### RGB Camera Calibration
    def get_Calibration_all(self):
        """Get calibration matrix of all cameras and save it in self
        
        Args:
            None

        Returns:
            Dictionary of calibration matrixs of all matrixs.
              dict( 
                Camera_Parameter: Camera_id : Matrix_type : value
              )
            Notice:
                Camera_id(str) in {'Camera_5mp': '0'~'47',  'Camera_12mp':'48'~'60'}
                Matrix_type in ['D', 'K', 'RT', 'Color_Calibration'] 
        """  
        if not 'Camera_Parameter' in self.smc:
            print("=== no key: Camera_Parameter.\nplease check available keys!")
            return None  

        if self.__calibration_dict__ is not None:
            return self.__calibration_dict__

        self.__calibration_dict__ = dict()
        for ci in self.smc['Camera_Parameter'].keys():
            self.__calibration_dict__.setdefault(ci,dict())
            for mt in ['D', 'K', 'RT', 'Color_Calibration'] :
                if mt in self.smc['Camera_Parameter'][ci]:
                    self.__calibration_dict__[ci][mt] = \
                        self.smc['Camera_Parameter'][ci][mt][()]
                else:
                    print(f"Warning: Matrix type '{mt}' not found for camera {ci}")
                    self.__calibration_dict__[ci][mt] = None
        return self.__calibration_dict__

    def get_Calibration(self, Camera_id):
        """Get calibration matrixs of a certain camera by its type and id 

        Args:
            Camera_id (int/str of a number):
                Camera_id(str) in {'Camera_5mp': '0'~'47',  
                    'Camera_12mp':'48'~'60'}
        Returns:
            Dictionary of calibration matrixs.
                ['D', 'K', 'RT', 'Color_Calibration'] 
        """
        if not 'Camera_Parameter' in self.smc:
            print("=== no key: Camera_Parameter.\nplease check available keys!")
            return None  

        camera_id_str = f'{int(Camera_id):02d}'
        if camera_id_str not in self.smc['Camera_Parameter']:
            print(f"=== no camera with ID: {camera_id_str}.\nplease check available camera IDs!")
            return None

        rs = dict()
        for k in ['D', 'K', 'RT', 'Color_Calibration'] :
            if k in self.smc['Camera_Parameter'][camera_id_str]:
                rs[k] = self.smc['Camera_Parameter'][camera_id_str][k][()]
            else:
                print(f"Warning: Matrix type '{k}' not found for camera {camera_id_str}")
                rs[k] = None
        return rs

    def release(self):
        self.smc.close()
        self.smc = None 
        self.__calibration_dict__ = None
        self.__kinect_calib_dict__ = None
        self.__available_keys__ = None
        self.actor_info = None 
        self.Camera_5mp_info = None
        self.Camera_12mp_info = None 
        self.Kinect_info = None
