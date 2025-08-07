- dna_rendering_sample_code.zip 
    1. SMCReader.py 
        provide "class SMCReader" to read smc file. 
    2. smc_reader_full.py
        read smc file.
        we provide two functions to read one sample and read all data respectively.
    3. vis_smc_install.sh 
        install necessary environment to run smc_visualization.py. 
        !!! Please register an account at https://smpl-x.is.tue.mpg.de/ before running the installation script !!!
    4. smc_visualization.py 
        visualize smplx and keypoints in annotation smc file.  
    5. gen_pcd_from_kinect.py 
        generate pointcloud from captured images. 

- data_used_in_4K4D
    1. annotations/ 
        rgb camera information, matting masks, keypoint_2d, keypoint_3d, smplx 
    2. kinect/ 
        raw kinect images, kinect depth masks, kinect camera information
    3. main/ 
        raw RGB images
    4. preview/ 
        preview videos of motion sequences
    5. data_used_in_4K4D_rgb_cams.zip
        rgb camera information
        (same with rgb camera data in annotations/, put it separately for users who want to download rgb camera data only) 
    6. apose/
        - apose_kinect/
            apose data: raw kinect images, kinect depth masks, kinect camera information
        - apose_main/
            apose data: raw rgb images, rgb camera information
        - 4K4D_action_apose_mapping.json
           many-to-one correspondence between motion sequences and apose sequences      
    7. data_used_in_4K4D_file_gid.json
        google's file id of shared data in "data_used_in_4K4D" folder, you can use it to batch download files.

- Part 1/
    1. dna_rendering_part1_annotations/
        rgb camera information, matting masks, keypoint_2d, keypoint_3d, smplx 
    2. dna_rendering_part1_main/
        raw rgb images
    3. dna_rendering_part1_kinect.zip
        raw kinect images, kinect depth masks, kinect camera information
    4. dna_rendering_part1_label.zip
        text labels
    5. dna_rendering_part1_preview.zip
        preview videos of motion sequences
    6. dna_rendering_part1_split_label.json 
        difficulty level of each motion sequence
    7. dna_rendering_part1_apose/
        - apose_kinect/
            apose data: raw kinect images, kinect depth masks, kinect camera information
        - apose_main/
            apose data: raw rgb images, rgb camera information
        - part1_action_apose_mapping.json
           many-to-one correspondence between motion sequences and apose sequences            
    8. Part_1_file_gid.json
        google's file id of shared data in "Part 1" folder, you can use it to batch download files.

- Part 2/
    1. dna_rendering_part2_annotations/
        rgb camera information, matting, keypoint_2d, keypoint_3d, smplx 
    2. dna_rendering_part2_kinect/
        raw kinect images, kinect depth mask, kinect camera information
    3. dna_rendering_part2_main/ 
        raw rgb images
    4. dna_rendering_part2_label.zip
        text labels
    5. dna_rendering_part2_preview.zip
        preview videos of motion sequences
    6. dna_rendering_part2_rgb_cams.zip 
        rgb camera information
        (same with rgb camera data in dna_rendering_part2_annotations/, put it separately for users who want to download rgb camera data only) 
    7. dna_rendering_part2_split_label.json 
        difficulty level of each motion sequence
    8. dna_rendering_part2_apose/
        - apose_kinect/
            apose data: raw kinect images, kinect depth masks, kinect camera information
        - apose_main/
            apose data: raw rgb images, rgb camera information
        - part2_action_apose_mapping.json
           many-to-one correspondence between motion sequences and apose sequences      
    9. Part_2_file_gid.json
        google's file id of shared data in "Part 2" folder, you can use it to batch download files.

- Part 3/
    1. dna_rendering_part3_kinect: 
        raw kinect images, kinect depth masks, kinect camera information
    2. dna_rendering_part3_main: 
        raw RGB images, RGB camera information
    3. dna_rendering_part3_preview:
        preview videos for each released id in part 3, each id has three motions.
    4. dna_rendering_part3_apose/
        - apose_kinect/
            apose data: raw kinect images, kinect depth masks, kinect camera information
        - apose_main/
            apose data: raw rgb images, rgb camera information
        - part3_action_apose_mapping.json
           many-to-one correspondence between motion sequences and apose sequences      
    5. Part_3_file_gid.json
        google's file id of shared data in "Part 3" folder, you can use it to batch download files.

- Part 4/
    1. dna_rendering_part4_kinect: 
        raw kinect images, kinect depth masks, kinect camera information
    2. dna_rendering_part4_main: 
        raw RGB images, RGB camera information
    3. dna_rendering_part4_preview:
        preview videos for each released id in part 4, each id has three motions.
    4. dna_rendering_part4_apose/
        - apose_kinect/
            apose data: raw kinect images, kinect depth masks, kinect camera information
        - apose_main/
            apose data: raw rgb images, rgb camera information
        - part4_action_apose_mapping.json
           many-to-one correspondence between motion sequences and apose sequences      
    5. Part_4_file_gid.json
        google's file id of shared data in "Part 4" folder, you can use it to batch download files.

- Part 5/
    1. dna_rendering_part5_kinect: 
        raw kinect images, kinect depth masks, kinect camera information
    2. dna_rendering_part5_main: 
        raw RGB images, RGB camera information
    3. dna_rendering_part5_preview:
        preview videos for each released id in part 5, each id has three motions.
    4. dna_rendering_part5_apose/
        - apose_kinect/
            apose data: raw kinect images, kinect depth masks, kinect camera information
        - apose_main/
            apose data: raw rgb images, rgb camera information
        - part5_action_apose_mapping.json
           many-to-one correspondence between motion sequences and apose sequences      
    5. Part_5_file_gid.json
        google's file id of shared data in "Part 5" folder, you can use it to batch download files.

- Part 6/
    1. dna_rendering_part6_kinect: 
        raw kinect images, kinect depth masks, kinect camera information
    2. dna_rendering_part6_main: 
        raw RGB images, RGB camera information
    3. dna_rendering_part6_preview:
        preview videos for each released id in part 6, each id has three motions.
    4. dna_rendering_part6_apose/
        - apose_kinect/
            apose data: raw kinect images, kinect depth masks, kinect camera information
        - apose_main/
            apose data: raw rgb images, rgb camera information
        - part6_action_apose_mapping.json
           many-to-one correspondence between motion sequences and apose sequences      
    5. Part_6_file_gid.json
        google's file id of shared data in "Part 6" folder, you can use it to batch download files.

