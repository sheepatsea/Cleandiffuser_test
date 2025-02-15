import os
import json
import numpy as np
import zarr
from tqdm import tqdm  # 用于显示进度条

def combine_and_save_to_zarr(source_dir, output_zarr):
    # 获取所有子文件夹
    folders = sorted([f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))])
    
    # 创建 zarr 根组
    root = zarr.group(output_zarr)
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')
    
    # 初始化列表用于收集数据
    all_states = []
    all_actions = []
    all_images0 = []
    all_images1 = []
    episode_ends = []  # 用于记录每个episode的结束位置
    
    current_step = 0  # 用于追踪当前步数
    
    # 读取所有文件夹中的数据
    for folder in tqdm(folders, desc="Processing folders"):
        folder_path = os.path.join(source_dir, folder)
        
        # 读取 json 文件
        json_file = os.path.join(folder_path, 'obs_action.json')
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_states.extend(data['obs']['jq'])
            all_actions.extend(data['act'])
        
        # 读取图像数据
        img_file = os.path.join(folder_path, 'images0.npz')
        if os.path.exists(img_file):
            images = np.load(img_file)['images']
            all_images0.extend(images)

        img_file = os.path.join(folder_path, 'images1.npz')
        if os.path.exists(img_file):
            images = np.load(img_file)['images']
            all_images1.extend(images)            
            
        # 更新 episode_ends
        current_step += len(data['act'])  # 使用动作长度来计算步数
        episode_ends.append(current_step)
    
    # 转换为numpy数组
    states = np.array(all_states)
    actions = np.array(all_actions)
    images0 = np.array(all_images0)
    images1 = np.array(all_images1)
    episode_ends = np.array(episode_ends)
    
    # 创建压缩器
    compressor = zarr.Blosc(
        cname='zstd',     # 使用 zstd 压缩算法
        clevel=5,         # 压缩级别 5
        shuffle=2,        # 使用位随机重排
        blocksize=0       # 自动块大小
    )
    
    # 保存数据到 data 组
    data_group.create_dataset(
        'state',
        data=states,
        chunks=(161, states.shape[1]),  # 使用与原始文件相同的块大小
        compressor=compressor,
        dtype='<f4',  # 使用32位浮点数
        fill_value=0.0,
        order='C'
    )
    
    data_group.create_dataset(
        'action',
        data=actions,
        chunks=(161, actions.shape[1]),
        compressor=compressor,
        dtype='<f4',
        fill_value=0.0,
        order='C'
    )
    
    data_group.create_dataset(
        'image0',
        data=images0,
        chunks=(161, *images0.shape[1:]),
        compressor=compressor,
        dtype='<f4',
        fill_value=0.0,
        order='C'
    )
    data_group.create_dataset(
        'image1',
        data=images1,
        chunks=(161, *images1.shape[1:]),
        compressor=compressor,
        dtype='<f4',
        fill_value=0.0,
        order='C'
    )

    # 保存元数据到 meta 组
    meta_group.create_dataset(
        'episode_ends',
        data=episode_ends,
        chunks=(161,),
        compressor=compressor,
        dtype='<i8',
        fill_value=0.0,
        order='C'
    )

if __name__ == "__main__":
    # 使用示例
    source_directory = "block_place"  # 源数据目录
    output_zarr_path = "block_place_replay.zarr"  # 输出zarr文件路径
    
    combine_and_save_to_zarr(source_directory, output_zarr_path)