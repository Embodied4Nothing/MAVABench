import numpy as np
import os
import open3d as o3d
import mediapy
import argparse
import traceback
from dm_control import viewer
from tqdm import tqdm
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from VLABench.utils.data_utils import process_observations
from typing import List, Dict
import os
import h5py
import numpy as np
import json
import openpi.shared.normalize as normalize
class TrajLogger:

    def __init__(self, camera_cnt: int, save_dir:str, filename:str):
    # camera_cnt：摄像头个数
    # save_dir：文件保存路径（必填）
    # filename：文件名（必填）
        self.camera_cnt = camera_cnt
        self.skills = []
        self.obs = []
        self.waypoints = []
        self.save_dir = save_dir
        self.filename = filename
        # self.stats = normalize.RunningStats()
    ## 记录 t 时刻。
    ## observations：与 vlabench 相同，observations["rgb"] 是一个列表，依次记录各个摄像头的视角
    ## actions: ndarray，按顺序记录各个机械臂的行为
    def log_step(self, t, observations, actions): 
        self.obs.extend(observations)
        self.waypoints.extend(actions)

        values = np.array(actions)
        self.stats.update(values.reshape(-1, values.shape[-1])) ## 增量计算 actions 的 norm_stats

    def mark_skill(self, skill_name:str, t_start, t_end, success, params):
        # params: 单个 skill 的信息（json 文本）
        self.skills.append((skill_name,t_start,t_end,success,params))
        
    def finish(self, meta):
        # meta 针对整条 episode 的元信息（json 文本）

        if not os.path.exists(os.path.join(self.save_dir)):
            os.makedirs(os.path.join(self.save_dir))
        
        # 将各个机械臂摄像头的视角分别绘制成视频
        frames = [] 
        if self.camera_cnt % 2 == 0 :
            for o in self.obs:
                frames.append(np.vstack([
                    np.hstack(o["rgb"][:self.camera_cnt/2]), 
                    np.hstack(o["rgb"][self.camera_cnt/2:self.camera_cnt])]))
        else: # 摄像头为奇数个时，在第一行添加一个黑屏
            for o in self.obs:
                down = [np.zeros_like(o["rgb"][0])]
                down.extend(o["rgb"][:self.camera_cnt/2])
                frames.append(np.vstack([
                    np.hstack(down), 
                    np.hstack(o["rgb"][self.camera_cnt/2:self.camera_cnt])]))

        mediapy.write_video(os.path.join(
            self.save_dir, f"demo_{self.filename}.mp4"), frames, fps=10)
        
        # 保存 hdf5，以 episode_id 为文件名
        hdf5_file = h5py.File(os.path.join(self.save_dir, f"data_{self.filename}.hdf5"), "a")
        group = hdf5_file.get("data")
        if group is None:
            group = hdf5_file.create_group("data")

        group.create_dataset(
            "trajectory", data=np.array(self.waypoints, dtype=np.float32), compression='gzip', compression_opts=9) # 保存 trajectory

        skills_group = group.create_group("skill")
        for (skill_name, t_start, t_end, params, success) in self.skills: # 保存 skill（可能重名，故按起始时间排列）
            skill_group = skills_group.create_group(f"{t_start}")
            skill_group.create_dataset("skill_name", data=np.array(skill_name.encode('utf-8')).astype("S"))
            task_info = json.dumps(params)
            skill_group.create_dataset("params", data=np.array(task_info.encode('utf-8')).astype("S"))
            skill_group.create_dataset("time_interval",data=np.array([t_start,t_end]))
            skill_group.create_dataset("success",data=np.array([success]))
        
        obs_group = group.create_group("observation")
        observations = process_observations(self.obs)
        for key, buffer in observations.items() :
            try: # 保存 observations
                buffer = np.array(buffer, dtype=np.float32) if key != "rgb" else np.array(buffer, dtype=np.uint8)
                obs_group.create_dataset(key, data=buffer, compression='gzip', compression_opts=9)
            except Exception as e:
                print(f"Error in saving {key}: {e}")


        meta_info = json.dumps(meta)
        group.create_dataset("meta_info", data=np.array(meta_info.encode('utf-8')).astype("S")) #保存 meta（json 格式，字符串）
        
        norm_stats = self.stats.get_statistics()  ## 保存 actions 的 norm_stats
        group.create_dataset("mean",data=np.asarray(norm_stats.mean))
        group.create_dataset("std",data=np.asarray(norm_stats.q99))
        group.create_dataset("q01",data=np.asarray(norm_stats.q01))
        group.create_dataset("q99",data=np.asarray(norm_stats.q99))

        hdf5_file.close()

        



        
