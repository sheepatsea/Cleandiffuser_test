import hydra
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import gym
import pathlib
import time
import numpy as np
import torch
import torch.nn as nn
from utils import set_seed, Logger
from torch.optim.lr_scheduler import CosineAnnealingLR

from cleandiffuser.env import pusht
from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.dataset.pusht_dataset import PushTImageDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters
    
from discoverse.airbot_play import AirbotPlayFIK
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSERT_DIR
from discoverse.envs.airbot_play_base import AirbotPlayCfg
from discoverse.utils import get_body_tmat, get_site_tmat, step_func, SimpleStateMachine
from discoverse.task_base import AirbotPlayTaskBase, recoder_airbot_play

class SimNode(AirbotPlayTaskBase):
    def __init__(self, config: AirbotPlayCfg, args):
        super().__init__(config)
        self.camera_0_pose = (self.mj_model.camera("eye_side").pos.copy(), self.mj_model.camera("eye_side").quat.copy())
        self.obs_steps = args.obs_steps
        self.obs_que = None
        self.obs_que_used = None
        self.video_list = list()

    def domain_randomization(self):
        # 随机 方块位置
        self.mj_data.qpos[self.nj+1+0] += 2.*(np.random.random() - 0.5) * 0.12
        self.mj_data.qpos[self.nj+1+1] += 2.*(np.random.random() - 0.5) * 0.08

        # 随机 杯子位置
        self.mj_data.qpos[self.nj+1+7+0] += 2.*(np.random.random() - 0.5) * 0.1
        self.mj_data.qpos[self.nj+1+7+1] += 2.*(np.random.random() - 0.5) * 0.05

    def check_success(self):
        tmat_block = get_body_tmat(self.mj_data, "block_green")
        tmat_bowl = get_body_tmat(self.mj_data, "bowl_pink")
        return (abs(tmat_bowl[2, 2]) > 0.99) and np.hypot(tmat_block[0, 3] - tmat_bowl[0, 3], tmat_block[1, 3] - tmat_bowl[1, 3]) < 0.02
    
    def reset(self):
        obs, t = super().reset(), 0
        from collections import  deque
        self.obs_que = deque([obs], maxlen=self.obs_steps+1) 
        self.obs_que_used = self._get_obs()
        return self.obs_que_used, t
    
    def stack_last_n_obs(all_obs, n_steps):
        assert(len(all_obs) > 0)
        result = np.zeros((n_steps,) + all_obs[-1].shape, 
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
        return result
    
    def _get_obs(self):
        result = dict()
        result['agent_pos'] = self.stack_last_n_obs(
                [np.array(obs['jq']) for obs in self.obs_que],
                self.obs_steps
            )
        imgs0 = list()
        imgs1 = list()
        for obs in self.obs_que:
            img = obs['img']
            img0 = np.transpose(img[0]/255, (2, 0, 1))
            img1 = np.transpose(img[1]/255, (2, 0, 1))
            imgs0.append(img0)
            imgs1.append(img1)
        result['image0'] = self.stack_last_n_obs(
                imgs0,
                self.obs_steps
            )
        result['image1'] = self.stack_last_n_obs(
                imgs1,
                self.obs_steps
            )
        return result
    
    def action(self, action):
        success = 0
        for act in action: #依次执行每个动作
            obs, _, _, _, _ = super().step(act)
            self.obs_que.append(obs) #添加单个obs
            self.video_list.append(obs['img'])
            if self.check_success():
                success = 1
                break
        return self._get_obs(), success




cfg = AirbotPlayCfg()
cfg.use_gaussian_renderer = True
cfg.init_key = "ready"
cfg.gs_model_dict["background"]  = "scene/lab3/point_cloud_1.ply"
cfg.gs_model_dict["drawer_1"]    = "hinge/drawer_1.ply"
cfg.gs_model_dict["drawer_2"]    = "hinge/drawer_2.ply"
cfg.gs_model_dict["bowl_pink"]   = "object/bowl_pink.ply"
cfg.gs_model_dict["block_green"] = "object/block_green.ply"

cfg.mjcf_file_path = "mjcf/tasks_airbot_play/block_place.xml"
cfg.obj_list     = ["drawer_1", "drawer_2", "bowl_pink", "block_green"]
cfg.timestep     = 1/240
cfg.decimation   = 4
cfg.sync         = False
cfg.headless     = True
cfg.render_set   = {
    "fps"    : 20,
    "width"  : 320,
    "height" : 240
}
cfg.obs_rgb_cam_id = [0, 1]
cfg.obs_depth_cam_id = [0, 1]
cfg.save_mjb_and_task_config = True
    

def make_env(args, idx):
    def thunk():
        env = gym.make(args.env_name)  
        video_recorder = VideoRecorder.create_h264(
                            fps=10,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=22,
                            thread_type='FRAME',
                            thread_count=1
                        )
        env = VideoRecordingWrapper(env, video_recorder, file_path=None, steps_per_render=1)
        env = MultiStepWrapper(env, n_obs_steps=args.obs_steps, n_action_steps=args.action_steps, max_episode_steps=args.max_episode_steps)
        # env.seed(args.seed+idx)
        print("Env seed: ", args.seed+idx)
        return env

    return thunk


def inference(args, sim_node:SimNode, dataset, agent, logger, gradient_step):
    """Evaluate a trained agent and optionally save a video."""
    # ---------------- Start Rollout ----------------
    episode_rewards = []
    episode_steps = []
    episode_success = []
    
    if args.diffusion == "ddpm":
        solver = None
    elif args.diffusion == "ddim":
        solver = "ddim"
    elif args.diffusion == "dpm":
        solver = "ode_dpmpp_2"
    elif args.diffusion == "edm":
        solver = "euler"




    for i in range(args.eval_episodes // args.num_envs): 
        # ep_reward = [0.0] * args.num_envs
        # step_reward = []
        obs, t = sim_node.reset() # {obs_name: (obs_steps, obs_dim)}
        success = 0

        while t < args.max_episode_steps:
            # 接收n个obs
            obs_dict = {}
            for k in obs.keys():
                obs_seq = obs[k].astype(np.float32)  # (obs_steps, obs_dim)
                nobs = dataset.normalizer['obs'][k].normalize(obs_seq)
                nobs = torch.tensor(nobs, device=args.device, dtype=torch.float32)  # (obs_steps, obs_dim)
                nobs = nobs[None, :].expand(args.num_envs, *nobs.shape) # torch.Size([num_envs, obs_steps, obs_dim])
                obs_dict[k] = nobs
            # predict
            with torch.no_grad():
                condition = obs_dict
                prior = torch.zeros((args.num_envs, args.horizon, args.action_dim), device=args.device)
                naction, _ = agent.sample(prior=prior, n_samples=args.num_envs, sample_steps=args.sample_steps,
                                        solver=solver, condition_cfg=condition, w_cfg=1.0, use_ema=True)

            # unnormalize prediction
            naction = naction.detach().to('cpu').numpy()  # (horizon, action_dim)
            action_pred = dataset.normalizer['action'].unnormalize(naction)  
            
            # get action
            start = args.obs_steps - 1
            end = start + args.action_steps
            action = np.squeeze(action_pred[:, start:end, :]) # 多一个env_num维度

            obs, success = sim_node.step(action)
            t += args.action_steps

            if success:
                break

        if not success: # 输出实际距离的负数 
            tmat_block = get_body_tmat(sim_node.mj_data, "block_green")
            tmat_bowl = get_body_tmat(sim_node.mj_data, "bowl_pink")
            success = - np.hypot(tmat_block[0, 3] - tmat_bowl[0, 3], tmat_block[1, 3] - tmat_bowl[1, 3])

        import mediapy
        for id in cfg.obs_rgb_cam_id:
            mediapy.write_video(os.path.join(args.work_dir, f"videos/{gradient_step}_{i}_cam_{id}.mp4"), [videos[id] for videos in sim_node.video_lst], fps=cfg.render_set["fps"])
        # ep_reward = np.around(np.array(ep_reward), 2)
        print(f"[Episode {1+i*(args.num_envs)}-{(i+1)*(args.num_envs)}] success:{success}")
        # episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success) if success==1 else episode_success.append(0)
    print(f"Mean step: {np.nanmean(episode_steps)} Mean reward: {np.nanmean(episode_rewards)} Mean success: {np.nanmean(episode_success)}")
    return {'mean_step': np.nanmean(episode_steps), 'mean_reward': np.nanmean(episode_rewards), 'mean_success': np.nanmean(episode_success)}


@hydra.main(config_path="../configs/dp/block_place/dit", config_name="block_place_image")
def pipeline(args):
    # ---------------- Create Logger ----------------
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)

    # ---------------- Create Environment ----------------
    sim_node = SimNode(cfg)
    sim_node.reset()
    # ---------------- Create Dataset ----------------
    dataset_path = os.path.expanduser(args.dataset_path)
    dataset = PushTImageDataset(dataset_path, horizon=args.horizon, obs_keys=args.obs_keys, 
                                pad_before=args.obs_steps-1, pad_after=args.action_steps-1, abs_action=args.abs_action)
    print(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    
    # --------------- Create Diffusion Model -----------------
    if args.nn == "dit":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import DiT1d
        
        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop, 
            use_group_norm=args.use_group_norm, use_seq=args.use_seq).to(args.device)
        nn_diffusion = DiT1d(
            args.action_dim, emb_dim=256*args.obs_steps, d_model=320, n_heads=10, depth=2, timestep_emb_type="fourier").to(args.device)

    elif args.nn == "chi_unet":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import ChiUNet1d

        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop, 
            use_group_norm=args.use_group_norm, use_seq=args.use_seq).to(args.device)
        nn_diffusion = ChiUNet1d(
            args.action_dim, 256, args.obs_steps, model_dim=256, emb_dim=256, dim_mult=[1, 2, 2],
            obs_as_global_cond=True, timestep_emb_type="positional").to(args.device)
        
    elif args.nn == "chi_transformer":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import ChiTransformer
        
        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop, 
            use_group_norm=args.use_group_norm, use_seq=args.use_seq, keep_horizon_dims=True).to(args.device)
        nn_diffusion = ChiTransformer(
            args.action_dim, 256, args.horizon, args.obs_steps, d_model=256, nhead=4, num_layers=4,
            timestep_emb_type="positional").to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}")
    
    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")
    
    if args.diffusion == "ddpm":
        from cleandiffuser.diffusion.ddpm import DDPM
        x_max = torch.ones((1, args.horizon, args.action_dim), device=args.device) * +1.0
        x_min = torch.ones((1, args.horizon, args.action_dim), device=args.device) * -1.0
        agent = DDPM(
            nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=args.device,
            diffusion_steps=args.sample_steps, x_max=x_max, x_min=x_min,
            optim_params={"lr": args.lr})
    elif args.diffusion == "edm":
        from cleandiffuser.diffusion.edm import EDM
        agent = EDM(nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=args.device,
                    optim_params={"lr": args.lr})
    else:
        raise NotImplementedError
    lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps)
    
    if args.mode == "train":
        # ----------------- Training ----------------------
        n_gradient_step = 0
        diffusion_loss_list = []
        start_time = time.time()
        for batch in loop_dataloader(dataloader):
            # get condition
            nobs = batch['obs']
            condition = {}
            for k in nobs.keys():
                condition[k] = nobs[k][:, :args.obs_steps, :].to(args.device)

            naction = batch['action'].to(args.device)

            # update diffusion
            diffusion_loss = agent.update(naction, condition)['loss']
            lr_scheduler.step()
            diffusion_loss_list.append(diffusion_loss)

            if n_gradient_step % args.log_freq == 0:
                metrics = {
                    'step': n_gradient_step,
                    'total_time': time.time() - start_time,
                    'avg_diffusion_loss': np.mean(diffusion_loss_list)
                }
                logger.log(metrics, category='train')
                diffusion_loss_list = []
            
            if n_gradient_step % args.save_freq == 0:
                logger.save_agent(agent=agent, identifier=n_gradient_step)
                
            if n_gradient_step % args.eval_freq == 0:
                print("Evaluate model...")
                agent.model.eval()
                agent.model_ema.eval()
                metrics = {'step': n_gradient_step}
                metrics.update(inference(args, sim_node, dataset, agent, logger, n_gradient_step))
                logger.log(metrics, category='inference')
                agent.model.train()
                agent.model_ema.train()
            
            n_gradient_step += 1
            if n_gradient_step > args.gradient_steps:
                # finish
                logger.finish(agent)
                break
    elif args.mode == "inference":
        # ----------------- Inference ----------------------
        if args.model_path:
            agent.load(args.model_path)
        else:
            raise ValueError("Empty model for inference")
        agent.model.eval()
        agent.model_ema.eval()

        metrics = {'step': 0}
        metrics.update(inference(args, sim_node, dataset, agent, logger, 0))
        logger.log(metrics, category='inference')
        
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    pipeline()









    

