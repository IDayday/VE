import os
import torch
import numpy as np
import random
import argparse

import gym
from multiworld.envs.mujoco import register_custom_envs as register_mujoco_envs
from utils.density import *
from utils.tools import evalPolicy, sample_and_preprocess_batch
import time
from VE_ant import VE
from HER import HERReplayBuffer, PathBuilder
from torch.utils.tensorboard import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES'] = "0"



def main(env, args, policy, replay_buffer, path_builder, writer):

    # Initialize environment
    obs = env.reset()
    done = False
    state = obs["observation"]
    goal = obs["desired_goal"]
    episode_timesteps = 0
    episode_num = 0
    train_successes = [] 
    success_in_train = 0
    DE_fit = False

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(state, goal)

        # Perform action
        next_obs, reward, _, status = env.step(action) 
        
        next_state = next_obs["observation"]
        done = status["xy-distance"][0] < args.distance_threshold

        path_builder.add_all(
            observations=obs,
            actions=action,
            rewards=reward,
            next_observations=next_obs,
            terminals=[1.0*done]
        )

        state = next_state
        obs = next_obs


        # Train agent after collecting enough data
        if t >= args.batch_size and t >= args.start_timesteps and not DE_fit:
            # init train FLOW
            for i in range(100):
                DE_fit = DE._optimize()
            if DE_fit:
                print("FLOW Initialization")
        if t >= args.batch_size and t >= args.start_timesteps and DE_fit:
            # train policy
            her_state_batch, her_action_batch, her_reward_batch, her_next_state_batch, her_done_batch, her_goal_batch, her_subgoal_batch = sample_and_preprocess_batch(
                replay_buffer,
                t=t,
                tag="her",
                batch_size=args.batch_size, 
                state_dim=state_dim,
                distance_threshold=args.distance_threshold,
                de = DE,
                device=args.device
            )
            bu_state_batch, bu_action_batch, bu_reward_batch, bu_next_state_batch, bu_done_batch, bu_goal_batch, bu_subgoal_batch = sample_and_preprocess_batch(
                replay_buffer,
                t=t,
                tag="buffer",
                batch_size=args.batch_size, 
                state_dim=state_dim,
                distance_threshold=args.distance_threshold,
                de = DE,
                device=args.device
            )

            sampled_subgoal_batch = torch.FloatTensor(replay_buffer.random_state_batch(args.batch_size, state_dim, 100)).to(args.device)
            bu_subgoal_batch = bu_subgoal_batch.reshape(args.batch_size,-1,state_dim)
            bu_subgoal_batch = torch.cat([bu_subgoal_batch, sampled_subgoal_batch], dim=1)

            state_batch = torch.cat([her_state_batch, bu_state_batch], dim=0)
            action_batch = torch.cat([her_action_batch, bu_action_batch], dim=0)
            reward_batch = torch.cat([her_reward_batch, bu_reward_batch], dim=0)
            next_state_batch = torch.cat([her_next_state_batch, bu_next_state_batch], dim=0)
            done_batch = torch.cat([her_done_batch, bu_done_batch], dim=0)
            goal_batch = torch.cat([her_goal_batch, bu_goal_batch], dim=0)

            task = torch.cat([bu_state_batch[:,:2], bu_goal_batch[:,:2]], dim=-1).cpu().numpy()
            lhb.push(task)

            policy.train(state_batch, action_batch, reward_batch, next_state_batch, done_batch, goal_batch, her_subgoal_batch, bu_subgoal_batch, t)


        if done or episode_timesteps >= args.max_episode_length: 
            if DE_fit:
            # train FLOW model
                for i in range(100):
                    if i%5 == 0:
                        DE._optimize()
                    samples = lhb.sample(2).reshape(-1,4)
                    DE._optimize(samples)
            train_successes.append(1.0*done)
            if len(train_successes) > 20:
                success_in_train = np.mean(train_successes[-20:])
            writer.add_scalar("success in train",success_in_train,t) 

            # Add path to replay buffer and reset path builder
            replay_buffer.add_path(path_builder.get_all_stacked())
            replay_buffer_2.add_path(path_builder.get_all_stacked())
            path_builder = PathBuilder()
            writer.add_scalar("xy-distance",status["xy-distance"][0],t) 

            obs = env.reset()
            done = False
            state = obs["observation"]
            goal = obs["desired_goal"]
            episode_timesteps = 0
            episode_num += 1 


        if (t + 1) % args.eval_freq == 0 and t >= args.start_timesteps:
            # Eval policy (the most difficult task)
            r_eval_distance, r_success_rate = evalPolicy(
                policy, test_env_1, 
                N=10*args.n_eval,
                Tmax=args.max_episode_length, 
                distance_threshold=args.distance_threshold,
                writer = writer,
                test_time = t
            )
            writer.add_scalar("Random tasks success in eval",r_success_rate,t) 
            d_eval_distance, d_success_rate = evalPolicy(
                policy, test_env_2, 
                N=args.n_eval,
                Tmax=args.max_episode_length, 
                distance_threshold=args.distance_threshold,
                writer = writer,
                test_time = t
            )
            writer.add_scalar("Difficult tasks success in eval",d_success_rate,t) 
            # Save policy
            # policy.save(folder)
            print(f"VE | EnvSteps: {round(t/1000,1)}k | Success Rate: {success_in_train}")
        
        # if (t + 1) % 1e4 == 0 and t >= args.start_timesteps:
        #     DE.save(folder, t+1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name",           default="AntU")
    parser.add_argument("--distance_threshold", default=0.5, type=float)
    parser.add_argument("--start_timesteps",    default=1e4, type=int)
    parser.add_argument("--eval_freq",          default=1e3, type=int)
    parser.add_argument("--max_timesteps",      default=5e6, type=int)
    parser.add_argument("--max_episode_length", default=600, type=int)
    parser.add_argument("--batch_size",         default=2048, type=int)
    parser.add_argument("--replay_buffer_size", default=1e6, type=int)
    parser.add_argument("--n_eval",             default=10, type=int)
    parser.add_argument("--device",             default="cuda")
    parser.add_argument("--seed",               default=56, type=int)
    parser.add_argument("--exp_name",           default="VE_antmaze")
    parser.add_argument("--alpha",              default=0.1, type=float)
    parser.add_argument("--Lambda",             default=0.1, type=float)
    parser.add_argument("--sg_lr",              default=1e-4, type=float)
    parser.add_argument("--q_lr",               default=1e-3, type=float)
    parser.add_argument("--pi_lr",              default=1e-3, type=float)
    args = parser.parse_args()
    print(args)

    # select environment
    if args.env_name == "AntU":
        train_env_name = "AntULongTrainEnv-v0"
        test_env_name = "AntULongTestEnv-v0"
    elif args.env_name == "AntFb":
        train_env_name = "AntFbMedTrainEnv-v1"
        test_env_name = "AntFbMedTestEnv-v1"
    elif args.env_name == "AntMaze":
        train_env_name = "AntMazeMedTrainEnv-v1"
        test_env_name = "AntMazeMedTestEnv-v1"
    elif args.env_name == "AntFg":
        train_env_name = "AntFgMedTrainEnv-v1"
        test_env_name = "AntFgMedTestEnv-v1"
    print("Environments: ", train_env_name, test_env_name)

    # Set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize environment
    register_mujoco_envs()
    vectorized = True
    env         = gym.make(train_env_name)
    test_env_1  = gym.make(train_env_name)
    test_env_2  = gym.make(test_env_name)
    action_dim  = env.action_space.shape[0]
    state_dim = 31
    goal_dim = state_dim
    ex_time = time.strftime("%m-%d_%H-%M-%S", time.localtime())
    folder = "results/{}/VE/{}/{}/".format(train_env_name, args.exp_name, ex_time)
    if not os.path.isdir(folder):
        os.makedirs(folder)


    # Create logger
    writer = SummaryWriter(folder)
 
    # Initialize replay buffer and path_builder
    replay_buffer = HERReplayBuffer(
        max_size=args.replay_buffer_size,
        env=env,
        fraction_goals_are_rollout_goals = 0.0,
        fraction_resampled_goals_are_env_goals = 0.0,
        fraction_resampled_goals_are_replay_buffer_goals = 0.5,
        ob_keys_to_save     =["state_achieved_goal", "state_desired_goal"],
        desired_goal_keys   =["desired_goal", "state_desired_goal"],
        observation_key     = 'observation',
        desired_goal_key    = 'desired_goal',
        achieved_goal_key   = 'achieved_goal',
        vectorized          = vectorized 
    )
    replay_buffer_2 = HERReplayBuffer(
        max_size=args.replay_buffer_size/10,
        env=env,
        fraction_goals_are_rollout_goals = 0.0,
        fraction_resampled_goals_are_env_goals = 0.0,
        fraction_resampled_goals_are_replay_buffer_goals = 0.0,
        ob_keys_to_save     =["state_achieved_goal", "state_desired_goal"],
        desired_goal_keys   =["desired_goal", "state_desired_goal"],
        observation_key     = 'observation',
        desired_goal_key    = 'desired_goal',
        achieved_goal_key   = 'achieved_goal',
        vectorized          = vectorized 
    )
    # Initialize replay buffer for record learning data
    lhb = Learn_History_Buffer(1000, args.seed)
    DE = FlowDensity(replay_buffer_2, args.batch_size, args.device, train_batch_size=4*args.batch_size)
    policy = VE(env_name=args.env_name, state_dim=state_dim, action_dim=action_dim, goal_dim=goal_dim, alpha=args.alpha, Lambda=args.Lambda, sg_lr=args.sg_lr, \
                 q_lr=args.q_lr, pi_lr=args.pi_lr, device=args.device, writer=writer)
    path_builder = PathBuilder()


    main(env, args, policy, replay_buffer, path_builder, writer)