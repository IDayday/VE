import numpy as np
import torch

def evalPolicy(policy, env, N=100, Tmax=100, distance_threshold=0.5, writer=None, test_time=0, action_bound=1):
    final_distance = []
    successes = [] 

    for _ in range(N):
        obs = env.reset()
        done = False
        state = obs["observation"]
        goal = obs["desired_goal"]
        t = 0
        while not done:
            action = policy.select_action(state, goal)
            next_obs, _, _, status = env.step(action_bound*action) 
            next_state = next_obs["observation"]
            state = next_state
            done = status["xy-distance"][0]< distance_threshold or t >= Tmax
            t += 1

        final_distance.append(status["xy-distance"][0])
        successes.append( 1.0 * (status["xy-distance"][0] < distance_threshold ))

    eval_distance, success_rate =np.mean(final_distance), np.mean(successes)
    # writer.add_scalar("success_rate",success_rate,test_time)
    # writer.add_scalar("eval_distance",eval_distance,test_time)
    return eval_distance, success_rate

def evalPolicy_img(policy, env, N=100, Tmax=100, distance_threshold=0.05, logger=None, writer=None, test_time=0, action_bound=1):
    final_rewards = []
    successes = [] 
    puck, arm = [], []
    puck_dist, arm_dist = [], []
    for _ in range(N):
        obs = env.reset()
        done = False
        state = obs["image_observation"]
        goal = obs["image_desired_goal"]
        t = 0
        while not done:
            action = policy.select_action(state, goal)
            next_obs, reward, _, _ = env.step(action_bound*action) 
            last_reward = reward
            
            next_state = next_obs["image_observation"]
            state = next_state
            reward = - np.sqrt(np.power(np.array(reward).reshape(2, 2), 2).sum(-1)).max(-1)    
            done = reward > -distance_threshold or t >= Tmax
            t += 1
        final_rewards.append(reward)
        successes.append( 1.0 * (reward > -distance_threshold ))

        # Additional info about arm and puck
        arm.append(1.0*(np.sqrt(np.power(np.array(last_reward[:2]), 2).sum()) < 0.06) )
        puck.append( 1.0*(np.sqrt(np.power(np.array(last_reward[2:]), 2).sum()) < 0.06) )
        arm_dist.append( np.sqrt(np.power(np.array(last_reward[:2]), 2).sum()) )
        puck_dist.append(np.sqrt(np.power(np.array(last_reward[2:]), 2).sum()) )

    eval_reward, success_rate = np.mean(final_rewards), np.mean(successes)

    # writer.add_scalar("success_rate",success_rate,test_time)
    # writer.add_scalar("eval_reward",eval_reward,test_time)
    return eval_reward, success_rate

def Select_relabel_goal(batch_data, goals, de, mean_density):
    state = batch_data["observations"]
    goal = batch_data["resampled_goals"]
    batch_size = state.shape[0]
    state_dim = state.shape[-1]
    key_goals_num = goals.shape[0]
    key_goals = goal[:key_goals_num,:]
    key_goals[:,:2] = goals.reshape(key_goals_num,-1)[:,:2]
    index = np.expand_dims(np.arange(batch_size),1)

    state_array = np.repeat(np.expand_dims(state, axis=1), key_goals_num, axis=1)
    key_goals_array = np.repeat(np.expand_dims(key_goals, axis=0), batch_size, axis=0)

    state_goal_array = np.concatenate((state_array.reshape(-1, state_dim)[:,:2], key_goals_array.reshape(-1, state_dim)[:,:2]),axis=-1)
    state_goal_log_density = de.evaluate_log_density(state_goal_array).reshape(batch_size, -1)
    density = np.exp(state_goal_log_density)

    for i in range(int(batch_size/2)):
        d_mean = density[i].mean()

        condition1 = (density[i]>0.8*d_mean)
        condition2 = (density[i]<1.2*d_mean)

        idx = np.where((condition1)&(condition2))[0]
        if len(idx)>0:
            g_idx = np.random.choice(idx)
            goal[i] = key_goals_array[i,g_idx]
        else:
            continue

    batch_data["resampled_goals"] = goal

    test_state_regoal_array = np.concatenate((state[:,:2], goal[:,:2]),axis=-1)
    test_state_regoal_density = np.clip(np.exp(de.evaluate_log_density(test_state_regoal_array).reshape(batch_size, -1)),0.0,1.0).mean()
    return batch_data, test_state_regoal_density


def sample_and_preprocess_batch(replay_buffer, t, tag=None, batch_size=1024, state_dim=31, distance_threshold=0.5, de=None, policy=None, mean_density=1.0, device=torch.device("cuda")):
    if tag == "her":
        batch = replay_buffer.her_random_batch(batch_size)
    elif tag == "final":
        batch = replay_buffer.her_final_batch(batch_size)
    elif tag == "buffer":
        batch = replay_buffer.buffer_random_batch(batch_size)
        candidate_goal = replay_buffer.random_state_batch(500, state_dim)
        batch, _ = Select_relabel_goal(batch, candidate_goal, de, mean_density)

    state_batch         = batch["observations"]
    action_batch        = batch["actions"]
    next_state_batch    = batch["next_observations"]
    goal_batch          = batch["resampled_goals"]
    reward_batch        = batch["rewards"]
    done_batch          = batch["terminals"] 
    subgoal_batch       = batch["possible_subgoals"]
    
    # Compute sparse rewards: -1 for all actions until the goal is reached
    reward_batch = - np.sqrt(np.power(np.array(next_state_batch - goal_batch)[:, :2], 2).sum(-1, keepdims=True))
    done_batch   = 1.0 * (reward_batch > -distance_threshold) 
    reward_batch = - np.ones_like(done_batch)

    # Convert to Pytorch
    state_batch         = torch.FloatTensor(state_batch).to(device)
    action_batch        = torch.FloatTensor(action_batch).to(device)
    reward_batch        = torch.FloatTensor(reward_batch).to(device)
    next_state_batch    = torch.FloatTensor(next_state_batch).to(device)
    done_batch          = torch.FloatTensor(done_batch).to(device)
    goal_batch          = torch.FloatTensor(goal_batch).to(device)
    subgoal_batch       = torch.FloatTensor(subgoal_batch).to(device)

    return state_batch, action_batch, reward_batch, next_state_batch, done_batch, goal_batch, subgoal_batch

