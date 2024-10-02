from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from copy import copy
from gym import utils
from gym.envs.mujoco import mujoco_env


class Reacher3DEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.viewer = None
        self.num_timesteps = 0
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.goal = np.zeros(3)
        mujoco_env.MujocoEnv.__init__(
            self, os.path.join(dir_path, "assets/classic_mujoco/reacher3d.xml"), 2
        )

    def step(self, a):
        self.num_timesteps += 1
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        # reward_ctrl = 0.0001 * -np.square(a).sum()

        achieved_goal = self.get_EE_pos(obs[None]).squeeze()


        fail = True

        xy_distance = np.sqrt(np.sum(np.square(achieved_goal - self.goal), axis=-1))
        if xy_distance <= 0.25:
            fail = False


        obs_dict = {
            "observation": obs,
            "desired_goal": self.goal,
            "achieved_goal": achieved_goal,
            'state_observation': obs,
            'state_desired_goal': self.goal,
            'state_achieved_goal': achieved_goal
        }

        return (
            obs_dict,
            -float(fail),
            self.num_timesteps >= 100,
            {"is_success": not fail,
             "xy-distance": [xy_distance]},
        )

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = 2.5
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 270

    def reset_model(self):
        qpos, qvel = np.copy(self.init_qpos), np.copy(self.init_qvel)
        qpos[-3:] += np.random.normal(loc=0, scale=0.1, size=[3])
        qvel[-3:] = 0
        self.goal = qpos[-3:]
        self.set_state(qpos, qvel)
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat,
                self.data.qvel.flat[:-3],
            ]
        )


    def compute_rewards(self, action, obs):
        goal = obs["desired_goal"]
        achieved = obs['achieved_goal']
        dis = np.sqrt(np.sum(np.square(goal - achieved), axis=-1))
        r = np.where(dis<0.25,0.,-1.)
        return r
    

    def get_EE_pos(self, states):
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = (
            states[:, :1],
            states[:, 1:2],
            states[:, 2:3],
            states[:, 3:4],
            states[:, 4:5],
            states[:, 5:6],
            states[:, 6:],
        )

        rot_axis = np.concatenate(
            [
                np.cos(theta2) * np.cos(theta1),
                np.cos(theta2) * np.sin(theta1),
                -np.sin(theta2),
            ],
            axis=1,
        )
        rot_perp_axis = np.concatenate(
            [-np.sin(theta1), np.cos(theta1), np.zeros(theta1.shape)], axis=1
        )
        cur_end = np.concatenate(
            [
                0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2),
                0.1 * np.sin(theta1) + 0.4 * np.sin(theta1) * np.cos(theta2) - 0.188,
                -0.4 * np.sin(theta2),
            ],
            axis=1,
        )

        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = np.cross(rot_axis, rot_perp_axis)
            x = np.cos(hinge) * rot_axis
            y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
            z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
            new_rot_axis = x + y + z
            new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
            new_rot_perp_axis[
                np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30
            ] = rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
            new_rot_perp_axis /= np.linalg.norm(
                new_rot_perp_axis, axis=1, keepdims=True
            )
            rot_axis, rot_perp_axis, cur_end = (
                new_rot_axis,
                new_rot_perp_axis,
                cur_end + length * new_rot_axis,
            )

        return cur_end

    def reset(self):
        self.num_timesteps = 0
        return super().reset()


if __name__ == "__main__":
    env = Reacher3DEnv()
    done = False
    obs = env.reset()
    counter = 0
    import pdb

    pdb.set_trace()
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        counter += 1
        print(obs, reward, done, info)
    print(counter)
