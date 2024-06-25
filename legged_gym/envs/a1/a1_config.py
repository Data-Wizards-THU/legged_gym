# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO, LeggedRobotCfgPPG
from legged_gym.envs.base.base_config import BaseConfig


class A1RoughCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.1,  # [rad]
            "RL_hip_joint": 0.1,  # [rad]
            "FR_hip_joint": -0.1,  # [rad]
            "RR_hip_joint": -0.1,  # [rad]
            "FL_thigh_joint": 0.8,  # [rad]
            "RL_thigh_joint": 1.0,  # [rad]
            "FR_thigh_joint": 0.8,  # [rad]
            "RR_thigh_joint": 1.0,  # [rad]
            "FL_calf_joint": -1.5,  # [rad]
            "RL_calf_joint": -1.5,  # [rad]
            "FR_calf_joint": -1.5,  # [rad]
            "RR_calf_joint": -1.5,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {"joint": 20.0}  # [N*m/rad]
        damping = {"joint": 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf"
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0


class A1RoughCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "flat_anymal_c"
        load_run = -1
        max_iterations = 300
        save_interval = 200  # check for potential saves every this many iterations


class A1RoughCfgPPG( LeggedRobotCfgPPG ):
    class algorithm( LeggedRobotCfgPPG.algorithm ):
        entropy_coef = 0.01
        n_pi = 16
    class runner( LeggedRobotCfgPPG.runner ):
        run_name = ''
        experiment_name = 'rough_a1'
    

class A1RoughCfgSAC(BaseConfig):
    seed = 8192
    runner_class_name = "LeggedGymRunner"

    # for agent
    class policy:
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        actor_activations = [
            "relu",
            "relu",
            "relu",
            "tanh",
        ]  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid, linear, softmax
        critic_activations = [
            "relu",
            "relu",
            "relu",
            "tanh",
        ]  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid, linear, softmax
        batch_size = 4096
        batch_count = 24

    # for agent
    class algorithm:
        # training params for SAC
        action_max = 100.0
        action_min = -100.0
        action_limit = 1.0
        reward_scale = 1.0
        actor_lr = 3e-4
        critic_lr = 3e-4
        alpha = 1e-4
        alpha_lr = 3e-3
        actor_noise_std = 1.0  # actor_noise_std(rsl_rl)
        chimera = True
        gradient_clip = 1.0  # gradient_clip(rsl_rl)
        log_std_max = 2
        log_std_min = -20
        storage_initial_size = 0
        storage_size = 1000000
        target_entropy = None

        # for AbstractActorCritic
        polyak = 0.995
        recurrent = False
        return_steps = 1
        _actor_input_size_delta = 0
        _critic_input_size_delta = 0

        # for Agent
        gamma = 0.99

    class runner:
        run_name = ""
        experiment_name = "SAC-a1"
        algorithm_class_name = "SAC"
        num_steps_per_env = 24  # per iteration
        max_iterations = 1500  # number of policy updates

        # logging
        save_interval = 1000  # check for potential saves every this many iterations
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt


class A1RoughCfgREDQ(BaseConfig):
    seed = 8192
    runner_class_name = "LeggedGymRunner"

    # for agent
    class policy:
        actor_hidden_dims = [256, 256, 256]
        critic_hidden_dims = [256, 256, 256]
        actor_activations = [
            "relu",
            "relu",
            "relu",
            "tanh",
        ]  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid, linear, softmax
        critic_activations = [
            "relu",
            "relu",
            "relu",
            "tanh",
        ]  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid, linear, softmax
        batch_size = 4096
        batch_count = 24  # utd_ratio

    # for agent
    class algorithm:
        # training params for REDQ
        num_Q = 10
        num_sample = 2
        q_target_mode = "min"
        policy_update_delay = 20
        action_max = 100.0
        action_min = -100.0
        action_limit = 1.0
        reward_scale = 1.0
        actor_lr = 3e-4
        critic_lr = 3e-4
        alpha = 1e-4
        alpha_lr = 3e-3
        actor_noise_std = 1.0  # actor_noise_std(rsl_rl)
        chimera = True
        gradient_clip = 1.0  # gradient_clip(rsl_rl)
        log_std_max = 2
        log_std_min = -20
        storage_initial_size = 0
        storage_size = 1000000
        target_entropy = None

        # for AbstractActorCritic
        polyak = 0.995
        recurrent = False
        return_steps = 1
        _actor_input_size_delta = 0
        _critic_input_size_delta = 0

        # for Agent
        gamma = 0.99

    class runner:
        run_name = ""
        experiment_name = "REDQ-a1"
        algorithm_class_name = "REDQ"
        num_steps_per_env = 24  # per iteration
        # num_steps_per_env = 1  # per iteration
        max_iterations = 1500  # number of policy updates

        # logging
        save_interval = 200  # check for potential saves every this many iterations
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
