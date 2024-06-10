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

from legged_gym.envs import task_registry
from legged_gym.utils import get_args

import os
import wandb
import time
from wandb_config import WANDB_API_KEY, WANDB_ENTITY
from rsl_rl.runners.callbacks import make_final_cb

os.environ["WANDB_API_KEY"] = WANDB_API_KEY

WANDB_LOG_ALGO = "PPO"


def make_wandb_cb(init_kwargs):
    assert "project" in init_kwargs, "The project must be specified in the init_kwargs."

    run = wandb.init(**init_kwargs)
    check_complete = make_final_cb(lambda *_: run.finish())

    def cb(runner, stat):
        mean_reward = (
            sum(stat["returns"]) / len(stat["returns"])
            if len(stat["returns"]) > 0
            else 0.0
        )
        mean_steps = (
            sum(stat["lengths"]) / len(stat["lengths"])
            if len(stat["lengths"]) > 0
            else 0.0
        )
        total_steps = (
            stat["current_iteration"] * runner.env.num_envs * runner._num_steps_per_env
        )
        step_collection_time = stat["collection_time"]
        step_update_time = stat["update_time"]
        step_training_time = stat["total_time"]
        training_time = stat["training_time"]
        sample_count = stat["sample_count"]

        data = {
            "mean_rewards": mean_reward,
            "mean_steps": mean_steps,
            "training_steps": total_steps,
            "training_time": training_time,
            "step_collection_time": step_collection_time,
            "step_update_time": step_update_time,
            "step_training_time": step_training_time,
            "sample_count": sample_count,
        }
        if WANDB_LOG_ALGO == "SAC":
            data.update(
                {
                    "actor_loss": stat["loss"]["actor"],
                    "alpha_loss": stat["loss"]["alpha"],
                    "critic1_loss": stat["loss"]["critic1"],
                    "critic2_loss": stat["loss"]["critic2"],
                    "log_alpha": runner.agent.log_alpha,
                }
            )
        elif WANDB_LOG_ALGO == "REDQ":
            avg_critic_loss = (
                sum([stat["loss"][f"critic_{i}"] for i in range(runner.agent.num_Q)])
                / runner.agent.num_Q
            )
            data.update(
                {
                    "actor_loss": stat["loss"]["actor"],
                    "alpha_loss": stat["loss"]["alpha"],
                    "avg_critic_loss": avg_critic_loss,
                    "log_alpha": runner.agent.log_alpha,
                }
            )

        run.log(data)

        check_complete(runner, stat)

    return cb


def get_all(cls):
    return {
        attr: getattr(cls, attr)
        for attr in dir(cls)
        if not callable(getattr(cls, attr)) and not attr.startswith("__")
    }


def train(args):
    global WANDB_LOG_ALGO
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # use default PPO runner
    # runner, train_cfg = task_registry.make_alg_runner(
    #     env=env, name=args.task, args=args
    # )
    # use the config named "experiment_name"(algorithm + task name)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.experiment_name, args=args
    )
    env_cfg_dict = dict()
    for key, value in vars(env_cfg).items():
        env_cfg_dict[key] = get_all(value)
    train_cfg_dict = dict()
    for key, value in vars(train_cfg).items():
        train_cfg_dict[key] = get_all(value)
    args_cfg_dict = vars(args)
    config = {"env": env_cfg_dict, "train": train_cfg_dict, "args": args_cfg_dict}
    # config wandb
    task_name = config["args"]["task"]
    alg_name = config["train"]["runner"]["algorithm_class_name"]
    WANDB_LOG_ALGO = alg_name
    run_name = alg_name + "-" + task_name + "-" + time.ctime()
    wandb_learn_config = dict(
        config=config,
        entity=WANDB_ENTITY,
        group=f"{task_name}_{alg_name}",
        project="legged_gym",
        tags=[alg_name, task_name, "train"],
        name=run_name,
    )
    runner._learn_cb.append(make_wandb_cb(wandb_learn_config))
    runner.learn(iterations=train_cfg.runner.max_iterations)


if __name__ == "__main__":
    args = get_args()
    train(args)
