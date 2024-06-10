from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.envs import Anymal, AnymalCFlatCfg, AnymalCFlatCfgSAC
from rsl_rl.algorithms import *
from rsl_rl.env.gym_env import GymEnv
from rsl_rl.runners.runner import Runner

from datetime import datetime
import optuna
import random
import os
import torch
import numpy as np


ALGORITHM = None
EXPERIMENT_DIR = os.environ.get("EXPERIMENT_DIRECTORY", "./")
EXPERIMENT_NAME = None

TRIALS = 100

EVAL_AGENTS = 64
EVAL_RUNS = 3
# EVAL_RUNS = 1
EVAL_STEPS = 300
# EVAL_STEPS = 10

TRAIN_ITERATIONS = None
TRAIN_TIMEOUT = 60 * 15  # 10 minutes
TRAIN_RUNS = 1
# TRAIN_RUNS = 1
TRAIN_SEED = None

APP_ARGS = None


def seed(s=None):
    seed = int(datetime.now().timestamp() * 1e6) % 2**32 if s is None else s

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_sac_hyperparams(trial, register_name):
    # for policy
    # actor_hidden_dims = trial.suggest_categorical(
    #     "actor_hidden_dims", [[64, 64, 64], [256, 256, 256], [512, 256, 128]]
    # )
    actor_hidden_dims = [256, 256, 256]
    # critic_hidden_dims = trial.suggest_categorical(
    #     "critic_hidden_dims", [[64, 64, 64], [256, 256, 256], [512, 256, 128]]
    # )
    critic_hidden_dims = [256, 256, 256]
    activations = (
        [  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid, linear, softmax
            ["relu", "relu", "relu", "tanh"],
            ["relu", "relu", "relu", "linear"],
            ["tanh", "tanh", "tanh", "linear"],
            ["elu", "elu", "elu", "tanh"],
            ["elu", "elu", "elu", "linear"],
        ]
    )
    actor_activations_id = trial.suggest_categorical(
        "actor_activations_id", [i for i in range(len(activations))]
    )
    actor_activations = activations[actor_activations_id]
    # batch_count = trial.suggest_int("batch_count", 4, 12)
    batch_count = 24
    # batch_count = 4 # batch_count(rsl_rl) mini batch size = num_envs*nsteps / nminibatches
    # batch_size = trial.suggest_int("batch_size", 256, 8192)
    batch_size = 4096

    # for SAC
    action_max = 100.0
    action_min = -100.0
    actor_lr = trial.suggest_float("actor_lr", 1e-3, 1e-1)
    # actor_noise_std = trial.suggest_float("actor_noise_std", 0.5, 1.0)
    actor_noise_std = 1.0
    alpha = trial.suggest_float("alpha", 0.1, 1)
    alpha_lr = trial.suggest_float("alpha_lr", 1e-3, 1e-1)
    # chimera = trial.suggest_categorical("chimera", [True, False])
    chimera = True
    critic_lr = trial.suggest_float("critic_lr", 1e-3, 1e-1)

    # gradient_clip = trial.suggest_float("gradient_clip", 0.8, 1.0)
    gradient_clip = 1.0
    log_std_max = trial.suggest_float("log_std_max", 1.0, 10)
    log_std_min = trial.suggest_float("log_std_min", -20.0, 0.0)

    storage_initial_size = 0
    storage_size = 100000
    target_entropy = None

    # for AbstractActorCritic
    # polyak = trial.suggest_float("polyak", 0.995, 0.998)
    polyak = 0.995

    recurrent = False
    return_steps = 1
    _actor_input_size_delta = 0
    _critic_input_size_delta = 0

    # for Agent
    # gamma = trial.suggest_float("gamma", 0.95, 0.99)
    gamma = 0.99

    # for runner
    algorithm_class_name = "SAC"
    # num_steps_per_env = trial.suggest_categorical(
    #     "num_steps_per_env", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # )
    num_steps_per_env = 24
    # num_steps_per_env = trial.suggest_categorical(
    #     "num_steps_per_env", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # )
    # max_iterations = 1500 # number of policy updates

    # logging
    save_interval = 50  # check for potential saves every this many iterations
    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt

    sac_config = AnymalCFlatCfgSAC()
    # for policy
    sac_config.policy.actor_hidden_dims = actor_hidden_dims
    sac_config.policy.critic_hidden_dims = critic_hidden_dims
    sac_config.policy.actor_activations = actor_activations
    sac_config.policy.batch_count = batch_count
    sac_config.policy.batch_size = batch_size
    # for SAC
    sac_config.algorithm.action_max = action_max
    sac_config.algorithm.action_min = action_min
    sac_config.algorithm.actor_lr = actor_lr
    sac_config.algorithm.actor_noise_std = actor_noise_std
    sac_config.algorithm.alpha = alpha
    sac_config.algorithm.alpha_lr = alpha_lr
    sac_config.algorithm.chimera = chimera
    sac_config.algorithm.critic_lr = critic_lr
    sac_config.algorithm.gradient_clip = gradient_clip
    sac_config.algorithm.log_std_max = log_std_max
    sac_config.algorithm.log_std_min = log_std_min
    sac_config.algorithm.storage_initial_size = storage_initial_size
    sac_config.algorithm.storage_size = storage_size
    sac_config.algorithm.target_entropy = target_entropy
    # for AbstractActorCritic
    sac_config.algorithm.polyak = polyak
    sac_config.algorithm.recurrent = recurrent
    sac_config.algorithm.return_steps = return_steps
    sac_config.algorithm._actor_input_size_delta = _actor_input_size_delta
    sac_config.algorithm._critic_input_size_delta = _critic_input_size_delta
    # for Agent
    sac_config.algorithm.gamma = gamma
    # for runner
    sac_config.runner.algorithm_class_name = algorithm_class_name
    sac_config.runner.num_steps_per_env = num_steps_per_env
    sac_config.runner.save_interval = save_interval
    sac_config.runner.resume = resume
    sac_config.runner.load_run = load_run
    sac_config.runner.checkpoint = checkpoint
    sac_config.runner.resume_path = resume_path

    # env_count = trial.suggest_categorical(
    #     "env_count", [1, 8, 16, 32, 64, 128, 256, 512]
    # )
    env_cfg = AnymalCFlatCfg()
    # env_cfg.env.num_envs = env_count
    task_registry.register(register_name, Anymal, env_cfg, sac_config)

    # agent_kwargs = dict(
    #     # for policy
    #     actor_hidden_dims=actor_hidden_dims,
    #     critic_hidden_dims=critic_hidden_dims,
    #     actor_activations=actor_activations,
    #     batch_count=batch_count,
    #     batch_size=batch_size,
    #     # for SAC
    #     action_max=action_max,
    #     action_min=action_min,
    #     actor_lr=actor_lr,
    #     actor_noise_std=actor_noise_std,
    #     alpha=alpha,
    #     alpha_lr=alpha_lr,
    #     chimera=chimera,
    #     critic_lr=critic_lr,
    #     gradient_clip=gradient_clip,
    #     log_std_max=log_std_max,
    #     log_std_min=log_std_min,
    #     storage_initial_size=storage_initial_size,
    #     storage_size=storage_size,
    #     target_entropy=target_entropy,
    #     # for AbstractActorCritic
    #     polyak=polyak,
    #     recurrent=recurrent,
    #     return_steps=return_steps,
    #     _actor_input_size_delta=_actor_input_size_delta,
    #     _critic_input_size_delta=_critic_input_size_delta,
    #     # for Agent
    #     gamma=gamma,
    #     # for runner
    #     algorithm_class_name=algorithm_class_name,
    #     num_steps_per_env=num_steps_per_env,
    #     save_interval=save_interval,
    #     resume=resume,
    #     load_run=load_run,
    #     checkpoint=checkpoint,
    #     resume_path=resume_path,
    # )

    # env_count = trial.suggest_categorical(
    #     "env_count", [1, 8, 16, 32, 64, 128, 256, 512]
    # )

    # env_kwargs = dict(environment_count=env_count)
    # return agent_kwargs, env_kwargs, runner_kwargs


samplers = {"SAC": sample_sac_hyperparams}
ENV = None


def objective(trial):
    seed()

    evaluations = []
    for instantiation in range(TRAIN_RUNS):
        seed(TRAIN_SEED)
        register_name = f"{EXPERIMENT_NAME}-instantiation"
        samplers[ALGORITHM](trial, register_name)

        runner, train_cfg = task_registry.make_alg_runner(
            env=ENV,
            name=register_name,
            args=APP_ARGS,
        )
        runner._learn_cb = [
            lambda _, stat: runner._log_progress(
                stat, prefix=f"learn {instantiation+1}/{TRAIN_RUNS}"
            )
        ]

        # eval_env_kwargs = copy.deepcopy(env_kwargs)
        # eval_env_kwargs["environment_count"] = EVAL_AGENTS
        # eval_env, eval_env_cfg = task_registry.make_env(
        #     name=APP_ARGS.task,
        #     args=APP_ARGS,
        # )
        # eval_env, env_cfg = task_registry.make_env(
        #     name=register_name,
        #     args=APP_ARGS,
        # )
        eval_runner, train_cfg = task_registry.make_alg_runner(
            env=ENV,
            name=register_name,
            args=APP_ARGS,
        )
        eval_runner.agent = runner.agent
        eval_runner._eval_cb = [
            lambda _, stat: runner._log_progress(
                stat, prefix=f"eval {instantiation+1}/{TRAIN_RUNS}"
            )
        ]

        try:
            runner.learn(iterations=TRAIN_ITERATIONS, timeout=TRAIN_TIMEOUT)
        except Exception:
            raise optuna.TrialPruned()

        intermediate_evaluations = []
        for eval_run in range(EVAL_RUNS):
            eval_runner._eval_cb = [
                lambda _, stat: runner._log_progress(
                    stat, prefix=f"eval {eval_run+1}/{EVAL_RUNS}"
                )
            ]

            seed()
            with torch.inference_mode():
                eval_runner.env.reset()
            intermediate_evaluations.append(eval_runner.evaluate(steps=EVAL_STEPS))
        eval = np.mean(intermediate_evaluations)

        trial.report(eval, instantiation)
        if trial.should_prune():
            raise optuna.TrialPruned()

        evaluations.append(eval)

    evaluation = np.mean(evaluations)

    return evaluation


def tune():
    assert (
        TRAIN_RUNS == 1 or TRAIN_SEED is None
    ), "If multiple runs are used, the seed must be None."

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{EXPERIMENT_DIR}/{EXPERIMENT_NAME}.db"
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)

    try:
        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            storage=storage,
            study_name=EXPERIMENT_NAME,
        )
    except Exception:
        study = optuna.load_study(
            pruner=pruner, storage=storage, study_name=EXPERIMENT_NAME
        )

    study.optimize(objective, n_trials=TRIALS)


if __name__ == "__main__":
    APP_ARGS = get_args()
    experiment_name = APP_ARGS.experiment_name
    ALGORITHM = experiment_name.split("-")[0]
    EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", f"tune-{experiment_name}")
    TRAIN_ITERATIONS = APP_ARGS.max_iterations
    ENV, env_cfg = task_registry.make_env(
        name=APP_ARGS.task,
        args=APP_ARGS,
    )
    tune()
