#!/bin/bash

conda activate rl
# python legged_gym/scripts/train.py --headless --task=a1 --experiment_name=a1 --run_name=zyx
# python legged_gym/scripts/train.py --headless --task=anymal_c_flat --experiment_name=SAC-anymal_c_flat --run_name=zyx-SAC-anymal_c_flat --max_iterations=5000
# python legged_gym/scripts/train.py --headless --task=anymal_c_flat --experiment_name=REDQ-anymal_c_flat --run_name=zyx-REDQ-anymal_c_flat --resume --load_run=Jun04_00-25-17_REDQ-SAC-anymal_c_flat --checkpoint=2000 --max_iterations=5000
# python legged_gym/scripts/train.py --headless --task=anymal_c_flat --experiment_name=PPO-anymal_c_flat --run_name=zyx-PPO-anymal_c_flat --max_iterations=5000

# python legged_gym/scripts/train.py --headless --task=a1 --experiment_name=PPO-a1 --run_name=zyx-PPO-a1 --max_iterations=5000
python legged_gym/scripts/train.py --headless --task=a1 --experiment_name=REDQ-a1 --run_name=zyx-REDQ-a1 --max_iterations=5000
# python legged_gym/scripts/train.py --headless --task=a1 --experiment_name=SAC-a1 --run_name=zyx-SAC-a1 --max_iterations=5000