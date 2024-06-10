#!/bin/bash

conda activate rl
# python legged_gym/scripts/play.py --headless --task=a1 --experiment_name=a1 --run_name=zyx
# python legged_gym/scripts/play.py --headless --task=anymal_c_flat --experiment_name=SAC-anymal_c_flat --run_name=zyx-SAC-anymal_c_flat --max_iterations=5000
python legged_gym/scripts/play.py --headless --task=anymal_c_flat --experiment_name=REDQ-anymal_c_flat --run_name=zyx-REDQ-anymal_c_flat --resume --load_run=Jun04_17-54-46_zyx-REDQ-anymal_c_flat --checkpoint=6999
# python legged_gym/scripts/play.py --headless --task=anymal_c_flat --experiment_name=PPO-anymal_c_flat --run_name=zyx-PPO-anymal_c_flat --max_iterations=5000

# python legged_gym/scripts/play.py --headless --task=a1 --experiment_name=PPO-a1 --run_name=zyx-PPO-a1 --max_iterations=5000
# python legged_gym/scripts/play.py --headless --task=a1 --experiment_name=REDQ-a1 --run_name=zyx-REDQ-a1 --max_iterations=5000