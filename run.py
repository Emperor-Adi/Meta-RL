import subprocess
import os
from  Project.BoxScript import Boxes
import argparse

cwd = os.getcwd()
py_path = "D:/Work/Installations/anaconda3/envs/RL/python.exe"

parser = argparse.ArgumentParser()
parser.add_argument('env_idx',type=int,help='Environment identifier')
args = parser.parse_args()

ENV_NAME = list(Boxes.keys())[args.env_idx]
EXP_REWARD = Boxes[ENV_NAME]["Expected Reward"]
box_types = ["Deterministic","Random","Extreme"]

for BOX_TYPE in box_types:
    subprocess.run("{} ./Project/train.py --env_name {} --env_version {} --expected_reward {}".format(py_path, ENV_NAME, BOX_TYPE, EXP_REWARD), cwd=cwd, shell=True)