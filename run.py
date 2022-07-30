import subprocess
import os
from  Project.BoxScript import Boxes
import sys

cwd = os.getcwd()
# py_path = "D:/Work/Installations/anaconda3/envs/RL/python.exe"

# parser = argparse.ArgumentParser()
# parser.add_argument('env_idx',type=int,help='Environment identifier')
# args = parser.parse_args()

for env_idx in sys.argv[1:]:
    ENV_NAME = list(Boxes.keys())[int(env_idx)]
    EXP_REWARD = Boxes[ENV_NAME]["Expected Reward"]
    box_types = ["Deterministic","Random","Extreme"]

    for BOX_TYPE in box_types:
        subprocess.run("python ./Project/train.py --env_name {} --env_version {} --expected_reward {}".format(ENV_NAME, BOX_TYPE, EXP_REWARD), cwd=cwd, shell=True)