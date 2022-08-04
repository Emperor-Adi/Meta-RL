import subprocess
import os
from  Project.BoxScript import Boxes
import sys

cwd = os.getcwd()

if sys.argv[1] == 'train':
    for env_idx in sys.argv[2:]:
        ENV_NAME = list(Boxes.keys())[int(env_idx)]
        EXP_REWARD = Boxes[ENV_NAME]["Expected Reward"]
        box_types = ["Deterministic","Random","Extreme"]

        for BOX_TYPE in box_types:
            subprocess.run("python ./Project/train.py --env_name {} --env_version {} --expected_reward {}".format(ENV_NAME, BOX_TYPE, EXP_REWARD), cwd=cwd, shell=True)

elif sys.argv[1] == 'test':
    pass

else:
    print("Invalid command")