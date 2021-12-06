import argparse

parser = argparse.ArgumentParser()

#positional args
parser.add_argument('env_name',help='Gym Environment Name')
parser.add_argument('weight_file',help="location of the required HDF5 file")

#optional args
parser.add_argument('-bs','--batch_size',type=int,help='Batch size')

#agent class args
parser.add_argument('-G','--gamma',type=float,help='Gamma value')
parser.add_argument('-gl','--gae_lambda',type=float,help='GAE lambda value')
parser.add_argument('-clr','--clipping_loss_ratio',type=float,help='Clipping Loss Ratio')
parser.add_argument('-elr','--entropy_loss_ratio',type=float,help='Entropy Loss Ratio')
parser.add_argument('-tua','--target_update_alpha',type=float)

args = parser.parse_args()

import numpy as np
import os
import time
import gym
from collections import deque
import tensorflow as tf
from tensorflow import keras as K
from Agent import *
import sys
import csv
import json

ENV_NAME = args.env_name
WEIGHT_FILE = args.weight_file
BATCH_SIZE = 16

if args.batch_size != None:
    BATCH_SIZE = args.batch_size

GAMMA=0.99
GAE_LAMBDA=0.95
CLIPPING_LOSS_RATIO=0.1
ENTROPY_LOSS_RATIO=0.001
TARGET_UPDATE_ALPHA=0.9

if args.gamma != None:
    GAMMA = args.gamma
if args.gae_lambda != None:
    GAE_LAMBDA = args.gae_lambda
if args.clipping_loss_ratio != None:
    CLIPPING_LOSS_RATIO = args.clipping_loss_ratio
if args.entropy_loss_ratio != None:
    ENTROPY_LOSS_RATIO = args.entropy_loss_ratio
if args.target_update_alpha != None:
    TARGET_UPDATE_ALPHA = args.target_update_alpha

env = gym.make(ENV_NAME)
agent = Agent(env.action_space.n, env.observation_space.shape, BATCH_SIZE, \
    GAMMA, GAE_LAMBDA, CLIPPING_LOSS_RATIO, ENTROPY_LOSS_RATIO, TARGET_UPDATE_ALPHA)
samples_filled = 0

agent.actor_network.load_weights(WEIGHT_FILE)

if not os.path.isfile('test_data.csv'):
    with open('test_data.csv','w+') as csvfile:
        csvwriter = csv.writer(csvfile)
        fields = ['Gym Environment','Episode','Episodic Reward','Weight File']
        csvwriter.writerow(fields)

try:
    with open('test_data.csv','a+') as csvfile:
        csvwriter = csv.writer(csvfile,lineterminator="\n")
        row = [ENV_NAME,0,0,WEIGHT_FILE]
        for i in range(20):
            episode_reward = 0
            state = env.reset()
            while True:
                action = agent.choose_action(state)
                #env.render()
                state, reward, done, _ = env.step(action)
                if done:
                    break
                episode_reward += reward
            #print('Episodes:', i, 'Episodic_Reward:', episode_reward)
            row[1],row[2] = i,episode_reward
            csvwriter.writerow(row)

except:
    with open('errorlogs.log','a+') as errlog:
        errlog.writelines("Test Error: "+str(ENV_NAME)+" "+str(WEIGHT_FILE)+"\n")

env.close()