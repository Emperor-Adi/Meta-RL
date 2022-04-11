from cmath import exp
import numpy as np
import os
import gym
from collections import deque
from Agent import *
from CustomEnvs import SpecialEnv
import csv
import argparse
from datetime import datetime

def main():
    #Handling command line args
    parser = argparse.ArgumentParser()

    #positional arguments
    parser.add_argument('--env_name',help='The Environment name from gym')
    parser.add_argument('--env_version',help='Decides the value range of the custom parameters')
    parser.add_argument('--expected_reward',type=float , help='The reward at which Environment is considered solved')


    #optional arguments
    parser.add_argument('-ti','--train_iterations',type=int,help='Maximum training iterations')
    parser.add_argument('-mel','--max_episode_length',type=int,help='Maximum steps in each episode')
    parser.add_argument('-tbs','--trajectory_buffer_size',type=int,help='Trajectory Buffer Size')
    parser.add_argument('-bs','--batch_size',type=int,help='Batch size')
    parser.add_argument('-re','--render_every',type=int,help='Render every this many episodes')

    #agent class args
    parser.add_argument('-G','--gamma',type=float,help='Gamma value')
    parser.add_argument('-gl','--gae_lambda',type=float,help='GAE lambda value')
    parser.add_argument('-clr','--clip_loss_ratio',type=float,help='Clipping Loss Ratio')
    parser.add_argument('-elr','--entropy_loss_ratio',type=float,help='Entropy Loss Ratio')
    parser.add_argument('-A','--alpha',type=float)

    args = parser.parse_args()


    ENV_NAME = args.env_name
    ENV_VERSION = args.env_version
    EXPECTED_REWARD = args.expected_reward
    # ENV_NAME = "CartPole-v1"
    # ENV_VERSION = "Deterministic"
    # EXPECTED_REWARD = 475


    TRAIN_ITERATIONS = 40000
    if args.train_iterations != None :
        TRAIN_ITERATIONS = args.train_iterations

    MAX_EPISODE_LENGTH = 16000
    if args.max_episode_length !=None :
        MAX_EPISODE_LENGTH = args.max_episode_length

    TRAJECTORY_BUFFER_SIZE = 32
    if args.trajectory_buffer_size !=None :
        TRAJECTORY_BUFFER_SIZE = args.trajectory_buffer_size

    BATCH_SIZE = 16
    if args.batch_size !=None :
        BATCH_SIZE = args.batch_size

    RENDER_EVERY = 10
    if args.render_every !=None :
        RENDER_EVERY = args.render_every

    GAMMA=0.99
    if args.gamma != None:
        GAMMA = args.gamma

    GAE_LAMBDA=0.95
    if args.gae_lambda != None:
        GAE_LAMBDA = args.gae_lambda

    CLIP_LOSS_RATIO=0.1
    if args.clip_loss_ratio != None:
        CLIP_LOSS_RATIO = args.clip_loss_ratio

    ENTROPY_LOSS_RATIO=0.001
    if args.entropy_loss_ratio != None:
        ENTROPY_LOSS_RATIO = args.entropy_loss_ratio

    ALPHA=0.9
    if args.alpha != None:
        ALPHA = args.alpha


    env = SpecialEnv(gym.make(ENV_NAME),ENV_NAME,ENV_VERSION)
    agent = Agent(env.action_space.n, env.observation_space.shape, BATCH_SIZE, \
        GAMMA, GAE_LAMBDA, CLIP_LOSS_RATIO, ENTROPY_LOSS_RATIO, ALPHA)
    samples_filled = 0

    try:
        with open('./Logs/train_data_'+ENV_NAME+'_'+ENV_VERSION+'_'+str(datetime.now().timestamp())+'.csv','w+') as csvfile:
            csvwriter = csv.writer(csvfile,lineterminator="\n")
            fields = ['Gym Environment','Episode','Episodic Reward','Expected Reward','Solved In','Batch Size']
            csvwriter.writerow(fields)
            scores_window = deque(maxlen=100)
            scores = []
            max_reward = -500
            for ep_count in range(TRAIN_ITERATIONS):
                s = env.reset()
                r_sum = 0
                row = [ENV_NAME,ENV_VERSION,None,None,EXPECTED_REWARD,None,BATCH_SIZE]
                for count_step in range(MAX_EPISODE_LENGTH):
                    # if count_step % RENDER_EVERY == 0 :
                    #     env.render()
                    a = agent.choose_action(s)
                    s_, r, done, _ = env.step(a)
                    r_sum += r
                    agent.store_transition(s, a, s_, r, done)
                    samples_filled += 1
                    if samples_filled == TRAJECTORY_BUFFER_SIZE:
                        for _ in range(TRAJECTORY_BUFFER_SIZE // BATCH_SIZE):
                            agent.train_network()
                        agent.memory.clear()
                        samples_filled = 0
                    s = s_
                    if done:
                        break
                scores_window.append(r_sum)
                scores.append(r_sum)
                row[2],row[3] = ep_count,r_sum
                if np.mean(scores_window)>=EXPECTED_REWARD:
                    # print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(ep_count-100, np.mean(scores_window)))
                    agent.actor_network.save_weights("Models/"+ENV_NAME+"_"+ENV_VERSION+"_"+str(r_sum)+".h5")
                    row[5],row[6] = ep_count-100,np.mean(scores_window)
                    csvwriter.writerow(row)
                    break
                max_reward = max(max_reward, r_sum)
                print('Episodes:', ep_count, 'Episodic_Reward:', r_sum)
                csvwriter.writerow(row)

    except Exception as e:
        print(e)
        with open('./Logs/errorlogs.log','a+') as errlog:
            errlog.writelines("Train Error: "+str(ENV_NAME)+" "+str(ENV_VERSION)+" "+str(EXPECTED_REWARD))
            errlog.writelines("\nTimeStamp: "+str(datetime.now())+"\n"+str(e)+"\n\n")

if __name__ == '__main__':
    main()