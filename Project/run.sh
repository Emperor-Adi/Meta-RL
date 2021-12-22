#!/bin/bash

environments=( "CartPole-v0" "CartPole-v1" "Acrobot-v1" "LunarLander-v2" )

declare -A train_args
declare -A test_args

train_args=( ["CartPole-v0"]=200 ["Acrobot-v1"]=-150 ["CartPole-v1"]=450 ["LunarLander-v2"]=200 )

test_args=( ["CartPole-v0"]="200.0CartPole-v0.h5" ["CartPole-v1"]="500.0CartPole-v1.h5" ["Acrobot-v1"]="acrobot_actor-87.h5" ["LunarLander-v2"]="81.17627923613364lunar_actor.h5" )

if [ $1 == "train" ]; then
    for env in ${environments[@]}
    do
        python3 train.py $env ${train_args[$env]}
    done
fi

if [ $1 == "test" ]; then
    for env in ${environments[@]}
    do
        python3 test.py $env "../models/${test_args[$env]}"
    done
fi