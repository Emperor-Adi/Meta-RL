#!/bin/bash

environments=( "CartPole-v0" "CartPole-v1" "Acrobot-v1" )

declare -A train_args
declare -A test_args

train_args=( ["CartPole-v0"]=200 ["Acrobot-v1"]=-90 ["CartPole-v1"]=450 )

test_args=( ["CartPole-v0"]="200.0CartPole-v0.h5" ["CartPole-v1"]="500.0CartPole-v1.h5" ["Acrobot-v1"]="acrobot_actor-87.h5" )

if [ $1 == "train" ]; then
    for env in ${environments[@]}
    do
        python3 train.py $env ${train_args[$env]}
    done
fi

if [ $1 == "test" ]; then
    for env in ${environments[@]}
    do
        python3 test.py $env ${test_args[$env]}
    done
fi