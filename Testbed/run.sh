#!/bin/bash

environments=( "SpecialCartPole-v0" "SpecialMountainCar-v0" "SpecialAcrobot-v1" "SpecialPendulum-v1" )

declare -A train_args
declare -A test_args

train_args=( ["SpecialCartPole-v0"]=195.0 ["SpecialMountainCar-v0"]=-110.0 ["SpecialAcrobot-v1"]=-100 ["SpecialPendulum-v1"] )

test_args=( ["SpecialCartPole-v0"]="" ["CartPole-v1"]="" ["Acrobot-v1"]="" ["LunarLander-v2"]="" )

if [ $1 == "train" ]; then
    for env in ${environments[@]}
    do
        python3 Project/train.py $env ${train_args[$env]}
    done
fi

if [ $1 == "test" ]; then
    for env in ${environments[@]}
    do
        python3 Project/test.py $env "Models/${test_args[$env]}"
    done
fi