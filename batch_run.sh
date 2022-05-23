#!/bin/sh
nohup python train_gift.py --experiment_name gift_model --embedding_dimension 96 --batch_size 16384 --learning_rate 0.1 > nohup1.out 2>&1 &
sleep 10
nohup python train_gift.py --experiment_name gift_model --embedding_dimension 256 --batch_size 16384 --learning_rate 0.1 > nohup2.out 2>&1 &
sleep 10
nohup python train_gift.py --experiment_name gift_model --embedding_dimension 96 --batch_size 8384 --learning_rate 0.1 > nohup3.out 2>&1 &
sleep 10
nohup python train_gift.py --experiment_name gift_model --embedding_dimension 96 --batch_size 16384 --learning_rate 0.01 > nohup4.out 2>&1 &
sleep 10
nohup python train_gift.py --experiment_name gift_model --embedding_dimension 96 --batch_size 16384 --learning_rate 0.05 > nohup4.out 2>&1 &