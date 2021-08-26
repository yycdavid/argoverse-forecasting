#!/bin/bash

segment=$1
step=$2
python lstm_prog_train_test.py --total_segments ${segment} --use_delta --test --model_path saved_models/lstm_prog_${segment}segment_train/LSTM_step${step}.pth.tar
