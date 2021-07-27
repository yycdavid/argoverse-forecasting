#!/bin/bash

step=$1
python lstm_prog_train_test.py --use_delta --test --model_path lstm_prog_one_segment/LSTM_step${step}.pth.tar
