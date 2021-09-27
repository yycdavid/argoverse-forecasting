python lstm_prog_train_test.py --mode split --data_path ../argoverse_data/processed/train_pen_2.pkl

python lstm_prog_reg.py --mode split --data_path ../argoverse_data/processed/val_pen_2.pkl

CUDA_VISIBLE_DEVICES=1 python lstm_prog_reg.py --mode train --train_path ../argoverse_data/processed/train_pen_2_split.pkl --val_path ../argoverse_data/processed/val_pen_2_split.pkl --test_path ../argoverse_data/processed/val_pen_2.pkl --split --result_dir reg_test

CUDA_VISIBLE_DEVICES=1 python lstm_prog_reg.py --mode test --test_path ../argoverse_data/processed/val_pen_05.pkl --model_file LSTM_stepbest.pth.tar --split --result_dir reg_lr_1e4_t005_v10 --use_gt_time

python lstm_prog_reg.py --mode subset --data_path ../argoverse_data/processed/train_pen_2.pkl

python lstm_prog_reg.py --mode split --data_path ../argoverse_data/processed/train_pen_2_1000.pkl

python lstm_prog_reg.py --mode split --data_path ../argoverse_data/processed/train_pen_2_5000.pkl

python lstm_prog_reg.py --mode split --data_path ../argoverse_data/processed/train_pen_2_10000.pkl

CUDA_VISIBLE_DEVICES=3 python lstm_prog_reg.py --mode train --train_path ../argoverse_data/processed/train_pen_2.pkl --val_path ../argoverse_data/processed/val_pen_2.pkl --lr 0.005 --result_dir base_lr_5e3_test --end_epoch 1000 --use_baseline --model_path results/base_lr_5e3/LSTM_stepbest.pth.tar

CUDA_VISIBLE_DEVICES=3 python lstm_prog_reg.py --mode train --train_path ../argoverse_data/processed/train_pen_2.pkl --val_path ../argoverse_data/processed/val_pen_2.pkl --lr 0.1 --result_dir base_lr_1e1 --end_epoch 1000 --use_baseline

CUDA_VISIBLE_DEVICES=3 python lstm_prog_reg.py --mode train --train_path ../argoverse_data/processed/train_pen_2.pkl --val_path ../argoverse_data/processed/val_pen_2.pkl --lr 0.001 --result_dir base_lr_1e3_cur --end_epoch 1000 --use_baseline
