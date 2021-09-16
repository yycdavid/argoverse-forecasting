python lstm_prog_train_test.py --mode split --data_path ../argoverse_data/processed/train_pen_2.pkl

python lstm_prog_reg.py --mode split --data_path ../argoverse_data/processed/val_pen_2.pkl

CUDA_VISIBLE_DEVICES=1 python lstm_prog_reg.py --mode train --train_path ../argoverse_data/processed/val_pen_05_split.pkl --val_path ../argoverse_data/processed/val_pen_05_split.pkl --split --result_dir reg_test

CUDA_VISIBLE_DEVICES=1 python lstm_prog_reg.py --mode test --test_path ../argoverse_data/processed/val_pen_05.pkl --model_file LSTM_step456.pth.tar --split --result_dir reg_test
