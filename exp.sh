python lstm_prog_train_test.py --mode split --data_path ../argoverse_data/processed/train_pen_2.pkl

python lstm_prog_reg.py --mode split --data_path ../argoverse_data/processed/val_pen_2.pkl

python lstm_prog_reg.py --mode train --train_path ../argoverse_data/processed/val_pen_05_split.pkl --val_path ../argoverse_data/processed/val_pen_05_split.pkl
