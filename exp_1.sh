
CUDA_VISIBLE_DEVICES=0 python lstm_prog_reg.py --mode train --train_path ../argoverse_data/processed/train_pen_2_split.pkl --val_path ../argoverse_data/processed/val_pen_2_split.pkl --test_path ../argoverse_data/processed/val_pen_2.pkl --split --lr 0.0005 --t_ratio 0.05 --v_ratio 10.0 --result_dir reg_lr_5e4_t005_v10

CUDA_VISIBLE_DEVICES=1 python lstm_prog_reg.py --mode train --train_path ../argoverse_data/processed/train_pen_2_split.pkl --val_path ../argoverse_data/processed/val_pen_2_split.pkl --test_path ../argoverse_data/processed/val_pen_2.pkl --split --lr 0.0001 --t_ratio 0.05 --v_ratio 10.0 --result_dir reg_lr_1e4_t005_v10

CUDA_VISIBLE_DEVICES=2 python lstm_prog_reg.py --mode train --train_path ../argoverse_data/processed/train_pen_2_split.pkl --val_path ../argoverse_data/processed/val_pen_2_split.pkl --test_path ../argoverse_data/processed/val_pen_2.pkl --split --lr 0.0005 --t_ratio 0.1 --v_ratio 10.0 --result_dir reg_lr_5e4_t01_v10

CUDA_VISIBLE_DEVICES=3 python lstm_prog_reg.py --mode train --train_path ../argoverse_data/processed/train_pen_2_split.pkl --val_path ../argoverse_data/processed/val_pen_2_split.pkl --test_path ../argoverse_data/processed/val_pen_2.pkl --split --lr 0.0005 --t_ratio 0.05 --v_ratio 20.0 --result_dir reg_lr_5e4_t005_v20

CUDA_VISIBLE_DEVICES=2 python lstm_prog_reg.py --mode train --train_path ../argoverse_data/processed/train_pen_2_split.pkl --val_path ../argoverse_data/processed/val_pen_2_split.pkl --test_path ../argoverse_data/processed/val_pen_2.pkl --split --lr 0.0005 --t_ratio 0.02 --v_ratio 10.0 --result_dir reg_lr_5e4_t002_v10
