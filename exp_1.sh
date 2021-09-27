CUDA_VISIBLE_DEVICES=0 python lstm_prog_reg.py --mode train --train_path ../argoverse_data/processed/train_pen_2_split.pkl --val_path ../argoverse_data/processed/val_pen_2_split.pkl --test_path ../argoverse_data/processed/val_pen_2.pkl --split --lr 0.0001 --t_ratio 0.05 --v_ratio 4.0 --v_ada --result_dir reg_lr_1e4_t005_v4ada --end_epoch 100

CUDA_VISIBLE_DEVICES=1 python lstm_prog_reg.py --mode train --train_path ../argoverse_data/processed/train_pen_2_split.pkl --val_path ../argoverse_data/processed/val_pen_2_split.pkl --test_path ../argoverse_data/processed/val_pen_2.pkl --split --lr 0.0001 --t_ratio 0.05 --v_ratio 1.0 --v_ada --result_dir reg_lr_1e4_t005_v1ada --end_epoch 100

CUDA_VISIBLE_DEVICES=2 python lstm_prog_reg.py --mode train --train_path ../argoverse_data/processed/train_pen_2_split.pkl --val_path ../argoverse_data/processed/val_pen_2_split.pkl --test_path ../argoverse_data/processed/val_pen_2.pkl --split --lr 0.0002 --t_ratio 0.05 --v_ratio 4.0 --v_ada --result_dir reg_lr_2e4_t005_v4ada --end_epoch 100

CUDA_VISIBLE_DEVICES=3 python lstm_prog_reg.py --mode train --train_path ../argoverse_data/processed/train_pen_2_split.pkl --val_path ../argoverse_data/processed/val_pen_2_split.pkl --test_path ../argoverse_data/processed/val_pen_2.pkl --split --lr 0.0001 --t_ratio 0.05 --v_ratio 7.0 --v_ada --result_dir reg_lr_1e4_t005_v7ada --end_epoch 100
