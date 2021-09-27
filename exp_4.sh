size=10000
CUDA_VISIBLE_DEVICES=2 python lstm_prog_reg.py --mode train --train_path ../argoverse_data/processed/train_pen_2_"$size".pkl --val_path ../argoverse_data/processed/val_pen_2.pkl --lr 0.01 --result_dir base_lr_1e2_"$size" --end_epoch 1000 --use_baseline
