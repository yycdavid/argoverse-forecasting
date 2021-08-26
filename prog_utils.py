import argparse
import numpy as np
from shapely.geometry import LineString, Point

def parse_arguments():
    """Arguments for running the baseline.

    Returns:
        parsed arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batch_size",
                        type=int,
                        default=1,
                        help="Test batch size")
    parser.add_argument("--model_path",
                        required=False,
                        type=str,
                        help="path to the saved model")
    parser.add_argument("--total_segments",
                        default=1,
                        type=int,
                        help="Number of program segments")
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the trajectories if non-map baseline is used",
    )
    parser.add_argument(
        "--use_delta",
        action="store_true",
        help="Train on the change in position, instead of absolute position",
    )
    parser.add_argument(
        "--train_features",
        default="Traj/forecasting_features_train.pkl",
        type=str,
        help="path to the file which has train features.",
    )
    parser.add_argument(
        "--val_features",
        default="Traj/forecasting_features_val.pkl",
        type=str,
        help="path to the file which has val features.",
    )
    parser.add_argument(
        "--test_features",
        default="Traj/forecasting_features_test.pkl",
        type=str,
        help="path to the file which has test features.",
    )
    parser.add_argument(
        "--joblib_batch_size",
        default=100,
        type=int,
        help="Batch size for parallel computation",
    )
    parser.add_argument("--wandb",
                    action="store_true",
                    help="Use wandb for logging")
    parser.add_argument("--use_map",
                        action="store_true",
                        help="Use the map based features")
    parser.add_argument("--use_social",
                        action="store_true",
                        help="Use social features")
    parser.add_argument("--test",
                        action="store_true",
                        help="If true, only run the inference")
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=512,
                        help="Training batch size")
    parser.add_argument("--val_batch_size",
                        type=int,
                        default=512,
                        help="Val batch size")
    parser.add_argument("--end_epoch",
                        type=int,
                        default=5000,
                        help="Last epoch")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning rate")
    parser.add_argument(
        "--traj_save_path",
        required=False,
        type=str,
        help=
        "path to the pickle file where forecasted trajectories will be saved.",
    )
    return parser.parse_args()

def left_or_right(cl, p, p_proj, d):
    '''
    Left: True
    Right: False
    '''
    epsilon = 1e-2
    p_proj_2 = cl.interpolate(d + epsilon)

    x0 = p_proj.x
    y0 = p_proj.y
    x1 = p_proj_2.x
    y1 = p_proj_2.y
    x2 = p.x
    y2 = p.y
    val = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    return val > 0

def point_by_dist_offset(cl, d, left, offset):
    proj_t = cl.interpolate(d)
    proj_t_delta = cl.interpolate(d + 1)
    x = proj_t_delta.x - proj_t.x
    y = proj_t_delta.y - proj_t.y
    if left:
        offset_dir = np.array([-y, x])
    else:
        offset_dir = np.array([y, -x])
    offset_vec = offset * offset_dir
    fitted_p = proj_t + offset_vec
    return fitted_p

def exec_prog(obs_xy, cl, n_step, v):
    cur_point = obs_xy[-1]
    fitted_xy = []
    cl_string = LineString(cl)
    if n_step == 0:
        return cur_point
    p_start = Point(cur_point)
    d_start = cl_string.project(p_start)
    p_start_proj = cl_string.interpolate(d_start)
    offset_start = p_start.distance(p_start_proj)
    delta_offset = offset_start / n_step
    left = left_or_right(cl_string, p_start, p_start_proj, d_start)

    for idx in range(n_step):
        i = idx + 1
        d = v * i + d_start
        offset = offset_start - delta_offset * i
        fitted_p = point_by_dist_offset(cl_string, d, left, offset)
        fitted_xy.append(fitted_p)
        cur_point = fitted_p
    return fitted_xy
