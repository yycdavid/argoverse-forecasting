import numpy as np
from shapely.geometry import LineString, Point

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
