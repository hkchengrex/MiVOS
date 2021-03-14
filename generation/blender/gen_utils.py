import numpy as np
import numpy.polynomial.polynomial as poly
from scipy import optimize


class Sampler:
    def __init__(self, data_list):
        self.data_list = data_list
        self.idx = 0
        self.permute()

    def permute(self):
        self.data_list = np.random.permutation(self.data_list)

    def next(self):
        if self.idx == len(self.data_list):
            self.permute()
            self.idx = 0
        data = self.data_list[self.idx]
        self.idx += 1
        return data

    def step_back(self):
        self.idx -= 1
        if self.idx == -1:
            self.idx = len(self.data_list) - 1

def test_path(prev_paths, path, tol=0.75):
    min_dist = float('inf')
    path = np.array(path)
    for p in prev_paths:
        p = np.array(p)
        # Find min distance as a constrained optimization problem
        poly_vals = p - path
        f = lambda x: np.linalg.norm(poly.polyval(x, poly_vals))
        optim_x = optimize.minimize_scalar(f, bounds=(0, 1), method='bounded')
        if optim_x.fun < tol:
            # print('Fail')
            return False
    # print('Success')
    return True

def pick_rand(min_v, max_v, shape=None):
    if shape is not None:
        return np.random.rand(shape)*(max_v-min_v) + min_v
    else:
        return np.random.rand()*(max_v-min_v) + min_v

def pick_normal_rand(mean, std, shape=None):
    return np.random.normal(mean, std, shape)

def pick_randint(min_v, max_v):
    return np.random.randint(min_v, max_v+1)

def normalize(a):
    return a / np.linalg.norm(a)

def get_2side_rand(max_delta, shape=1):
    return np.random.rand(shape)*2*max_delta-max_delta

def get_vector_in_frustum(min_base, max_base, min_into, max_into, cam_min_into):
    y = pick_rand(min_into, max_into)

    f_min_base = min_base * ((y - cam_min_into) / (min_into - cam_min_into))
    f_max_base = max_base * ((y - cam_min_into) / (min_into - cam_min_into))

    x = pick_rand(f_min_base, f_max_base)
    z = pick_rand(f_min_base, f_max_base)
    return np.array((x, y, z))

def get_vector_in_block(min_base, max_base, min_into, max_into):
    x = pick_rand(min_base, max_base)
    y = pick_rand(min_into, max_into)
    z = pick_rand(min_base, max_base)
    return np.array((x, y, z))

def get_vector_on_sphere(radius):
    x1 = np.random.normal(0, 1)
    x2 = np.random.normal(0, 1)
    x3 = np.random.normal(0, 1)
    norm = (x1*x1 + x2*x2 + x3*x3)**(1/2)
    pt = radius*np.array((x1,x2,x3))/norm
    return pt

def get_next_vector_in_block(curr_vec, max_delta, min_base, max_base, min_into, max_into):
    new_point = get_vector_in_block(min_base, max_base, min_into, max_into)

    max_delta = np.abs(max_delta)
    dist_vec = (new_point - curr_vec)
    for i in range(3):
        if dist_vec[i] > max_delta[i]:
            dist_vec[i] = np.sign(dist_vec[i]) * max_delta[i]

    new_point = curr_vec + dist_vec
    return new_point

def get_next_vector_in_frustum(curr_vec, max_delta, min_base, max_base, min_into, max_into, cam_min_into):
    new_point = get_vector_in_frustum(min_base, max_base, min_into, max_into, cam_min_into)

    max_delta = np.abs(max_delta)
    dist_vec = (new_point - curr_vec)
    for i in range(3):
        if dist_vec[i] > max_delta[i]:
            dist_vec[i] = np.sign(dist_vec[i]) * max_delta[i]

    new_point = curr_vec + dist_vec
    return new_point