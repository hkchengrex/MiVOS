import sys
import os

import numpy as np
import numpy.polynomial.polynomial as poly
import yaml

from argparse import ArgumentParser
from progressbar import progressbar

from gen_utils import *

"""
Hyperparameters
"""
n_frames = 160
res_x = 256*3
res_y = 256*2
cam_min_fov = 0.7
cam_max_fov = 0.8

degree = 4

min_num_obj = 3
max_num_obj = 5

# Small and big objects
obj_min_scale = 1.25
obj_max_scale = 3.0
min_scale_change = 0.8
max_scale_change = 1.25

# Not strictly followed as these are used to generate the control points only
obj_max_tsl_per_frame = 0.03
obj_max_rot_per_frame = 0.02
cam_max_tsl_per_frame = 0.10
cam_look_at_max_shift = 0.10

enter_static_prob = 0.10
conti_static_prob = 0.35

min_num_lights = 1
max_num_lights = 3
light_min_str = 1000
light_max_str = 15000

"""
These define the bounding volume of objects
It is larger than the viewing frustum
"""

# Camera position
cam_min_base = -0.3
cam_max_base = 0.3
cam_min_into = -6
cam_max_into = -5

# "look at" target for the camera
lka_min_base = -0.5
lka_max_base = 0.5
lka_min_into = 3
lka_max_into = 4

# Object positions
obj_min_base = -1.05
obj_max_base = 1.05
obj_min_into = -2
obj_max_into = 5

light_radius = 11
light_degree = 0
light_base_color = [0.5, 0.5, 0.5]
light_mod_color = 0.5

sky_light_min = 0.1
sky_light_max = 1.0

# texture replacement
mapped_replace_prob = 0.5
unmapped_replace_prob = 1.0


def get_default_dict(text_id):
    # Put some static information
    d = dict()
    d['version'] = 2
    d['setup'] = {
        "blender_install_path": "/home/<env:USER>/blender/",
        "pip": []
    }
    d['global'] = {
        "all": {"output_dir": "<args:2>/%s" % text_id}
    }
    d['modules'] = [
        {"module": "main.Initializer", "config": {}},
    ]

    return d

def add_renderer(d):
    # Add renderers and writer
    d['modules'].append({
        "module": "renderer.SimRgbRenderer",
            "config": {
                "resolution_x": res_x,
                "resolution_y": res_y,
                "output_key": "colors"
            }
    })
    d['modules'].append({
        "module": "renderer.SegMapPngRenderer",
            "config": {
                "resolution_x": res_x,
                "resolution_y": res_y,
                "map_by": "instance"
            }
    })
    d['modules'].append({
        "module": "writer.RGBSegWriter",
            "config": {}
    })

    return d

def get_cam_runner():
    # Default stuff
    d = {"module": "camera.CameraTrajectoryRunner",
        "config": {
            "intrinsics": {
            "resolution_x": res_x,
            "resolution_y": res_y,
            "fov": pick_rand(cam_min_fov, cam_max_fov),
            },}
        }
    max_tsl_dist = cam_max_tsl_per_frame * n_frames / degree
    max_ltl_dist = cam_look_at_max_shift * n_frames / degree

    # Create control points
    loc_pts = np.zeros((degree+1, 3))
    lok_pts = np.zeros((degree+1, 3))
    loc_pts[0] = get_vector_in_block(cam_min_base, cam_max_base, cam_min_into, cam_max_into)
    lok_pts[0] = get_vector_in_block(lka_min_base, lka_max_base, lka_min_into, lka_max_into)
    is_static = np.random.rand() < enter_static_prob
    for i in range(1, degree+1):
        if is_static:
            loc_pts[i] = loc_pts[i-1]
            lok_pts[i] = lok_pts[i-1]
            if np.random.rand() > conti_static_prob: # Inverted
                is_static = False
        else:
            this_tsl_dist = pick_normal_rand(0, max_tsl_dist, 3)
            this_ltl_dist = pick_normal_rand(0, max_ltl_dist, 3)
            loc_pts[i] = get_next_vector_in_block(loc_pts[i-1], this_tsl_dist, cam_min_base, cam_max_base, cam_min_into, cam_max_into)
            lok_pts[i] = get_next_vector_in_block(lok_pts[i-1], this_ltl_dist, lka_min_base, lka_max_base, lka_min_into, lka_max_into)
            if np.random.rand() < enter_static_prob:
                is_static = True

    # Create polynomial
    Xs = np.array([i/degree for i in range(degree+1)])
    loc_poly = poly.polyfit(Xs, loc_pts, deg=degree).astype(float).tolist()
    lok_poly = poly.polyfit(Xs, lok_pts, deg=degree).astype(float).tolist()

    d['config']['cam_poses'] = {
        'location_poly': loc_poly,
        'look_at_poly': lok_poly,
    }

    return d

def get_object_runner():
    # Pick object model
    obj_name = obj_sampler.next()

    # Is our model texture mapped?
    mapped = False
    try:
        with open(os.path.join(obj_root, obj_name, 'models', 'model_normalized.mtl'), 'r') as f:
            for line in f:
                if 'map_Kd' in line:
                    mapped = True
                    break
    except FileNotFoundError:
        pass

    if mapped:
        replace = np.random.rand() < mapped_replace_prob
    else:
        replace = np.random.rand() < unmapped_replace_prob

    # Default stuff
    d = {"module": "object.ObjectTrajectoryRunner",
        "config": {
            'path': "<args:0>/%s/models/model_normalized.obj" % obj_name,
            'seed': pick_randint(0, 2**31)},
        }

    if replace:
        texture = tex_sampler.next()
        d['config']['texture'] = "<args:1>/%s" % texture

    max_tsl_dist = obj_max_tsl_per_frame * n_frames / degree
    max_rot_dist = obj_max_rot_per_frame * n_frames / degree

    # Create control points
    loc_pts = np.zeros((degree+1, 3))
    rot_pts = np.zeros((degree+1, 3))
    scl_pts = np.zeros((degree+1, 3))
    loc_pts[0] = get_vector_in_frustum(obj_min_base, obj_max_base, obj_min_into, obj_max_into, cam_min_into)
    rot_pts[0] = np.random.rand(3) * np.pi * 2
    scl_pts[0] = pick_rand(obj_min_scale, obj_max_scale, 3)
    is_static = np.random.rand() < enter_static_prob
    for i in range(1, degree+1):
        if is_static:
            loc_pts[i] = loc_pts[i-1]
            rot_pts[i] = rot_pts[i-1]
            scl_pts[i] = scl_pts[i-1]
            if np.random.rand() > conti_static_prob: # Inverted
                is_static = False
        else:
            this_tsl_dist = pick_normal_rand(0, max_tsl_dist, 3)
            loc_pts[i] = get_next_vector_in_frustum(loc_pts[i-1], this_tsl_dist, obj_min_base, obj_max_base, obj_min_into, obj_max_into, cam_min_into)
            rot_pts[i] = rot_pts[i-1] + pick_normal_rand(0, max_rot_dist, 3)
            scl_pts[i] = scl_pts[i-1] * pick_rand(min_scale_change**(1/degree), max_scale_change**(1/degree), 3)
            if np.random.rand() < enter_static_prob:
                is_static = True

    # Create polynomial
    Xs = np.array([i/degree for i in range(degree+1)])
    loc_poly = poly.polyfit(Xs, loc_pts, deg=degree).astype(float).tolist()
    rot_poly = poly.polyfit(Xs, rot_pts, deg=degree).astype(float).tolist()
    scl_poly = poly.polyfit(Xs, scl_pts, deg=degree).astype(float).tolist()

    d['config']['poses'] = {
        'location_poly': loc_poly,
        'rotation_poly': rot_poly,
        'scale_poly': scl_poly,
    }

    return d, loc_poly

def get_light_runner(light_name):
    # Default stuff
    d = {"module": "lighting.LightTrajectoryRunner",
        "config": {
            'name': light_name,
            'light':{
                'type': 'POINT',
                'energy': pick_randint(light_min_str, light_max_str),}
            }
        }
    degree = light_degree

    color = light_base_color + pick_rand(0, light_mod_color, 3)
    color = color / color.max()
    d['config']['light']['color'] = color.astype(float).tolist()

    # Create control points
    loc_pts = np.zeros((degree+1, 3))
    rot_pts = np.zeros((degree+1, 3))
    loc_pts[0] = get_vector_on_sphere(light_radius)
    rot_pts[0] = np.random.rand(3) * 3.14 * 2
    for i in range(1, degree+1):
        loc_pts[i] = get_vector_on_sphere(light_radius)
        rot_pts[i] = np.random.rand(3) * 3.14 * 2

    # Create polynomial
    if degree == 0:
        Xs = np.array([0])
    else:
        Xs = np.array([i/degree for i in range(degree+1)])
    loc_poly = poly.polyfit(Xs, loc_pts, deg=degree).astype(float).tolist()
    rot_poly = poly.polyfit(Xs, rot_pts, deg=degree).astype(float).tolist()

    d['config']['poses'] = {
        'location_poly': loc_poly,
        'rotation_poly': rot_poly,
    }

    return d


def get_skybox():
    sky = sky_sampler.next()
    d = {"module": "loader.SkyBoxLoader",
        "config": {
            'path': "<args:1>/%s" % sky,
            'location': [0, 12, 0],
            'rotation': [np.pi/2, 0, 0],
            'strength': pick_rand(sky_light_min, sky_light_max),
        }
    }

    return d

def gen_yaml(id):
    text_id = '%s%05d' % (args.prefix, id)
    yaml_file = os.path.join(out_root, '%s.yaml' % text_id)
    d = get_default_dict(text_id)

    ## Setup VOSTrajRunner
    traj_runner = {
        'module': "composite.VOSTrajRunner",
        'config': {
            "n_frames": n_frames,
        }
    }
    # Setup CamRunner
    camera_runner = get_cam_runner()
    traj_runner['config']['camera_runner'] = camera_runner

    # Setup a list of ObjectRunner
    n_objects = pick_randint(min_num_obj, max_num_obj)
    object_runners = [None] * n_objects
    prev_paths = []
    for i in range(n_objects):
        # Try until the paths are not close to each other
        object_runners[i], path = get_object_runner()
        num_trials = 0
        while (not test_path(prev_paths, path)) and num_trials < 50:
            num_trials += 1
            obj_sampler.step_back()
            object_runners[i], path = get_object_runner()
        prev_paths.append(path)
    traj_runner['config']['object_runners'] = object_runners

    # Setup a list of LightRunner
    n_lights = pick_randint(min_num_lights, max_num_lights)
    light_runners = [None] * n_lights
    for i in range(n_lights):
        light_runners[i] = get_light_runner('light_%02d'%i)
    traj_runner['config']['light_runners'] = light_runners

    # Write to external
    d['modules'].append(traj_runner)
    d['modules'].append(get_skybox())

    d = add_renderer(d) # Add at last as ordering matters
    with open(yaml_file, 'w') as f:
        yaml.dump(d, f, default_flow_style=True)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--models')
    parser.add_argument('--texture')
    parser.add_argument('--prefix', default='')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('output')
    args = parser.parse_args()

    obj_root = args.models
    obj_cls_list = os.listdir(obj_root)
    obj_list = []
    for obj_cls in obj_cls_list:
        if '.json' not in obj_cls:
            obj_list.extend([os.path.join(obj_cls, c) for c in os.listdir(os.path.join(obj_root, obj_cls))])
    obj_list = [o for o in obj_list if os.path.exists(os.path.join(obj_root, o, 'models', 'model_normalized.obj'))]

    sky_root = os.path.join(args.texture, 'skybox')
    sky_cls_list = os.listdir(sky_root)
    sky_list = []
    for sky_cls in sky_cls_list:
        sky_list.extend([os.path.join('skybox', sky_cls, c) for c in os.listdir(os.path.join(sky_root, sky_cls))])

    tex_root = os.path.join(args.texture, 'texture')
    tex_cls_list = os.listdir(tex_root)
    tex_list = []
    for tex_cls in tex_cls_list:
        tex_list.extend([os.path.join('texture', tex_cls, c) for c in os.listdir(os.path.join(tex_root, tex_cls))])

    # Permutation + running over it is better than random selection as the latter creates a binomial distribution kinda
    obj_sampler = Sampler(obj_list)
    sky_sampler = Sampler(sky_list)
    tex_sampler = Sampler(tex_list)

    print('Number of objects: ', len(obj_list))
    print('Number of sky box textures: ', len(sky_list))
    print('Number of normal textures: ', len(tex_list))

    out_root = args.output
    os.makedirs(out_root, exist_ok=True)

    num_samples = int(args.num_samples)

    for i in progressbar(range(num_samples)):
        gen_yaml(i)
