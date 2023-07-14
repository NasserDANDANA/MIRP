import urllib
import bz2
import os
import numpy as np
import matplotlib.pyplot as plt
import math

from stereo_calibrate import *


camera_params = np.zeros((13, ), dtype=np.float32)
camera_params[0] = 0
camera_params[1] = 0
camera_params[2] = 0
camera_params[3] = 0
camera_params[4] = 0
camera_params[5] = 0
camera_params[6] = 360
camera_params[7] = 360
camera_params[8] = 0
camera_params[9] = 0
camera_params[10] = 0
camera_params[11] = 0
camera_params[12] = 0

# print(camera_params)

image_size = np.array([640, 480])


center = image_size/2

points_3d = object_points_1Hz
# print(points_3d.shape)

camera_indices = np.zeros((50, ), dtype=np.int32)

point_indices = np.arange(len(points_3d), dtype=np.int32)
# print(point_indices)

points_2d = images_points1_1Hz
# print(points_2d.shape)

n_cameras = 1
n_points = 61


def get_rotate3_3(rot_vecs):
    matrix = np.zeros((3,3), dtype=np.float32)

    ca = cos(rot_vecs[0])
    cb = cos(rot_vecs[1])
    cy = cos(rot_vecs[2])

    sa = sin(rot_vecs[0])
    sb = sin(rot_vecs[1])
    sy = sin(rot_vecs[2])


    matrix[0, 0] = ca*cb
    matrix[0, 1] = (ca*sb*sy) - (sa*cy)
    matrix[0, 2] = (ca*sb*cy) + (sa*sy)

    matrix[1, 0] = sa*cb
    matrix[1, 1] = (sa*sb*sy) + (ca*cy)
    matrix[1, 2] = (sa*sb*cy) - (ca*sy)
    
    matrix[2, 0] = -sb
    matrix[2, 1] = cb*sy
    matrix[2, 2] = cb*cy

    return matrix


def transform (args):
    rotation_matrix, translation_matrix, points = args

    transformed_points = np.dot(rotation_matrix, points) + translation_matrix
    
    # transformed_points = np.map(np.dot, [(rotation_matrix, point) for point in points_3d]) + camera_params[:, 3:6]

    return transformed_points


def project(translated_points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    # points_proj = rotate(points, camera_params[:, :3])
    # points_proj += camera_params[:, 3:6]
    
    points_proj = -translated_points[:, :2] / translated_points[:, 2, np.newaxis]
    fx = camera_params[0, 6]
    fy = camera_params[0, 7]
    k1 = camera_params[0, 8] 
    k2 = camera_params[0, 9]
    k3 = camera_params[0, 10]
    p1 = camera_params[0, 11]
    p2 = camera_params[0, 12]
    r_pow2 = np.sum(points_proj**2-center**2, axis=1)
    n = 1 + k1 * r_pow2 + k2 * r_pow2**2 + k3 * r_pow2**4
    points_cal = np.zeros(points_proj.shape)
    
    for index, point in enumerate(points_proj):

        # Normalized coord
        point[0] = point[0] / fx 
        point[1] = point[1] / fy

        diff_x = point[0] - center[0]
        diff_y = point[1] - center[1] 

        # points_cal[index, 0] = (point[0] ) * fx
        # points_cal[index, 1] = (point[1] ) * fy

        points_cal[index, 0] = point[0] +  diff_x * n[index] + p1 * (r_pow2[index] + 2 * (diff_x**2) ) + 2 * p2 * diff_x * diff_y
        points_cal[index, 1] = point[1] +  diff_y * n[index] + 2 * p1 * diff_x * diff_y + p2 * (r_pow2[index] + 2 * (diff_y**2))

        # points_cal[index, 0] = points_cal[index, 0]  * fx
        # points_cal[index, 1] = points_cal[index, 1]  * fy

        
    # points_proj *= (r * f)[:, np.newaxis]
    # print("p", points_proj.shape)
    return points_cal


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, points_3D):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 13].reshape((n_cameras, 13))
    rot_mats = np.asarray(list(map(get_rotate3_3, camera_params[:, :3])))
    transformed_points = np.asarray(list(map(transform, [(rot_mats[0], camera_params[0, 3:6], points_3D[i]) for i in point_indices])))

    # print("##", transformed_points.shape)
    points_proj = project(transformed_points, camera_params[camera_indices])
    # diffs = np.sqrt(np.square(points_proj[:, 0] - points_2d[:, 0]) + np.square(points_proj[:, 1] - points_2d[:, 1]))

    diffs = points_proj - points_2d
    return np.sum(diffs **2)

from scipy.sparse import lil_matrix
def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = len(point_indices) * 2
    num_params = n_cameras * 13
    
    A = lil_matrix((m, num_params), dtype=np.float32)
    # print(A.shape)
    i = np.arange(len(camera_indices))
    for s in range(13):
        A[2 * i, camera_indices * 13 + s] = 1
        A[2 * i + 1, camera_indices * 13 + s] = 1

    # for s in range(3):
    #     A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
    #     A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A


x0 = camera_params
# print(x0.shape)

f0 = fun(x0, n_cameras, n_points,camera_indices, point_indices, points_2d, points_3d)

A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

import time
from scipy.optimize import least_squares, Bounds, differential_evolution
lower_b = [-2*math.pi, -2*math.pi, -2*math.pi, -4, -4, -4, 100, 100, -10, -10, -10, -10, -10]
upper_b = [2*math.pi, 2*math.pi, 2*math.pi, 4, 4, 4, 700, 700,  10, 10, 10, 10, 10]
bound = Bounds(lower_b, upper_b, keep_feasible=True)
print(bound)

t0 = time.time()
# jac_sparsity=A, x_scale='jac', verbose=2, x0
res = differential_evolution(fun, bounds=bound,  tol=1e-9, strategy='best2exp', popsize=100,
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d, points_3d), 
                    maxiter=99999, disp=True, polish=True, updating='deferred', x0=x0)
t1 = time.time()

print("Optimization took {0:.0f} seconds".format(t1 - t0))

plt.plot(res.fun)
plt.show()

print(res)
print('Camera params : ', res.x)

# Camera parameters predicted
camera_params_test = np.array(res.x)




rot_matrix = get_rotate3_3(camera_params_test[:3])
trans_matrix = camera_params_test[3:6]

# test_points = np.array([0.02805, -0.00492, -0.131109]) # 1st frame left arm ANSWER = 264, 292
# test_points = np.array([0.028345, -0.004632, -0.065613]) # 20th frame left arm ANSWER = 257, 299
test_points = np.array([0.027741, -0.044416, -0.026495]) # 1st frame right arm ANSWER = 382, 339
# test_points = np.array([0.027633, -0.04505, -0.019887]) # 20th frame right arm ANSWER = 382, 336
# test_points = np.array([0.043791, -0.020398, -0.459207]) # 100th frame left arm ANSWER = 494, 336



args = (rot_matrix, trans_matrix, test_points)


transformed_pts = transform(args).reshape((1,3))

#Transorm 1D array in 2D
# x0 = np.hstack((camera_params_test.ravel()))
# camera_params_test = x0[:1 * 9].reshape((1, 9))
camera_params_test = camera_params_test[None, :]

project_points = project(translated_points=transformed_pts, camera_params=camera_params_test)
print(project_points)