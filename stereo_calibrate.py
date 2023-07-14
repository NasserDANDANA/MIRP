import cv2 as cv
import numpy as np
import os
import glob
from math import *
import matplotlib.pyplot as plt
import h5py
import pandas as pd

image_size = (640, 480)

# GET objectPoints
f = h5py.File("./Data/Surgical_Training_Vids/KT/kinematics/KT.h5")

for key in f.keys():
    # print(key) #Names of the root level object names in HDF5 file - can be groups or datasets.
    # print(type(f[key])) # get the object type: usually group or dataset

    if "B004" in key:
        var = f[key].values()
        for v in var:
            if "kinematics" in v.name:
                # print(v[()])

                kine_z_l = v[0:50, 41]
                kine_z_r = v[0:50, 60]

                kine_l = v[0:50 , [39, 40, 41]]
                kine_r = v[0:50 , [58, 59, 60]]

                kine_l_1Hz = v[::30, [39, 40, 41]]
                kine_r_1Hz = v[::30, [58, 59, 60]]

                all_kine_l = v[100::, [39, 40, 41]]
                all_kine_r = v[100::, [58, 59, 60]]


                kine_l = v[:, [39, 40, 41]]
                kine_r = v[:, [58, 59, 60]]


object_points_1Hz = np.append(kine_l, kine_r, axis=0)         
# object_points_1Hz = np.append(kine_l_1Hz, kine_r_1Hz, axis=0)
# print(object_points_1Hz.shape)
# object_points = np.append(kine_l, kine_r, axis=0)
# object_points_np = object_points.astype(np.float32)
# # object_points = [object_points_np]

all_object_points = np.append(all_kine_l, all_kine_r, axis=0)


# GET imagesPoints1
# csv_file =  "./Data/Surgical_Training_Vids/KT/Knot_Tying_B004 - Copie.xlsx"
csv_file =  "./Data/Surgical_Training_Vids/KT/Knot_Tying_B004.xlsx"
images_points1 = pd.read_excel(csv_file, usecols='A:D', header=None).to_numpy(dtype=np.float32)
# images_points1_l = images_points1[:, [0, 1]]
# images_points1_r = images_points1[:, [2, 3]]

images_points1_l_1Hz = images_points1[:, [0, 1]]
images_points1_r_1Hz = images_points1[:, [2, 3]]

images_points1_1Hz = np.append(images_points1_l_1Hz, images_points1_r_1Hz, axis=0)
images_points1_1Hz = images_points1_1Hz.astype(np.float32)

# print(images_points1_1Hz)


# images_points1 = np.append(images_points1_l, images_points1_r, axis=0)
# images_points1 = images_points1.astype(np.float32)
# # images_points1 = [images_points1]

# # GET imagesPoints2
# images_points2 = pd.read_excel(csv_file, usecols='E:H', header=None).to_numpy(dtype=np.float32)
# images_points2_l = images_points2[:, [0, 1]]
# images_points2_r = images_points2[:, [2, 3]]

# images_points2 = np.stack((images_points2_l, images_points2_r), axis=1)





# def get_equals_z(array):

#     index1 = 0
#     index2 = 0

#     index1_list = []
#     index2_list = []
    

#     dict = {}

#     count_per_index = 0

#     for value in array:

#         index2 = index1

#         count_per_index = 0
        
#         for i in range(index1, len(array)):

#             if index1 == 0 and abs(value - array[i]) <= 0.0015:

#                 index2_list.append(i)

#                 count_per_index += 1

#             index2 +=1

#         if(count_per_index > 0):
#             dict[index1] = count_per_index

#         index1 += 1


#     dict = sorted(dict.items(), key=lambda x:x[1], reverse=True)

#     # print(index2_list)
#     # print("#######")
#     # print(dict)
#     # print("#######")


#     # OBJECT POINTS and IMAGE POINTS
#     object_pts = []
#     image_pts1 = []
#     image_pts2 = []

#     for index in index2_list:

#         object_pts.append(kine_l[index, :])
#         image_pts1.append(images_points1_l[index, :])
#         image_pts2.append(images_points2_l[index, :])

    
    
#     object_pts = np.array(object_pts)
#     image_pts1 = np.array(image_pts1)
#     image_pts2 = np.array(image_pts2)

#     # print(image_pts1.shape)
#     # print(image_pts1)
#     # print(image_pts2.shape)
#     # print(image_pts2)

#     object_pts = [object_pts]
#     image_pts1 = [image_pts1]
#     image_pts2 = [image_pts2]
#     return object_pts, image_pts1, image_pts2

# # object_points, images_points1, images_points2 = get_equals_z(kine_z_l)




# # camera_matrix1 = cv.initCameraMatrix2D(object_points, imagePoints=images_points1, imageSize=image_size)
# # print(camera_matrix1)

# matrix = np.zeros((3,3), dtype=float)
# matrix[0, 2] = 320
# matrix[1, 2] = 240
# matrix[0,0] = 368
# matrix[1,1] = 368
# matrix[2, 2] = 1
# print(matrix)

# dist_coeff = np.array([0,0,0,0])
# retval1, camera_matrix1, dist_coeff1, rvecs1, tvecs1 = cv.calibrateCamera(object_points, imagePoints=images_points1, imageSize=image_size,flags=cv.CALIB_USE_INTRINSIC_GUESS^cv.CALIB_USE_EXTRINSIC_GUESS^ cv.CALIB_FIX_PRINCIPAL_POINT, cameraMatrix=matrix, distCoeffs=dist_coeff)
# cam1 = cv.calibrateCamera(object_points, imagePoints=images_points1, imageSize=image_size,flags=cv.CALIB_USE_INTRINSIC_GUESS^cv.CALIB_USE_EXTRINSIC_GUESS^ cv.CALIB_FIX_PRINCIPAL_POINT, cameraMatrix=matrix, distCoeffs=dist_coeff)

# print("OUTPUTS OF CALIBRATE FUNCTION (CAMERA 1) :")
# for output in cam1:

#     print(output)


# retval2, camera_matrix2, dist_coeff2, rvecs2, tvecs2 = cv.calibrateCamera(object_points, imagePoints=images_points2, imageSize=image_size,flags=cv.CALIB_USE_INTRINSIC_GUESS^cv.CALIB_USE_EXTRINSIC_GUESS^cv.CALIB_FIX_PRINCIPAL_POINT , cameraMatrix=matrix, distCoeffs=dist_coeff)
# cam2 = cv.calibrateCamera(object_points, imagePoints=images_points2, imageSize=image_size,flags=cv.CALIB_USE_INTRINSIC_GUESS^cv.CALIB_USE_EXTRINSIC_GUESS^cv.CALIB_FIX_PRINCIPAL_POINT , cameraMatrix=matrix, distCoeffs=dist_coeff)
# print("OUTPUTS OF CALIBRATE FUNCTION (CAMERA 2) :")
# for output in cam2:

#     print(output)

# S = cv.stereoCalibrate(objectPoints=object_points, imagePoints1=images_points1, 
#                        imagePoints2=images_points2, imageSize=image_size, cameraMatrix1=camera_matrix1, distCoeffs1=dist_coeff1, 
#                         cameraMatrix2=camera_matrix2, distCoeffs2=dist_coeff2, flags=cv.CALIB_USE_INTRINSIC_GUESS^cv.CALIB_FIX_PRINCIPAL_POINT)



# print("OUPUTS OF STEREO CALIBRATE : ")
# for output in S:

#     print(str(output) + "\n")





































# DISPLAY EACH FRAME TO GET ARMS POSITIONS (x, y)
# def display(path: str) :

#     video = cv.VideoCapture(path)
#     print(path)

#     #Resize the window
#     # cv.namedWindow("Frame", cv.WINDOW_NORMAL)
#     # cv.resizeWindow("Frame", 640, 480)

#     frame_num = 0
    
#     while video.isOpened():

#         ret, frame = video.read()

#         #If the frame cannot be red
#         if not ret:
#             #print("")
#             break
        
#         frame_num += 1
    
#         frame2 = np.squeeze(frame)

#         plt.imshow(frame2)
#         plt.title(str(frame_num))
#         plt.show()

#         # Specify the type
#         # frame = frame.astype(float)
#         # cv.imshow("Frame", frame)

#         # cv.waitKey(5000)

#     video.release()

#     cv.destroyAllWindows()


# #Path where videos are splitted in 3 folders (\KT , \NP and \SU in that case)
# path = r"C:\Users\nassb\OneDrive\Documents\COURS\Project1\Data\Surgical_Training_Vids\SU"

# for vid in os.listdir(path):
#     if "Suturing_B005_capture2" in vid:
#         display(path + '\\' + vid)


