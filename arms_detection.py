import cv2 as cv
import numpy as np
import os
import glob
from math import *

def average(path: str) :

    #Get the video name
    video_name = path.split('/')
    video_name = video_name[len(video_name) - 1]

    video = cv.VideoCapture(path)

    #Path where the background will be saved
    path =  "../Data/Backgrounds/" + video_name + "_background.png"

    if(os.path.exists(path)):
        print("Backrgound exists")
        return path

    frames_number = 0
    array = []

    #Resize the window
    cv.namedWindow("Frame", cv.WINDOW_NORMAL)
    cv.resizeWindow("Frame", 640, 480)
    
    while video.isOpened():

        ret, frame = video.read()

        #If the frame cannot be red
        if not ret:
            #print("")
            break

        #Counting frame numbers
        frames_number += 1
        
        #Specify the type
        frame = frame.astype(float)

        #Create the first array
        if frames_number == 1:
            array = np.array(frame)

        #Add arrays
        else :
            array = np.add(array,np.array(frame))

        cv.imshow('Frame', frame)

    print(frames_number)

    #Average of the image
    avg_view = np.divide(array,frames_number)

    #Display the current frame
    cv.imshow('Frame', avg_view)

    #cv.waitKey(25)

    video.release()

    #Save the average as an image
    cv.imwrite(path, avg_view)

    cv.destroyAllWindows()

    return path



def objects_mouvements(path):

    #Get current video name
    video_name = path.split('/')
    video_name = video_name[len(video_name) - 1]

    #Create the directory where images and the video will be saved
    path2 = "../Data/Objects_mouvement/" + video_name

    try:
        os.mkdir(path2)

    #If directory already exists
    except OSError as error: 
        print("") 

    #Directory of images
    path2 += "/Images"

    try:
        os.mkdir(path2)
        
    #If directory already exists
    except OSError as error: 
        print("") 

    #Directory of colored arms images
    path2 += "/ColoredArms"
    try:
        os.mkdir(path2)
        
    #If directory already exists
    except OSError as error: 
        print("") 

    #Get the background of the video
    background_path = average(path)
    back_img = cv.imread(background_path)

    #Guassian blur
    k_blur = 1
    back_img = cv.GaussianBlur(back_img, (k_blur, k_blur), 0)

    height, width, layers = back_img.shape
    size = (width, height)

    #Dimensions of the window
    cv.namedWindow("Frame", cv.WINDOW_NORMAL)
    cv.resizeWindow("Frame", 640, 480)

    #Start getting video
    video = cv.VideoCapture(path)

    frame_count = 0
    k1 = 3
    img_array = []
    img_array2 = []
    
    
    while video.isOpened():

        ret, frame = video.read()

        cv.imshow('Frame', frame)

        #If the frame cannot be red
        if not ret:
            #print("")
            break

        frame_count += 1

        #Copy of the original frame
        frame_cop = frame.copy()

        #Gaussian Blur
        frame = cv.GaussianBlur(frame, (k_blur, k_blur), 0)
        
        #Substraction between the frame and the background
        abs_sub = cv.absdiff(frame, back_img)
        abs_sub = cv.cvtColor(abs_sub, cv.COLOR_BGR2GRAY)

        #Binarization
        th, frame_object = cv.threshold(abs_sub, 50, 255, cv.THRESH_BINARY)

        #Copy of the thresholded image
        frame_copy = frame_object.copy()

        #Fill holes
        mask = np.zeros((height+2, width + 2), np.uint8)
        cv.floodFill(frame_copy, mask, (0,0), 255)

        #Invert image pixels
        inv_frame_copy = cv.bitwise_not(frame_copy)

        #Overlay images
        frame_object = frame_object | inv_frame_copy

        #Closure
        kernel = np.ones((k1, k1), np.uint8)
        frame_object = cv.erode(frame_object, kernel, iterations=10)
        frame_object = cv.dilate(frame_object, kernel, iterations=10)

        frame_object = cv.dilate(frame_object, kernel, iterations=30)
        frame_object = cv.erode(frame_object, kernel, iterations=10)

        kernel2 = np.ones((k1, 3*k1), np.uint8)
        frame_object = cv.dilate(frame_object, kernel2, iterations=5)

        #Colored arms
        frame_object = cv.cvtColor(frame_object, cv.COLOR_GRAY2RGB)
        vis = np.where(frame_object == [255, 255, 255] , frame_cop/255 , [0,0,0])
        
        #Specify the type
        frame_object = frame_object.astype(float)
        vis = vis.astype(float)

        #Display the current frame
        
        
        #Keys listener
        key = cv.waitKey(10)&0xFF
        if key == ord('q'):
            break

        if key == ord('p'):
            k_blur = min(21, k_blur + 2)
            print(k_blur)

        if key == ord('m'):
            k_blur = max(1, k_blur - 2)
            print(k_blur)


        #Save the binary frame
        filename = "../Data/Objects_mouvement/" + video_name + "/Images/" + str(frame_count) + ".jpg"
        cv.imwrite(filename, frame_object)

        #Save the colored arm frame
        filename = "../Data/Objects_mouvement/" + video_name + "/Images/ColoredArms/" + str(frame_count) + ".jpg"
        cv.imwrite(filename, vis*255)

    cv.destroyAllWindows()

    
    #Build video binary
    for file in sorted(glob.glob("../Data/Objects_mouvement/" + video_name + "/Images/*.jpg"), key=os.path.getmtime):
        img = cv.imread(file)
        img_array.append(img)

    path3 = "../Data/Objects_mouvement/" + video_name + "/"
    out = cv.VideoWriter(path3 + "object_mouvement_binary.avi" ,cv.VideoWriter_fourcc(*'DIVX'), 15, size)

    #Put images in the video
    for i in range(len(img_array)):
        out.write(img_array[i])


    #Build video colored arms
    for file in sorted(glob.glob("../Data/Objects_mouvement/" + video_name + "/Images/ColoredArms/*.jpg"), key=os.path.getmtime):
        img = cv.imread(file)
        img_array2.append(img)

    path3 = "../Data/Objects_mouvement/" + video_name + "/"
    out = cv.VideoWriter(path3 + "object_mouvement_colored_arms.avi" ,cv.VideoWriter_fourcc(*'DIVX'), 15, size)

    #Put images in the video
    for i in range(len(img_array2)):
        out.write(img_array2[i])

    video.release()
    out.release()


objects_mouvements('..\Data\Videos/video_1.avi')

