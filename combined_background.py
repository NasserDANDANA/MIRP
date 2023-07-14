import cv2 as cv
import numpy as np
import os
from math import *

def average_combined(path: str):

    #Fill out 2 arrays (1 for left view and 1 for right view)
    videos_array1 = []
    videos_array2 = []

    for video in os.listdir(path):

        #To have the view of the video then
        split_path = video.split('capture')

        if split_path[1] == "1.avi":
            videos_array1.append(video)
        else :
            videos_array2.append(video)
    
    average(videos_array1, path, "capture1")
    average(videos_array2, path, "capture2")
    

def average(videos_array, path, view: str):

    #Directory where the background is  saved
    path_back_dir = path + "\\" + view
    try:
        os.mkdir(path_back_dir)

    #If directory already exists
    except OSError as error: 
        print(error) 

    #Path where the background will be saved
    path_back = path_back_dir + "/background.jpg"
    
    if(os.path.exists(path_back)):
        print("Background exists")
        return

    frames_number = 0
    array = []

    #All video of a side view
    for video in videos_array:

        video = cv.VideoCapture(path + '/' +video)

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

            print(np.array(frame)[0,0])

            #Create the first array
            if frames_number == 1:
                array = np.array(frame)

            #Add arrays
            else :
                array = np.add(array,np.array(frame))

            #Keys listener
            key = cv.waitKey(10)&0xFF
            if key == ord('q'):
                break

            #Display the current frame
            cv.imshow('Frame', frame)

            cv.waitKey(5)


    #Average of the image
    avg_view = np.divide(array,frames_number)

    video.release()

    #Save the average as an image
    cv.imwrite(path_back, avg_view)

    cv.destroyAllWindows()


#Path where videos are splitted in 3 folders (\KT , \NP and \SU in that case)
path = r"C:\Users\nassb\OneDrive\Documents\COURS\Project1\Data\Surgical_Training_Vids"

for dir in os.listdir(path):
    average_combined(path + '\\' + dir)


