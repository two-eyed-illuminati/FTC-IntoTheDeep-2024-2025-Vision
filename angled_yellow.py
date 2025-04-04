import cv2
import numpy as np

def runPipeline(original_image, llrobot):
    largestContour = np.array([[]])
    llpython = [0,0,0,0,0,0,0,0]

    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HLS)

    #Blur to reduce noisiness
    filtered_image = cv2.GaussianBlur(image, (29, 29), sigmaX=0, sigmaY=0)
    #Filter out everything that's not the same color as the sample
    mask = cv2.inRange(filtered_image, np.array([10, 0, 100]),
                                        np.array([30, 255, 255]))
    filtered_image = cv2.bitwise_and(image, image, mask = mask)
    
    original_edges = cv2.Canny(image=cv2.split(filtered_image)[1], threshold1=70, threshold2=140)
    
    #Make sure there aren't any breaks/gaps in the middle of edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    original_edges = cv2.dilate(original_edges, kernel)

    #Use white to represent areas where we think there is a sample, black now means the edge/border of a sample or no sample
    edges = cv2.bitwise_not(original_edges)
    edges = cv2.bitwise_and(edges, edges, mask = mask)

    #Helps find exactly the coordinates of the samples
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 100
    #Samples are 3.5 x 1.5 inches
    min_ratio = 0.1
    max_ratio = 30.0

    sample_contours = []
    for i in contours:
        if(cv2.contourArea(i) > min_area):
            w_h_rat = cv2.minAreaRect(i)[1][0]/cv2.minAreaRect(i)[1][1]
            if((w_h_rat > min_ratio and w_h_rat < max_ratio) or (w_h_rat > 1/max_ratio and w_h_rat < 1/min_ratio)):
                sample_contours.append(i)
    boxed_sample_contours = []
    for i in sample_contours:
        rect = cv2.minAreaRect(i)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image,[box],0,(0,255,255),2)
    if len(sample_contours) > 0:
        largestContour = max(sample_contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(largestContour)
        llpython = [1,x,y,w,h,9,8,7]

    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_HLS2BGR) 

    return largestContour, image, llpython
