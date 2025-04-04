import cv2
import numpy as np

def runPipeline(original_image, llrobot):
    best_contour = np.array([[]])
    llpython = []

    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb)

    #Blur to reduce noisiness
    filtered_image = cv2.GaussianBlur(image, (7, 7), sigmaX=0, sigmaY=0)
    #Filter out everything that's not the same color as the sample
    mask = cv2.inRange(filtered_image, np.array([27, 130, 70]),
                                        np.array([150, 170, 118]))
    filtered_image = cv2.bitwise_and(filtered_image, filtered_image, mask = mask)
    
    mask_sum = np.sum(mask)
    edge_sum = mask_sum
    threshold = 5
    original_edges = None
    #Account for different lighting conditions requiring different thresholds for edge detection;
    #If a lot of the area we're looking at is detected as edges, then our threshold is probably too low
    while threshold < 40:
        #Detect large brightness changes between pixels; this likely represents the edge/border of a sample
        original_edges = cv2.Canny(image=cv2.split(filtered_image)[0], threshold1=threshold, threshold2=threshold*2)
        edge_sum = np.sum(original_edges)
        threshold += 2
        if edge_sum / mask_sum < 0.13:
            break
    
    #Make sure there aren't any breaks/gaps in the middle of an edge
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if(edge_sum != 0):
        original_edges = cv2.dilate(original_edges, kernel)

    #Use white to represent areas where we think there is a sample, black now means the edge/border of a sample or no sample
    edges = cv2.bitwise_not(original_edges)
    edges = cv2.bitwise_and(edges, edges, mask = mask)

    #Helps find exactly the coordinates of the samples
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 5000
    #Samples are 3.5 x 1.5 inches
    min_ratio = 1.5
    max_ratio = 6.0

    #Filter out all contours which are extraneous
    sample_contours = []
    boxed_sample_contours = []
    for i in contours:
        if(cv2.contourArea(i) > min_area):
            rect = cv2.minAreaRect(i)
            w_h_rat = rect[1][0]/rect[1][1]
            if((w_h_rat > min_ratio and w_h_rat < max_ratio) or (w_h_rat > 1/max_ratio and w_h_rat < 1/min_ratio)):
                sample_contours.append(i)
                boxed_sample_contours.append(rect)

                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(image,[box],0,(0,255,255),2)
    
    #Find closest contour & its angle
    if len(sample_contours) > 0:
        closest_rect_idx, closest_rect = min(enumerate(boxed_sample_contours),
            key=lambda rect: abs(rect[1][0][0]-640/2)+abs(rect[1][0][1]-480/2))
        best_contour = sample_contours[closest_rect_idx]

        angle = None
        w, h = closest_rect[1]
        if h > w:
            angle = closest_rect[2]
        else:
            angle = closest_rect[2] - 90
        image = cv2.putText(image, str(angle)[:5]+" deg", (int(closest_rect[0][0] - 10), int(closest_rect[0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 2)
        llpython = [angle, 640/2 - closest_rect[0][0], 480/2 - closest_rect[0][1]]

    #Convert back to BGR
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    print(image[240][320])
    image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)

    adapt = cv2.adaptiveThreshold(cv2.split(filtered_image)[0],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    adapt = cv2.bitwise_and(adapt, mask)
    adapt = cv2.cvtColor(adapt, cv2.COLOR_GRAY2BGR)
    filtered_darkness = cv2.cvtColor(cv2.split(filtered_image)[0], cv2.COLOR_GRAY2BGR)
    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_YCrCb2BGR)

    return best_contour, edges, llpython
