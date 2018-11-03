"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np

# enable this flag to pass Bonnie
traffic_light_bonnie = False

bonnies_test = False
debug = False
plot = True
error_centroid = (-1,-1)

def printConfig():
    if bonnies_test: 
        print('bonnie_test is enabled')
    else:
        print('bonnie_test is disabled')

    if plot:
        print('plot is enabled')
    else:
        print('plot is disabled')
    if debug:
        print('debug is enable')
    else:
        print('debug is disabled')

if bonnies_test:
    warning_hough_params = {'rho':1,'theta':1,'threshold':5,'min_length':20,'min_gap':10}
    yield_hough_params = {'rho':1,'theta':6,'threshold':1,'min_length':1,'min_gap':1}
    stop_hough_params = {'rho':1,'theta':1,'threshold':1,'min_length':1,'min_gap':1}
    # DNE sign params are in its function
    # Traffic light circle parameters
    traffic_light_hough_params = {'rho':1,'theta':1,'threshold':10,'min_length':1,'min_gap':1}
    radii_range = range(15,30,1)
    traffic_light_circle_hough_params = {'radii_range':radii_range,'param1':40,'param2':2}
else:
    traffic_light_hough_params = {'rho':1,'theta':1,'threshold':1,'min_length':1,'min_gap':1}

    # color masking does not work well here due to that the sun has the same color the warning sign
    # To fix: tune the parameters -> circles have shorter line length
    # this passes the local test file and works fine for part3
    # For part 4: warning sign is much hard to detect as the noise makes the sun detectable in this case. 
    warning_hough_params = {'rho':1,'theta':1,'threshold':35,'min_length':15,'min_gap':0}

    # color masking can indentify 3 objects: traffic light (red circle), do not enter sign, and yield sign. 
    # To fix: tuine the parameters: first two objects are circle and their straight lines are much shorted than the yield sign
    # increase the min_length
    yield_hough_params = {'rho':1,'theta':1,'threshold':20,'min_length':70,'min_gap':20}

    # stop sign does not need tuning as the color is unique as compare to other red objects
    stop_hough_params = {'rho':1,'theta':6,'threshold':1,'min_length':1,'min_gap':1} 

    # using hough circle for multiple signs on the image
    # tune the radii range to the size of DNE sign sshould make it work
    # increase param2 -> less circles
    # NOTE: now it may detect stop sign as circle, but it is only taking the first cicle as the centroid
    # DNE sign should be the first circle that it should have get more votes in the houghCicle proceddure.

    # TODO: this method is very dependent on the size of the DNE sign on the image.
    radii_range = range(15,40,1)
    dne_hough_params = {'radii_range':radii_range,'param1':40,'param2':15}

    # Traffic light circle parameters
    radii_range = range(15,30,1)
    traffic_light_circle_hough_params = {'radii_range':radii_range,'param1':40,'param2':2}

printConfig()

def getTrafficLightState(img_in,radii_range):
    # Convert to grayscale
    img_in_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blue to reduce the noise and false positive rate
    img_in_gaussian_blur = cv2.GaussianBlur(img_in_gray, (9,9), 0)

    # Hough circle
    coordinates = [0]
    param2_val = 15
    while len(coordinates) != 3 and param2_val > 0:
        circles = cv2.HoughCircles(img_in_gaussian_blur, cv2.cv.CV_HOUGH_GRADIENT,1, max(radii_range), 
            param1=25,param2=param2_val,
            minRadius=min(radii_range),
            maxRadius=max(radii_range))

        circles = np.uint16(np.around(circles))
        coordinates = []
        sun_luma_threshold = 200
        # (i[0],i[1]) -> (col,row)
        for i in circles[0,:]:
            x,y = i[0],i[1]
            if img_in_gray[y][x] > sun_luma_threshold:
                radius = i[2]
                red_zone_coord = img_in[y-radius*2][x]
                if red_zone_coord[0] != 0 or red_zone_coord[1] != 0:
                    continue
            coordinates.append([x,y])
        param2_val -= 1

    #inorder of red to green light
    coordinates.sort(key=lambda x: x[1])
    state = "Unknown"
    for x,y in coordinates:
        bgr = img_in[y][x]
        if 255 in bgr:
            if bgr[1] == 255 and bgr[2] == 255:
                state = 'yellow'
            elif bgr[1] == 255:
                state = 'green'
            else:
                state = 'red'
    return tuple(coordinates[1]),state

def traffic_light_detection(img_in, radii_range, enable_circle_method=True, force_circle_method_off=False):

    # do not care about state here
    # plot images
    image_name = 'traffic_light/traffic_light'
    method = 'segment'
    # HSV color masking
    hsv,mask,masked_img = colorMask(img_in,color_mask_single='gray')
    # gaussian filter
    img_in_gaussian_blur = cv2.GaussianBlur(masked_img, (9,9), 0)
    # Canny Edge Detection:
    threshold1 = 150
    threshold2 = 350
    apertureSize = 5
    img_in_canny = cv2.Canny(img_in_gaussian_blur, threshold1 = threshold1, threshold2 = threshold2, apertureSize = apertureSize)
    lines = houghLines(img_in_canny,traffic_light_hough_params)
    
    if (lines is None and enable_circle_method) and force_circle_method_off is False:
        method = 'circle'
#        hsv,mask,masked_img = colorMask(img_in,color_mask_single='warning')
        img_in_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
        img_in_gaussian_blur = cv2.GaussianBlur(img_in_gray, (9,9), 0)
        lines = houghLines(img_in_gaussian_blur,traffic_light_circle_hough_params,method='circle')

    if plot:
        cv2.imwrite("./{}_masked_img.png".format(image_name), masked_img)
        cv2.imwrite("./{}_mask.png".format(image_name), mask)
        cv2.imwrite("./{}_canny.png".format(image_name), img_in_canny)
    drawHoughLines(img_in,lines,image_name=image_name,method=method)

    coord,state = getTrafficLightState(img_in,radii_range)

    if traffic_light_bonnie:
        return coord,state

    if method == 'segment':
        return findCentroid(img_in, lines, 'traffic_light_segment', image_name),state  
    else:
        return findCentroid(img_in, lines, 'traffic_light_circle', image_name),state

def yield_sign_detection(img_in):

  # plot images
    image_name = 'yield/yield_sign'
    # HSV color masking
    hsv,mask,masked_img = colorMask(img_in, color_mask_single = 'yield')

    # gaussian filter
    img_in_gaussian_blur = cv2.GaussianBlur(masked_img, (9,9), 0)

    # Canny Edge Detection:
    threshold1 = 150
    threshold2 = 350
    apertureSize = 5
    img_in_canny = cv2.Canny(img_in_gaussian_blur, threshold1 = threshold1, threshold2 = threshold2, apertureSize = apertureSize)

    lines = houghLines(img_in_canny,yield_hough_params)

    if plot:
        cv2.imwrite("./{}_masked_img.png".format(image_name), masked_img)
        cv2.imwrite("./{}_mask.png".format(image_name), mask)
        cv2.imwrite("./{}_canny.png".format(image_name), img_in_canny)

    drawHoughLines(img_in,lines,image_name = image_name)
    return findCentroid(img_in, lines, 'yield', image_name)

def stop_sign_detection(img_in):

    # plot images
    image_name = 'stop/stop_sign'

    # HSV color masking
    hsv,mask,masked_img = colorMask(img_in, color_mask_single = 'stop')

    # gaussian filter
    img_in_gaussian_blur = cv2.GaussianBlur(masked_img, (9,9), 0)

    # Canny Edge Detection:
    threshold1 = 150
    threshold2 = 350
    apertureSize = 5
    img_in_canny = cv2.Canny(img_in_gaussian_blur, threshold1 = threshold1, threshold2 = threshold2, apertureSize = apertureSize)

    # Hough space lines
    lines = houghLines(img_in_canny,stop_hough_params)

    if plot:
        cv2.imwrite("./{}_masked_img.png".format(image_name), masked_img)
        cv2.imwrite("./{}_mask.png".format(image_name), mask)
        cv2.imwrite("./{}_canny.png".format(image_name), img_in_canny)
    drawHoughLines(img_in,lines,image_name=image_name)
    return findCentroid(img_in, lines, 'stop', image_name)

def warning_sign_detection(img_in):

    # plot images
    image_name = 'warning/warning_sign'

    # HSV color masking
    hsv,mask,masked_img = colorMask(img_in, color_mask_single = 'warning')

    # gaussian filter
    img_in_gaussian_blur = cv2.GaussianBlur(masked_img, (9,9), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (3, 3))
    img_erode = cv2.dilate(masked_img, kernel, iterations=6)
    # Canny Edge Detection:
    threshold1 = 150
    threshold2 = 350
    apertureSize = 5
    img_in_gaussian_blur = cv2.GaussianBlur(img_erode, (9,9), 0)

    img_in_canny = cv2.Canny(img_in_gaussian_blur, threshold1 = threshold1, threshold2 = threshold2, apertureSize = apertureSize)

    # Hough space lines
    lines = houghLines(img_in_canny,warning_hough_params)

    if plot:
        cv2.imwrite("./{}_masked_img.png".format(image_name), masked_img)
        cv2.imwrite("./{}_mask.png".format(image_name), mask)
        cv2.imwrite("./{}_canny.png".format(image_name), img_in_canny)
        cv2.imwrite("./{}_gaussian_blur.png".format(image_name), img_in_gaussian_blur)
        cv2.imwrite("./{}_erode.png".format(image_name), img_erode)
    drawHoughLines(img_in,lines,image_name=image_name)

    return findCentroid(img_in, lines, 'warning', image_name)

def construction_sign_detection(img_in):

    # plot images
    image_name = 'construction/construction_sign'

    # HSV color masking
    hsv,mask,masked_img = colorMask(img_in, color_mask_single = 'construct')

    # gaussian filter
    img_in_gaussian_blur = cv2.GaussianBlur(masked_img, (9,9), 0)

    # Canny Edge Detection:
    threshold1 = 150
    threshold2 = 350
    apertureSize = 5
    img_in_canny = cv2.Canny(img_in_gaussian_blur, threshold1 = threshold1, threshold2 = threshold2, apertureSize = apertureSize)

    # Hough space lines
    hough_params = {'rho':1,'theta':1,'threshold':5,'min_length':20,'min_gap':10}
    lines = houghLines(mask,hough_params)

    if plot:
        cv2.imwrite("./{}_masked_img.png".format(image_name), masked_img)
        cv2.imwrite("./{}_mask.png".format(image_name), mask)
        cv2.imwrite("./{}_canny.png".format(image_name), img_in_canny)
    drawHoughLines(img_in,lines,image_name=image_name)

    return findCentroid(img_in, lines, 'construction', image_name)

def do_not_enter_sign_detection(img_in):

    # plot images
    image_name = 'dne/dne_sign'

    # HSV color masking
    hsv,mask,masked_img = colorMask(img_in, color_mask_single = 'dne')

    # Hough space
    if bonnies_test:
        circle_hough = False
    else:
        circle_hough = True

    if circle_hough:
        method = 'circle'
        img_in_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

        img_in_gaussian_blur = cv2.GaussianBlur(img_in_gray, (9,9), 0)
        lines = houghLines(img_in_gaussian_blur,dne_hough_params,method = method)
    else:
        # gaussian filter
        img_in_gaussian_blur = cv2.GaussianBlur(masked_img, (9,9), 0)

        # Canny Edge Detection:
        threshold1 = 150
        threshold2 = 350
        apertureSize = 5
        img_in_canny = cv2.Canny(img_in_gaussian_blur, threshold1 = threshold1, threshold2 = threshold2, apertureSize = apertureSize)
        hough_params = {'rho':1,'theta':1,'threshold':5,'min_length':1,'min_gap':1}
        method = 'segment'
        lines = houghLines(img_in_canny,hough_params)
    
    if plot:
        cv2.imwrite("./{}_masked_img.png".format(image_name), masked_img)
        cv2.imwrite("./{}_mask.png".format(image_name), mask)
        if circle_hough is False:
            cv2.imwrite("./{}_canny.png".format(image_name), img_in_canny)
    drawHoughLines(img_in,lines,image_name=image_name, method=method)
    return findCentroid(img_in, lines, 'dne', image_name)

def traffic_sign_detection(img_in):

    ret_dict = {}
    error_coord = (-1,-1)
    
    yield_coord = yield_sign_detection(img_in)
    stop_coord = stop_sign_detection(img_in)
    warning_coord = warning_sign_detection(img_in)
    construction_coord = construction_sign_detection(img_in)
    no_entry_coord = do_not_enter_sign_detection(img_in)
    traffic_light_coord,state = traffic_light_detection(img_in,range(10,30,1), force_circle_method_off=True)
    dict_all_centroid = {'no_entry':no_entry_coord,'stop':stop_coord,'yield':yield_coord,'warning':warning_coord,'construction':construction_coord,'traffic_light':traffic_light_coord}

    print('INFO: def traffic_sign_detections: dict_all_centroid ->',dict_all_centroid)
    for sign in dict_all_centroid:
        centroid = dict_all_centroid[sign]
        if centroid != error_coord:
            ret_dict[sign] = centroid
    print('INFO: def traffic_sign_detections: ret_dict ->',ret_dict) 
    return ret_dict

def traffic_sign_detection_noisy(img_in):

    img_filter = cv2.medianBlur(img_in,3)

    ret_dict = {}
    error_coord = (-1,-1)
    
    yield_coord = yield_sign_detection(img_filter)
    stop_coord = stop_sign_detection(img_filter)
    warning_coord = warning_sign_detection(img_filter)
    construction_coord = construction_sign_detection(img_filter)
    no_entry_coord = do_not_enter_sign_detection(img_filter)
    traffic_light_coord,state = traffic_light_detection(img_filter,range(15,30,1))
    dict_all_centroid = {'no_entry':no_entry_coord,'stop':stop_coord,'yield':yield_coord,'warning':warning_coord,'construction':construction_coord,'traffic_light':traffic_light_coord}

    print('INFO: def traffic_sign_detections_noisy: dict_all_centroid ->',dict_all_centroid)
    for sign in dict_all_centroid:
        centroid = dict_all_centroid[sign]
        if centroid != error_coord:
            ret_dict[sign] = centroid
    print('INFO: def traffic_sign_detections_noisy: ret_dict ->',ret_dict) 
    return ret_dict

def traffic_sign_detection_challenge(img_in):

    ret_dict = {}
    error_coord = (-1,-1)
    
    yield_coord = yield_sign_detection(img_in)
    stop_coord = stop_sign_detection(img_in)
    warning_coord = warning_sign_detection(img_in)
    construction_coord = construction_sign_detection(img_in)
    no_entry_coord = do_not_enter_sign_detection(img_in)
    traffic_light_coord,state = traffic_light_detection(img_in,range(10,30,1), force_circle_method_off=True)
    dict_all_centroid = {'no_entry':no_entry_coord,'stop':stop_coord,'yield':yield_coord,'warning':warning_coord,'construction':construction_coord,'traffic_light':traffic_light_coord}

    print('INFO: def traffic_sign_detections: dict_all_centroid ->',dict_all_centroid)
    for sign in dict_all_centroid:
        centroid = dict_all_centroid[sign]
        if centroid != error_coord:
            ret_dict[sign] = centroid
    print('INFO: def traffic_sign_detections: ret_dict ->',ret_dict) 
    return ret_dict


# Additonal helper functions
def findCentroid(img_in, lines, sign, image_name):
    '''
    Each sign needs to modify for different problem. So I seperated them from Bonnie submission to keep it clean
    '''
    if sign == 'warning':
        if bonnies_test:
            return (findSigns(img_in,lines,image_name))
        else:
            return (findWarning(img_in,lines,image_name))
    elif sign == 'construction':
        if bonnies_test:
            return (findSigns(img_in,lines,image_name))
        else:
            return (findConstruction(img_in,lines,image_name))
    elif sign == 'stop':
        if bonnies_test:
            return (findSigns(img_in,lines,image_name))
        else:
            return (findStop(img_in,lines,image_name))
    elif sign == 'traffic_light_segment':
        return (findTrafficLightsSegment(img_in,lines,image_name))
    elif sign == 'traffic_light_circle':
        return (findTrafficLightsCircle(img_in,lines,image_name))
    elif sign == 'dne':
        if bonnies_test:
            return (findSigns(img_in,lines,image_name))
        else:
            return (findDoNotEnter(img_in,lines,image_name))
    elif sign == 'yield':
        if bonnies_test:
            return (findYieldBonnie(img_in,lines,image_name))
        else:
            return (findYield(img_in,lines,image_name))
    else:
        print("ERROR: Do not have valid sign name to find the centroid.")
        return error_centroid

def findTrafficLightsSegment(img_in,houghLines, image_name):
    img_copy = np.copy(img_in)
    data_pts = []
    centroid = (-1,-1)
    if houghLines is None:
        return centroid

    for line in houghLines[0]:
        data_pts.append([line[0],line[1]])
        data_pts.append([line[2],line[3]])

    left_upper_vert = (min(map(lambda x: x[0], data_pts)),min(map(lambda x: x[1], data_pts)))
    left_lower_vert = (min(map(lambda x: x[0], data_pts)),max(map(lambda x: x[1], data_pts)))
    right_upper_vert = (max(map(lambda x: x[0], data_pts)),min(map(lambda x: x[1], data_pts)))
    right_lower_vert = (max(map(lambda x: x[0], data_pts)),max(map(lambda x: x[1], data_pts)))

    all_verts = [left_lower_vert,left_upper_vert,right_upper_vert,right_lower_vert]

    centroid = ((right_lower_vert[0]-left_lower_vert[0])/2+left_lower_vert[0], (left_lower_vert[1]-left_upper_vert[1])/2+left_upper_vert[1])
    all_verts.append(centroid)
    if plot:
        for vert in all_verts:
            cv2.circle(img_copy, vert, 1, (0,0,0), 2)
        cv2.imwrite("./{}_centroid.png".format(image_name), img_copy)
    return centroid

def findTrafficLightsCircle(img_in,circles,image_name):
    if circles is None:
        return (-1,-1)
    centroid = []
    img_copy = np.copy(img_in)
    for i in circles[0]:
        x,y = i[0],i[1]
        centroid.append((x,y))

    if plot:        
        for x,y,r in circles[0]:
            cv2.circle(img_copy, (x, y), 2, (0,0,0), 2)
            #cv2.circle(img_copy, (x, y), r, (0,0,0), 2)
            break
        print('plot:{}_centroid.png'.format(image_name))
        cv2.imwrite("./{}_centroid.png".format(image_name), img_copy)
    return centroid[0]

def findSigns(img_in, houghLines, image_name):
    img_copy = np.copy(img_in)
    data_pts = []
    centroid = (-1,-1)
    if houghLines is None:
        return centroid

    for line in houghLines[0]:
        data_pts.append([line[0],line[1]])
        data_pts.append([line[2],line[3]])
    left_upper_vert = (min(map(lambda x: x[0], data_pts)),min(map(lambda x: x[1], data_pts)))
    left_lower_vert = (min(map(lambda x: x[0], data_pts)),max(map(lambda x: x[1], data_pts)))
    right_upper_vert = (max(map(lambda x: x[0], data_pts)),min(map(lambda x: x[1], data_pts)))
    right_lower_vert = (max(map(lambda x: x[0], data_pts)),max(map(lambda x: x[1], data_pts)))

    all_verts = [left_lower_vert,left_upper_vert,right_upper_vert,right_lower_vert]

    centroid = ((right_lower_vert[0]-left_lower_vert[0])/2+left_lower_vert[0], (left_lower_vert[1]-left_upper_vert[1])/2+left_upper_vert[1])
    all_verts.append(centroid)
    if plot:
        for vert in all_verts:
            cv2.circle(img_copy, vert, 1, (0,0,0), 2)
        cv2.imwrite("./{}_centroid.png".format(image_name), img_copy)
    return centroid

def findYieldBonnie(img_in,houghLines,image_name):
    img = np.copy(img_in)
    data_pts = []
    centroid = (-1,-1)
    if houghLines is None:
        return centroid
    for line in houghLines[0]:
        data_pts.append([line[0],line[1]])
        data_pts.append([line[2],line[3]])
    all_verts = []
    top_right_vert = (max(map(lambda x: x[0], data_pts)),min(map(lambda x: x[1], data_pts)))
    top_left_vert = (min(map(lambda x: x[0], data_pts)),min(map(lambda x: x[1], data_pts)))
    diff = (max(map(lambda x: x[0], data_pts)) - min(map(lambda x: x[0], data_pts)))/2
    bottom_vert = (min(map(lambda x: x[0], data_pts))+diff,max(map(lambda x: x[1], data_pts)))
    x1,x2,x3 = top_left_vert[0],top_right_vert[0],bottom_vert[0]
    y1,y2,y3 = top_left_vert[1],top_right_vert[1],bottom_vert[1]

    # check the angle to rule out false positives
    angle = computeAngle(top_right_vert,top_left_vert,bottom_vert)
    if angle > 65 or angle < 55:
        return centroid

    centroid = ((x1+x2+x3)/3, (y1+y2+y3)/3)
    all_verts = [top_left_vert,top_right_vert,bottom_vert,centroid]

    if plot:
        for vert in all_verts:
            cv2.circle(img, vert, 1, (0,0,0), 2)
        cv2.imwrite("./{}_centroid.png".format(image_name), img)
    return centroid

def findYield(img_in,houghLines,image_name):
    img = np.copy(img_in)
    data_pts = []
    centroid = (-1,-1)
    if houghLines is None:
        return centroid

    for line in houghLines[0]:
        data_pts.append([line[0],line[1]])
        data_pts.append([line[2],line[3]])
    all_verts = []
    top_right_vert = (max(map(lambda x: x[0], data_pts)),min(map(lambda x: x[1], data_pts)))
    top_left_vert = (min(map(lambda x: x[0], data_pts)),min(map(lambda x: x[1], data_pts)))
    diff = (max(map(lambda x: x[0], data_pts)) - min(map(lambda x: x[0], data_pts)))/2
    bottom_vert = (min(map(lambda x: x[0], data_pts))+diff,max(map(lambda x: x[1], data_pts)))
    x1,x2,x3 = top_left_vert[0],top_right_vert[0],bottom_vert[0]
    y1,y2,y3 = top_left_vert[1],top_right_vert[1],bottom_vert[1]

    # check the angle to rule out false positives
    angle = computeAngle(top_right_vert,top_left_vert,bottom_vert)
    if angle > 65 or angle < 55:
        return centroid

    centroid = ((x1+x2+x3)/3, (y1+y2+y3)/3)
    all_verts = [top_left_vert,top_right_vert,bottom_vert,centroid]
    # verify that the vertices are on the sign
    '''
    yield_bgr_color = [0,0,255]
    for vert in all_verts:
        dot = img_in[vert[0]][vert[1]]
        if dot[0] != yield_bgr_color[0]:
            return (-1,-1)
        if dot[1] != yield_bgr_color[1]:
            return (-1,-1)
        if dot[2] != yield_bgr_color[2]:
            return (-1,-1)
    '''
    if plot:
        for vert in all_verts:
            cv2.circle(img, vert, 1, (0,0,0), 2)
        print('plot:{}_centroid.png'.format(image_name))
        cv2.imwrite("./{}_centroid.png".format(image_name), img)
    return centroid

def findDoNotEnter(img_in,circles,image_name):
    if circles is None:
        return (-1,-1)
    if len(circles[0]) > 1:
        print('INFO: def findDoNotEnter: found multiple circles. Taking the first center as the dne sign.')
    elif len(circles[0]) == 0:
        return (-1,-1)    

    centroid = []
    img_copy = np.copy(img_in)
    for i in circles[0]:
        x,y = i[0],i[1]
        centroid.append((x,y))

    if plot:        
        for x,y,r in circles[0]:
            cv2.circle(img_copy, (x, y), 2, (0,0,0), 2)
            #cv2.circle(img_copy, (x, y), r, (0,0,0), 2)
            break
        print('plot:{}_centroid.png'.format(image_name))
        cv2.imwrite("./{}_centroid.png".format(image_name), img_copy)
    return centroid[0]

def findConstruction(img_in,houghLines,image_name):
    img_copy = np.copy(img_in)
    data_pts = []
    centroid = (-1,-1)
    if houghLines is None:
        return centroid

    for line in houghLines[0]:
        data_pts.append([line[0],line[1]])
        data_pts.append([line[2],line[3]])
    left_upper_vert = (min(map(lambda x: x[0], data_pts)),min(map(lambda x: x[1], data_pts)))
    left_lower_vert = (min(map(lambda x: x[0], data_pts)),max(map(lambda x: x[1], data_pts)))
    right_upper_vert = (max(map(lambda x: x[0], data_pts)),min(map(lambda x: x[1], data_pts)))
    right_lower_vert = (max(map(lambda x: x[0], data_pts)),max(map(lambda x: x[1], data_pts)))

    all_verts = [left_lower_vert,left_upper_vert,right_upper_vert,right_lower_vert]

    centroid = ((right_lower_vert[0]-left_lower_vert[0])/2+left_lower_vert[0], (left_lower_vert[1]-left_upper_vert[1])/2+left_upper_vert[1])
    all_verts.append(centroid)
    if plot:
        for vert in all_verts:
            cv2.circle(img_copy, vert, 1, (0,0,0), 2)
        print('plot:{}_centroid.png'.format(image_name))
        cv2.imwrite("./{}_centroid.png".format(image_name), img_copy)
    return centroid

def findWarning(img_in,houghLines,image_name):
    img_copy = np.copy(img_in)
    data_pts = []
    centroid = (-1,-1)
    if houghLines is None:
        return centroid

    for line in houghLines[0]:
        data_pts.append([line[0],line[1]])
        data_pts.append([line[2],line[3]])
    left_upper_vert = (min(map(lambda x: x[0], data_pts)),min(map(lambda x: x[1], data_pts)))
    left_lower_vert = (min(map(lambda x: x[0], data_pts)),max(map(lambda x: x[1], data_pts)))
    right_upper_vert = (max(map(lambda x: x[0], data_pts)),min(map(lambda x: x[1], data_pts)))
    right_lower_vert = (max(map(lambda x: x[0], data_pts)),max(map(lambda x: x[1], data_pts)))

    all_verts = [left_lower_vert,left_upper_vert,right_upper_vert,right_lower_vert]

    centroid = ((right_lower_vert[0]-left_lower_vert[0])/2+left_lower_vert[0], (left_lower_vert[1]-left_upper_vert[1])/2+left_upper_vert[1])
    all_verts.append(centroid)
    if plot:
        for vert in all_verts:
            cv2.circle(img_copy, vert, 1, (0,0,0), 2)
        print('Plot centroid:',image_name)
        cv2.imwrite("./{}_centroid.png".format(image_name), img_copy)
    return centroid

def findStop(img_in,houghLines,image_name):
    img_copy = np.copy(img_in)
    data_pts = []
    centroid = (-1,-1)
    if houghLines is None:
        return centroid    

    for line in houghLines[0]:
        data_pts.append([line[0],line[1]])
        data_pts.append([line[2],line[3]])
    left_upper_vert = (min(map(lambda x: x[0], data_pts)),min(map(lambda x: x[1], data_pts)))
    left_lower_vert = (min(map(lambda x: x[0], data_pts)),max(map(lambda x: x[1], data_pts)))
    right_upper_vert = (max(map(lambda x: x[0], data_pts)),min(map(lambda x: x[1], data_pts)))
    right_lower_vert = (max(map(lambda x: x[0], data_pts)),max(map(lambda x: x[1], data_pts)))

    all_verts = [left_lower_vert,left_upper_vert,right_upper_vert,right_lower_vert]

    centroid = ((right_lower_vert[0]-left_lower_vert[0])/2+left_lower_vert[0], (left_lower_vert[1]-left_upper_vert[1])/2+left_upper_vert[1])
    all_verts.append(centroid)
    if plot:
        for vert in all_verts:
            cv2.circle(img_copy, vert, 1, (0,0,0), 2)
        print('plot:{}_centroid.png'.format(image_name))
        cv2.imwrite("./{}_centroid.png".format(image_name), img_copy)
    return centroid

def drawHoughLines(img_in, lines, method = 'segment', image_name = 'hough_line', color = [0,0,0], thickness = 2 ):
    if lines is None:
        return -1

    if debug:
        print("Debug: hough lines are:",lines)
        print("Debug: number of hough lines are:",len(lines[0]))
    img_copy = np.copy(img_in)
    if method == 'segment':
        for line in lines[0]:
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]
            cv2.line(img_copy,(x1,y1),(x2,y2),color,thickness)
    elif method == 'circle':
        for x,y,r in lines[0]:
            cv2.circle(img_copy, (x, y), 2, color, thickness)
            cv2.circle(img_copy, (x, y), r, color, thickness)       
    else:
        for line in lines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(img_copy,(x1,y1),(x2,y2),color,thickness) 

    if plot:
        cv2.imwrite("./{}_hough_line.png".format(image_name), img_copy)

def houghLines(img_in, params, method = 'segment'):
    if method == 'segment':
        expected_params = ['rho', 'theta', 'threshold', 'min_length', 'min_gap']
        if len(expected_params) != len(params.keys()):
            raise ValueError("def houghLines: the length of parameters is not expected.")
        for param in expected_params:
            if param not in params:
                raise ValueError("def houghLines: parameter",param,"is not found in the argument.")
        lines = cv2.HoughLinesP(img_in, rho = params['rho'], theta = params['theta']*np.pi/180, threshold = params['threshold'], minLineLength = params['min_length'], maxLineGap = params['min_gap'])
    elif method == 'circle':
        radii_range = params['radii_range']
        param1 = params['param1']
        param2 = params['param2']

        circles = cv2.HoughCircles(img_in, cv2.cv.CV_HOUGH_GRADIENT,1, max(radii_range), 
            param1=param1,param2=param2,
            minRadius=min(radii_range),
            maxRadius=max(radii_range))
        if circles is not None:
            circles = np.uint16(np.around(circles))
        return circles
    else:
        # not used
        lines = cv2.HoughLines(img_in, rho=params['rho'], theta=params['theta']*np.pi/180, threshold=params['threshold'])
    return lines

def colorMask(img_in, color_mask_lower = False, color_mask_upper = False, color_mask_single = False):

    color_map = {'yield':[0,255,255],'stop':[0,255,204],'dne':[0,255,255],'warning':[30,255,255],'construct':[15,255,255],'black':[0,0,0],'gray':[0,0,51]}

    img_copy = np.copy(img_in)
    if not color_mask_single and not color_mask_upper and not color_mask_single:
        raise ValueError("def colorMask: the color must be specified.")

    if color_mask_single is not False:
        if color_mask_single in color_map:
            color_arg = color_map[color_mask_single]
        else:
            color_arg = color_mask_single
        lower_color = np.array(color_arg)
        upper_color = np.array(color_arg)
    else:
        lower_color = np.array(color_mask_lower)
        upper_color = np.array(color_mask_upper)

    if debug:
        print("Debug: lower_color",lower_color," Upper_color",upper_color)
    hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    masked_img = cv2.bitwise_and(img_in,img_in, mask= mask)

    return (hsv,mask,masked_img)


def computeAngle(c0, c1, c2):
    d0 = np.array(c0) - np.array(c1)
    d1 = np.array(c2) - np.array(c1)
    return np.degrees(np.math.atan2(np.linalg.det([d0,d1]),np.dot(d0,d1)))