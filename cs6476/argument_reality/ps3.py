"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
    x1,y1 = p0
    x2,y2 = p1
    dist = ((x1-x2)**2 + (y1-y2) **2)**0.5
    return dist


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    row,col = image.shape[:2]
    ret_list = [(0,0),(0,row-1),(col-1,0),(col-1,row-1)]
    return ret_list 

def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    img_copy = np.copy(image)
    w,h = template.shape[:2]
    ret_pts = []

    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(9,9),0)
    img_blur = cv2.medianBlur(img_blur,9)
    
    method = 'harris'
    if method == 'harris':
        # Detector parameters
        # bonnie parameters
        blockSize = 10 
        apertureSize = 3
        k = 0.04
        threshold = 0.10
        # local parameters
        # blockSize = 10 
        # apertureSize = 3 
        # k = 0.04 
        # threshold = 0.2

        # 0.9 passed just circles
        # Detecting corners
        # find Harris corners

        gray = np.float32(img_blur)
        dst = cv2.cornerHarris(gray,blockSize,apertureSize,k)
        ret,dst = cv2.threshold(dst,threshold*dst.max(),255,0)
        kernel = np.ones((5,5),np.uint8)
        dst = cv2.dilate(dst,kernel,iterations=1)
        #cv2.imwrite('./harris_img.png',dst)
        
        data_pts = np.transpose(np.nonzero(dst))
        data_pts = np.float32(data_pts)

        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        ret,label,center=cv2.kmeans(data_pts,4,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        for y,x in center:
            ret_pts.append((int(x),int(y)))

    coord_dist = {} 
    for x,y in ret_pts:
        coord_dist[int(euclidean_distance((x,y),(0,0)))] = (x,y)
    ret_pts = []
    for dist in sorted(coord_dist.keys()):
        ret_pts.append(coord_dist[dist])
    return ret_pts

def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    img_copy = np.copy(image)
    p1,p2,p3,p4 = markers
    color = [0,255,0]
    cv2.line(img_copy,p1,p2,color=color,thickness=thickness)
    cv2.line(img_copy,p1,p3,color=color,thickness=thickness)
    cv2.line(img_copy,p3,p4,color=color,thickness=thickness)
    cv2.line(img_copy,p2,p4,color=color,thickness=thickness)
    return img_copy

def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    true_dst = np.copy(imageB)
    src = np.copy(imageA)
    homography = np.linalg.inv(homography)
    H = homography 

    # create indices of the destination image and linearize them
    h, w = true_dst.shape[:2]
    indy, indx = np.indices((h, w), dtype=np.float32)
    lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])

    # warp the coordinates of src to those of true_dst
    map_ind = H.dot(lin_homg_ind)
    map_x, map_y = map_ind[:-1]/map_ind[-1]  
    map_x = map_x.reshape(h, w).astype(np.float32)
    map_y = map_y.reshape(h, w).astype(np.float32)

    # remap
    cv2.remap(src, map_x, map_y,  cv2.INTER_LINEAR,dst = true_dst, borderMode=cv2.BORDER_TRANSPARENT)
    return true_dst


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    
    sp1,sp2,sp3,sp4 = src_points
    sp1x,sp1y = sp1
    sp2x,sp2y = sp2
    sp3x,sp3y = sp3
    sp4x,sp4y = sp4
    dp1,dp2,dp3,dp4 = dst_points
    dp1x,dp1y = dp1
    dp2x,dp2y = dp2
    dp3x,dp3y = dp3
    dp4x,dp4y = dp4
    A = np.matrix([[sp1x, sp1y, 1, 0, 0, 0, -sp1x*dp1x, -sp1y*dp1x],
                  [0, 0, 0, sp1x, sp1y, 1, -sp1x*dp1y, -sp1y*dp1y], 
                  [sp2x, sp2y, 1, 0, 0, 0, -sp2x*dp2x, -sp2y*dp2x],
                  [0, 0, 0, sp2x, sp2y, 1, -sp2x*dp2y, -sp2y*dp2y], 
                  [sp3x, sp3y, 1, 0, 0, 0, -sp3x*dp3x, -sp3y*dp3x],
                  [0, 0, 0, sp3x, sp3y, 1, -sp3x*dp3y, -sp3y*dp3y],
                  [sp4x, sp4y, 1, 0, 0, 0, -sp4x*dp4x, -sp4y*dp4x],
                  [0, 0, 0, sp4x, sp4y, 1, -sp4x*dp4y, -sp4y*dp4y]])
    
    b_vect = [dp1x, dp1y, dp2x, dp2y, dp3x, dp3y, dp4x, dp4y]
    b = np.asarray(b_vect)
    H, residuals, rank, sv = np.linalg.lstsq(A,b,rcond=-1)
    H = np.append(H,[1])
    H = np.reshape(H,(3,3))
    return H


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None
