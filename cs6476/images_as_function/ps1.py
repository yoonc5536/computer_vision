import math
import numpy as np
import cv2
import sys
from numpy import inf
# # Implement the functions below.


def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    temp_image = np.copy(image)
    return temp_image[:,:,2]


def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """
    temp_image = np.copy(image)
    return temp_image[:,:,1]

def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """
    temp_image = np.copy(image)
    return temp_image[:,:,0]

def swap_green_blue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """

    temp_image = np.copy(image)
    # for every pixel, it swaps the blue data with the green data
    for i in range(0,len(temp_image)):
        for j in range(0,len(temp_image[0])):
            temp = temp_image[i][j][0]
            temp_image[i][j][0] = temp_image[i][j][1]
            temp_image[i][j][1] = temp
    return temp_image


def copy_paste_middle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    temp_src = np.copy(src)
    temp_dst = np.copy(dst)

    src_row = len(temp_src)/2 - shape[0]/2
    src_col = len(temp_src[0])/2 - shape[1]/2

    dst_row = len(temp_dst)/2 - shape[0]/2
    dst_col = len(temp_dst[0])/2 - shape[1]/2
    for i in range(0,shape[0]):
        for j in range(0,shape[1]):
            temp_dst[dst_row+i][dst_col+j] = temp_src[src_row+i][src_col+j]
    return temp_dst


def image_stats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """
    debug_flag = 0
    min_val = inf
    max_val = -inf
    sum_val = 0
    width = len(image)
    height = len(image[0])
    total_pixels = width*height
    for i in range(0,width):
        for j in range(0,height):
            val = image[i][j]
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
            sum_val += val
    avg_val = sum_val/float(total_pixels)

    # calculate the stddev
    diff_sum = 0
    for i in range(0,width):
        for j in range(0,height):
            diff_sum += (image[i,j] - avg_val)**2
    stddev = (diff_sum/float(total_pixels))**0.5

    # convert variable to float as specified in the requirement above.
    min_val = float(min_val)
    max_val = float(max_val)
    avg_val = float(avg_val)
    stddev = float(stddev)
    if debug_flag:
        print("#### IMAGE_STATS")
        print(min_val, max_val, sum_val,total_pixels,avg_val)  
        print(type(min_val),type(max_val),type(avg_val),type(stddev))
    return (min_val,max_val,avg_val,stddev)

def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """
    
    stats = image_stats(image)
    stddev = stats[3]
    mean = stats[2]
    temp_image = np.copy(image)
    temp_image = temp_image.astype(dtype=np.float64)
    for i in range(0,len(temp_image)):
        for j in range(0,len(temp_image[0])):
            temp_image[i][j] = ((temp_image[i][j] - mean)/stddev) * scale
    #return temp_image.astype(dtype=np.uint8)
    return temp_image

def shift_image_left(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """
    if shift == 0:
        return image
    
    temp_image = np.copy(image)
    
    row = len(temp_image)
    col = len(temp_image[0])
    for i in range(0,row):
        temp_image[i][0:col-shift] = temp_image[i][shift:col]
        temp_image[i][col-shift:col] = [image[i][col-1]]*shift
    return temp_image

def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """
    temp_img = np.zeros(img1.shape, np.float64)
    min_val = inf
    max_val = -inf
    for i in range(0,len(temp_img)):
        for j in range(0,len(temp_img[0])):
            diff = int(img1[i][j]) - int(img2[i][j])
            if diff < min_val:
                min_val = diff
            if diff > max_val:
                max_val = diff
            temp_img[i][j] = diff 
            #print(temp_img[i][j])
    max_diff = max_val - min_val
    if max_diff == 0:
        max_diff = 1
    #print(max_diff,max_val,min_val)
    for i in range(0,len(temp_img)):
        for j in range(0,len(temp_img[0])):
            temp_img[i][j] = 255 * ((temp_img[i][j] - min_val)/float(max_diff))
           # print(temp_img[i][j])
    #temp_img = temp_img.astype(dtype=np.uint8)
    #print(temp_img)
    return temp_img

def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """
    temp_image = np.copy(image)
    temp_image = temp_image.astype(dtype=np.float64)
    row = temp_image.shape[0]
    col = temp_image.shape[1]
    noise = np.random.randn(row, col) * sigma
    noisy_image = np.copy(temp_image)
    noisy_image[:,:,channel] = temp_image[:,:,channel] + noise
    return noisy_image
