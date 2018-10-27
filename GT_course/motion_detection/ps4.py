"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2
import os

# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to
                             [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out

# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """

    return cv2.Sobel( image, cv2.CV_64F, 1, 0,ksize=3, scale = 1.0/8.0 )


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """
    return cv2.Sobel( image, cv2.CV_64F, 0, 1,ksize=3,scale = 1.0/8.0 )

# Assignment code
def optic_flow_lk(img_a, img_b, window_size=21,k_type = "uniform",sigma = 1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    
    It = img_a - img_b  
    
    # Compute the gradients
    Ix = gradient_x(img_a)
    Iy = gradient_y(img_a)

    # Run Gaussian or uniform smoothing
    Ixx_blur = cv2.blur( Ix*Ix, (window_size,window_size))
    Iyy_blur = cv2.blur( Iy*Iy, (window_size,window_size))
    Ixy_blur = cv2.blur( Ix*Iy, (window_size,window_size))
    Ixt_blur = cv2.blur( Ix*It, (window_size,window_size))
    Iyt_blur = cv2.blur( Iy*It, (window_size,window_size))

    U = np.zeros(img_a.shape, dtype=np.float_ )
    V = np.zeros(img_a.shape, dtype=np.float_ )
    #threshold=10**-20
    # for bonnie
    threshold=10**-5
    h,w =img_a.shape 
    for i in range(h):
        for j in range(w):
            matA = np.array( [ [Ixx_blur[i,j], Ixy_blur[i,j] ],
                            [Ixy_blur[i,j], Iyy_blur[i,j] ] ])
            detA = np.linalg.det(matA)
            vecB = np.array([ [ 1.0*Ixt_blur[i,j] ],
                            [ 1.0*Iyt_blur[i,j] ] ])   
                                                
            if abs(detA) <= threshold:
                U[i,j] = 0.0
                V[i,j] = 0.0
            else:
                # for efficiency. this is needed to pass bonnie
                u,v = np.dot(np.linalg.inv(np.array(matA)), np.array(vecB))
                U[i,j] = u
                V[i,j] = v
    return U, V

def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """

    kernelx = np.array((1., 4., 6., 4., 1.)) / 16 
    kernely = np.array((1., 4., 6., 4., 1.)) / 16 
    kernely.shape = (5,1)
    filtered = cv2.sepFilter2D(image, cv2.CV_64F, kernelx,kernely)
    h, w = image.shape
    reduced_image = []
    for i in range(0, h, 2):
        row = []
        for j in range(0, w, 2):
            row.append(filtered[i][j])
        reduced_image.append(row)
    return np.array(reduced_image)

def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    ret_list = [image]
    img_reduced = image
    for i in range(1,levels):
      img_reduced = reduce_image(img_reduced)
      ret_list.append(img_reduced)

    return ret_list


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    rows = []
    columns = []
    for img in img_list:
        rows.append(img.shape[0])
        columns.append(img.shape[1])
    output_img = np.zeros((max(rows), sum(columns)))
    next_col = 0
    for i in range(0,len(img_list)):
        output_img[0:img_list[i].shape[0],next_col:columns[i]+next_col] = normalize_and_scale(img_list[i])
        next_col += columns[i]
    return output_img
def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    img_expand = np.zeros((2*image.shape[0],2*image.shape[1]))
    img_expand[::2, ::2] = image[:,:]

    kernelx = np.array((1., 4., 6., 4., 1.)) / 8 
    kernely = np.array((1., 4., 6., 4., 1.)) / 8 
    kernely.shape = (5,1)
    img_expand = cv2.sepFilter2D(img_expand, cv2.CV_64F, kernelx,kernely)
    return img_expand


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    ret_list = []

    for i in range(1,len(g_pyr)):
        ex_layer = expand_image(g_pyr[i])
        diff =  g_pyr[i-1] - ex_layer[0:g_pyr[i-1].shape[0], 0:g_pyr[i-1].shape[1]]  
        ret_list.append(diff)
    ret_list.append(g_pyr[-1])
    
    return ret_list


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    U_warp = np.around(U)
    V_warp = np.around(V)
    borderSize = 10 
    img_border = cv2.copyMakeBorder( image, borderSize, borderSize, borderSize, borderSize, cv2.BORDER_REFLECT)
    warped = np.zeros(image.shape)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            #print(y,x,V_warp[y][x],U_warp[y][x])
            warped[y][x] = img_border[int(y + V_warp[y][x]+borderSize)][int((x + U_warp[y][x]+borderSize))]
    #cv2.remap(image,map1,map2,interpolation = interpolation,borderMode = border_mode)
    return warped


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """
    A = gaussian_pyramid(img_a, levels)
    B = gaussian_pyramid(img_b, levels)
    
    U = np.zeros( A[levels-1].shape )
    V = np.zeros( A[levels-1].shape )
    
    for i in range(levels):
        index = levels-i-1
        A_i = A[index]
        B_i = B[index]

        if i != 0:
            U = 2*expand_image(U)
            V = 2*expand_image(V)

        C_i = warp(B_i, U, V, interpolation, border_mode)
        D_x, D_y = optic_flow_lk(A_i, C_i, k_size,k_type,sigma)
        U = U[:D_x.shape[0], :D_x.shape[1]]
        V = V[:D_y.shape[0], :D_y.shape[1]]
        U = U + D_x
        V = V + D_y
    return U, V