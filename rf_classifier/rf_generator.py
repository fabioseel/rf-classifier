import numpy as np

def gabor_kernel(shape=(16, 16), frequency=1/8, theta=np.pi/4, sigma_x=3, sigma_y=3, phase_offset=0, center_offset=(0, 0), factor=1, offset=0):
    """
    Generate a Gabor kernel with the specified shape, center offset, and phase offset.

    Parameters:
        shape (tuple, optional): Desired output shape (height, width) of the Gabor kernel. Default is (16, 16).
        frequency (float, optional): Spatial frequency of the sinusoidal component. Default is 0.2.
        theta (float, optional): Orientation of the filter (in radians). Default is 0.
        sigma_x (float, optional): Standard deviation of the Gaussian envelope in the x-direction. Default is 1.
        sigma_y (float, optional): Standard deviation of the Gaussian envelope in the y-direction. Default is 1.
        phase_offset (float, optional): Phase offset of the Gabor kernel (in radians). Default is 0.
        center_offset (tuple, optional): Center offset of the Gaussian envelope (offset_x, offset_y). Default is (0, 0).
        factor (float, optional): scale the resulting filters values
        offset  (float, optional): added to the resulting filters values

    Returns:
        numpy.ndarray: Gabor kernel with the specified shape, center offset, and phase offset.
    """
    x, y = np.meshgrid(np.linspace(-shape[1]/2, shape[1]/2, shape[1]), np.linspace(-shape[0]/2, shape[0]/2, shape[0]))
    
    # Apply center offset to the coordinates
    x -= center_offset[0]
    y -= center_offset[1]
    
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    
    envelope = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2))
    sinusoid = np.cos(2 * np.pi * frequency * x_theta + phase_offset)
    
    gabor_kernel = envelope * sinusoid
    return gabor_kernel * factor + offset

def center_surround(shape=(16,16), theta =np.pi/4, sigma_x1=2, sigma_y1=2, sigma_x2=4, sigma_y2=4, center_offset=(0, 0), factor=1, offset=0):
    """
    Generate a Gabor kernel with the specified shape, center offset, and phase offset.

    Parameters:
        shape (tuple, optional): Desired output shape (height, width) of the center surround rf. Default is (16, 16).
        theta (float, optional): Orientation of the filter (in radians). Default is 0.
        sigma_x1 (float, optional): Standard deviation of the inner Gaussian in the x-direction. Default is 1.
        sigma_y1 (float, optional): Standard deviation of the inner Gaussian in the y-direction. Default is 1.
        sigma_x2 (float, optional): Standard deviation of the outer Gaussian in the x-direction. Default is 2.
        sigma_y2 (float, optional): Standard deviation of the outer Gaussian in the y-direction. Default is 2.
        center_offset (tuple, optional): Center offset of the Gaussian envelope (offset_x, offset_y). Default is (0, 0).
        factor (float, optional): scale the resulting filters values
        offset  (float, optional): added to the resulting filters values

    Returns:
        numpy.ndarray: Gabor kernel with the specified shape, center offset, and phase offset.
    """

    x, y = np.meshgrid(np.linspace(-shape[1]/2, shape[1]/2, shape[1]), np.linspace(-shape[0]/2, shape[0]/2, shape[0]))
    
    # Apply center offset to the coordinates
    x -= center_offset[0]
    y -= center_offset[1]
    
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gaussian1 = np.exp(-0.5 * (x_theta**2 / sigma_x1**2 + y_theta**2 / sigma_y1**2))
    gaussian2 = np.exp(-0.5 * (x_theta**2 / sigma_x2**2 + y_theta**2 / sigma_y2**2))
    return (2*gaussian1-gaussian2)*factor + offset

def noise(shape=(16,16), low=0, high=1):
    """
    Generate random noise with uniform distribution.

    Parameters:
        shape (tuple): The shape of the noise array.
        low (float): The lower bound of the uniform distribution.
        high (float): The upper bound of the uniform distribution.

    Returns:
        numpy.ndarray: A 2D array of random noise with shape `shape`.
    """
    return np.random.uniform(low=low, high=high,size=shape)

def decorrelated_color_noise(shape=(16,16), low=0, high=1):
    """
    Generate decorrelated color noise.

    Parameters:
        shape (tuple): The shape of the color noise array.
        low (float): The lower bound of the uniform distribution.
        high (float): The upper bound of the uniform distribution.

    Returns:
        numpy.ndarray: A 3D array of decorrelated color noise with shape `(shape[0], shape[1], 3)`.
    """
    return np.stack([noise(shape, low, high),noise(shape, low, high),noise(shape, low, high)], axis=2)

def greyscale_to_color(greyscale_input, r_factor=1,g_factor=1,b_factor=1):
    """
    Convert a greyscale image to a color image by duplicating the input channel and scaling it according to the factors.

    Parameters:
        greyscale_input (numpy.ndarray): The input greyscale image.
        r_factor (float): Scaling factor for the red channel.
        g_factor (float): Scaling factor for the green channel.
        b_factor (float): Scaling factor for the blue channel.

    Returns:
        numpy.ndarray: A 3D array representing the color image with shape `(greyscale_input.shape[0], greyscale_input.shape[1], 3)`.
    """
    return np.stack([greyscale_input*r_factor,greyscale_input*g_factor,greyscale_input*b_factor], axis=2)

def color(shape=(16,16), color=(1,0,0)):
    """
    Generate an image of given shape, filled with the defined color.

    Parameters:
        shape (tuple, optional): Desired output shape (height, width) of the Image. Default is (16, 16).
        color (tuple, optional): Desired color (r,g,b)

    Returns:
        numpy.ndarray: Image with shape (w,h,3).
    """
    return np.stack([np.full(shape, fill_value=color[0]),np.full(shape, fill_value=color[1]),np.full(shape, fill_value=color[2])], axis=2)
