import numpy as np

def gabor_kernel(shape=(16, 16), frequency=1/8, theta=np.pi/4, sigma_x=3, sigma_y=3, phase_offset=0, center_offset=(0, 0), factor=1, offset=0):
    """
    Generate a Gabor kernel with the specified shape, center offset, and phase offset.

    Parameters:
        shape (tuple, optional): Desired output shape (height, width) of the Gabor kernel. Default is (128, 128).
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
        shape (tuple, optional): Desired output shape (height, width) of the Gabor kernel. Default is (128, 128).
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