import numpy as np
from scipy import ndimage as nd

# Identify target location and orientation
def localize(img):
    
    # Initialize filter and values
    filtr  = np.zeros((16, 16))
    square = np.ones((10, 10))
    filtr[3:13, 3:13] = square
    max_val = 0
    max_loc = (0,0)
    max_ang = 0

    # Find target location and orientation
    for angle in np.arange(0, 90, 10):
	    green  = nd.convolve(img[1,:,:], nd.rotate(filtr, angle))
	    loc    = np.unravel_index(green.argmax(), green.shape)
	    val    = green[loc]
	    if val > max_val:
	    	max_val = val
	    	max_loc = loc
	    	max_ang = angle

	# Mark the target with a small red square
    img[:, max_loc[0]-2:max_loc[0]+3, max_loc[1]-2:max_loc[1]+3] = 1.0
    img[1, max_loc[0]-2:max_loc[0]+3, max_loc[1]-2:max_loc[1]+3] = 0.0
    img[2, max_loc[0]-2:max_loc[0]+3, max_loc[1]-2:max_loc[1]+3] = 0.0
    
    # Return everything
    return img, max_loc, max_ang
