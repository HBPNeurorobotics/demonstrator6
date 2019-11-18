import numpy as np
import cv2
import os
from scipy import optimize as op


# Filter for target detection
local =  os.environ['HBP']+'/Experiments/demonstrator6/resources/'
pths  = [local+'fltr_'+c+'.bmp' for c in ['r', 'g', 'b', 'k']]
flts  = [cv2.imread(pth, cv2.IMREAD_GRAYSCALE) for pth in pths]
(flt_cols, flt_rows) = flts[0].shape


# Identify target location and orientation
def localize_target(img):

    # Prepare the probe
    probe = 255*img.numpy().transpose(1,2,0)
    probe = cv2.cvtColor(np.ascontiguousarray(probe, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
 
    # Detect the best matching location
    max_val  = 0
    targ_pos = None
    for flt in flts:
        res = cv2.matchTemplate(probe, flt, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(res)
        if val > max_val and val > 0.65:
            max_val  = val
            targ_pos = loc

    # Return the target position (order = row, col)
    if targ_pos is None:
        return targ_pos
    else:
        return (targ_pos[1] + flt.shape[1]//2, targ_pos[0] + flt.shape[0]//2)


# Highlight the target with a red box
def mark_target(img, targ_pos):

    # If the target position exists, mark it
    if targ_pos is not None:
        (cr, cc) = targ_pos
        (tr, tc) = (cr - flt_rows//2, cc - flt_cols//2)
        (br, bc) = (cr + flt_rows//2, cc + flt_cols//2)
        (tr, tc) = (max(tr,  0            ), max(tc, 0             ))
        (br, bc) = (min(br, img.shape[1]-1), min(bc, img.shape[2]-1))
        img[:,[tr,br],  tc:bc ] = 0.0
        img[:, tr:br , [tc,bc]] = 0.0
        img[0,[tr,br],  tc:bc ] = 1.0
        img[0, tr:br , [tc,bc]] = 1.0
    
    # Return the marked images
    return img


# Identify target location and orientation
def complete_target_positions(targ_pos):

    # Build the existing data points
    time_data = [t+1  for t, p in enumerate(targ_pos) if p is not None]
    row_data  = [p[0] for p in targ_pos if p is not None]
    col_data  = [p[1] for p in targ_pos if p is not None]
    
    # Fit curves to find the missing points
    pr, _ = op.curve_fit(f, time_data, row_data)
    pc, _ = op.curve_fit(f, time_data, col_data)

    # Complete the missing points
    time = [t+1 for t in range(len(targ_pos))]
    rows = [int(f(t, *pr)) if p is None else p[0] for (t,p) in zip(time, targ_pos)]
    cols = [int(f(t, *pc)) if p is None else p[1] for (t,p) in zip(time, targ_pos)]

    # Return the complete set of points to the simulation
    return zip(rows, cols)


# Simple function
def f(x, p0, p1):
    return p0*x + p1
