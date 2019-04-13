'''
util.py

Cedrick Argueta
cdrckrgt@stanford.edu

utility functions
'''
import numpy as np

def getDistance2(p0, p1):
    dx = p0[0] - p1[0]
    dy = p0[1] - p1[1]
    return dx ** 2 + dy ** 2

def fit180(angle):
    if angle > 180:
        angle -= 360.
    elif angle < -180:
        angle += 360.
    return angle

def getTrueBearing(theta, pose):
    xr = theta[0] - pose[0]        
    yr = theta[1] - pose[1]        
    return np.degrees(np.arctan2(xr, yr)) % 360.
