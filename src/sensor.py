'''
sensor.py

Cedrick Argueta
cdrckrgt@stanford.edu

sensor stuff
'''
import numpy as np

class Sensor(object):
    def __init__(self):
        raise Exception("please instantiate a specific sensor, this is just a base class!")
    
    def observe(self):
        raise Exception("please instantiate a specific sensor, this is just a base class!")

class BearingOnlySensor(Sensor):
    def __init__(self, sigma):
        self.sigma = sigma # std dev for noise in observations

    def observe(self, theta, pose):
        truth = self.getTrueBearing(theta, pose)
        noise = self.sigma * np.random.randn()
        return (truth + noise) % 360.
 
    @staticmethod
    def getTrueBearing(self, theta, pose):
        xr = theta[0] - pose[0]        
        yr = theta[1] - pose[1]        
        return np.degrees(np.arctan2(xr, yr)) % 360.
