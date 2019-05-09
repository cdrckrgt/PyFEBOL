'''
sensor.py

Cedrick Argueta
cdrckrgt@stanford.edu

sensor stuff
'''
import numpy as np
from scipy.stats import norm
from PyFEBOL import util

class Sensor(object):
    def __init__(self):
        raise Exception("please instantiate a specific sensor, this is just a base class!")
    
    def observe(self):
        raise Exception("please instantiate a specific sensor, this is just a base class!")

class BearingOnlySensor(Sensor):
    def __init__(self, sigma):
        self.sigma = sigma # std dev for noise in observations

    def observe(self, theta, pose):
        truth = util.getTrueBearing(theta, pose)
        noise = self.sigma * np.random.randn()
        return (truth + noise) % 360.
 
    def prob(self, theta, pose, obs):
        bearing = util.getTrueBearing(theta, pose)
        obsDiff = util.fit180(obs - bearing)
        return norm.pdf(obsDiff, 0, self.sigma) # will this return a vector?
