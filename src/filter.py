'''
filter.py

Cedrick Argueta
cdrckrgt@stanford.edu

filter stuff
'''
import numpy as np

class Filter(object):
    def __init__(self):
        raise Exception("Please instantitate a specific filter!")

    def update(self):
        raise Exception("Please instantitate a specific filter!")

    def centroid(self):
        raise Exception("Please instantitate a specific filter!")

    def covariance(self):
        raise Exception("Please instantitate a specific filter!")

    def entropy(self):
        raise Exception("Please instantitate a specific filter!")

    def reset(self):
        raise Exception("Please instantitate a specific filter!")


class DiscreteFilter(Filter):
    def __init__(self, domain, buckets, sensor):
        self.domain = domain
        self.df = np.ones(buckets, buckets) / (buckets ** 2) # buckets is num buckets per side
        self.sensor = sensor
        
