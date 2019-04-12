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
        self.df = np.ones((buckets, buckets)) / (buckets ** 2) # buckets is num buckets per side
        self.sensor = sensor
        self.cellSize = domain.length / buckets
        self.buckets = buckets

    def update(self, pose, obs):
        '''
        updates filter with new information (obs)
        '''
        bucketSum = 0.0
        for i in range(self.buckets):
            for j in range(self.buckets):
                if self.df[i, j] > 0:
                    x = (i - 0.5) * self.cellSize
                    y = (j - 0.5) * self.cellSize
                    self.df[i, j] *= self.sensor.prob(self.domain.getTargetLocation(), pose, obs)
                    bucketSum += self.df[i, j]
        for i in range(self.buckets):
            for j in range(self.buckets):
                self.df[i, j] /= bucketSum # normalization
                    

    def centroid(self):
        x = 0
        y = 0
        for i in range(self.buckets):
            for j in range(self.buckets):
                x += (i - 0.5) * self.df[i, j]
                y += (j - 0.5) * self.df[i, j]
        return x * self.cellSize, y * self.cellSize
        

    def covariance(self):
        mu_x = 0.0
        mu_y = 0.0
        c_xx = 0.0
        c_xy = 0.0
        c_yy = 0.0
        for i in range(self.buckets):
            for j in range(self.buckets):
                x = (i - 0.5) * self.cellSize
                y = (j - 0.5) * self.cellSize

                mu_x += x * self.df[i, j]
                mu_y += y * self.df[i, j]

                c_xx += self.df[i, j] * x * x
                c_yy += self.df[i, j] * y * y
                c_xy += self.df[i, j] * x * y
        c_xx -= (mu_x * mu_x)
        c_yy -= (mu_y * mu_y)
        c_xy -= (mu_x * mu_y)
        return np.matrix([[c_xx+1e-4, c_xy], [c_xy, c_yy+1e-4]])

    def entropy(self):
        s = 0
        for i in range(self.buckets):
            for j in range(self.buckets):
                x = self.df[i, j]
                if x > 0.0:
                    s += x * np.log(x)
        return -s

    def reset(self):
        self.__init__() # is this bad
