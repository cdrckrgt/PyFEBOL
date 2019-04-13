'''
searchdomain.py

Cedrick Argueta
cdrckrgt@stanford.edu

the area in which the drone moves
'''
import numpy as np

class SearchDomain(object):
    def __init__(self, length, sourceMotion='none'):
        self.length = length
        self.theta = (np.random.rand() * self.length, np.random.rand() * self.length)  # random RF source location
        self.sourceMotion = sourceMotion

    def moveTarget(self):
        if self.sourceMotion == 'none':
            return
        else:
            pass # TODO: change to allow source movement
    
    def getTheta(self):
        return self.theta


if __name__ == '__main__':
    sd = SearchDomain(100)
    print(sd.getTheta())