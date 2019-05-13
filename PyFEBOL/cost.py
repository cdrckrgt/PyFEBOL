'''
cost.py

Cedrick Argueta
cdrckrgt@stanford.edu

cost models
'''
import numpy as np

class CostModel(object):
   def __init__(self):
        raise Exception("please instantiate a specific cost model, this is just a base class!")

   def getCost(self, domain, drone, filter_, action):
        raise Exception("please instantiate a specific cost model, this is just a base class!")

class ConstantCostModel(CostModel):
    '''
    returns a constant cost
        - cost should be negative to disincentivize living longer
    '''
    def __init__(self, cost):
        self.cost = cost

    def getCost(self, domain, drone, filter_, action):
        return self.cost

class SimpleDistanceCostModel(CostModel):
    '''
    returns the negative norm of the difference of the position of the target and seeker.
        - incentivizes ending the episode ASAP, lest it continue to accumulate negative rewards
        - rewards are scaled based on how far the seeker is from the target
    '''
    def __init__(self):
        pass

    def getCost(self, domain, drone, filter_, action):
        x, y, _ = drone.getPose()
        x_theta, y_theta = domain.getTheta()
        return -np.linalg.norm(np.array([x, y]) - np.array([x_theta, y_theta]))

class EntropyCost(CostModel):
    def __init__(self):
        pass
