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

class DistanceCostModel(CostModel):
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

class EntropyCostModel(CostModel):
    '''
    returns the negative entropy of the filter.
        - incentivizes ending the episode ASAP, lest it continue to accumulate large negative rewards
        - also incentivizes keeping entropy low
    '''
    def __init__(self):
        pass

    def getCost(self, domain, drone, filter_, action):
        entropy = filter_.entropy()
        return -entropy

class EntropyDistanceCostModel(CostModel):
    '''
    louis' cost model. cost is:
        entropy + lambda * expectation over all bins (collision occurs)
    
    incentivizes keeping entropy low, while also staying a distance away from target
    '''
    def __init__(self, lambda_, threshold):
        self.lambda_ = lambda_
        self.threshold = threshold

    def getCost(self, domain, drone, filter_, action):
        entropy = filter_.entropy()

        it = np.nditer(filter_.getBelief(), flags=['multi_index'])
        expectation = 0
        while not it.finished:
            prob = it[0] # value of prob in this bucket
            idx = it.multi_index # index of this bucket in filter (channel, x, y)
            x = (idx[1] + 0.5) * filter_.cellSize
            y = (idx[2] + 0.5) * filter_.cellSize
 
            x_seeker, y_seeker, _ = drone.getPose()
            norm = np.linalg.norm(np.array([x, y]) - np.array([x_seeker, y_seeker]))

            if norm < self.threshold: # if there's a collision for this bin
                expectation += prob

            it.iternext()

        expectation *= self.lambda_

        return -(entropy + expectation) # cost negative to discourage high entropy and collisions
        
class HighestProbDistanceCostModel(CostModel):
    '''
    rewards highest prob in a bucket
    penalizes near collisions
    '''
    def __init__(self, lambda_, threshold):
        self.lambda_ = lambda_
        self.threshold = threshold

    def getCost(self, domain, drone, filter_, action):

        max_prob = filter_.maxProbBucket()
        expectation = 0.0

        F = filter_.getBelief().squeeze()
        i, j = np.nonzero(F)
        x = (j + 0.5) * filter_.cellSize
        y = (i + 0.5) * filter_.cellSize

        centers = np.dstack((x, y)).squeeze()

        x_seeker, y_seeker, _ = drone.getPose()
        pose = np.array([x_seeker, y_seeker])

        if centers.shape == pose.shape:
            norms = np.linalg.norm(centers - pose)
        else:
            norms = np.linalg.norm(centers - pose, axis=1)

        expectation = np.sum(F[i, j][norms < self.threshold])
        expectation *= self.lambda_

        return (max_prob - expectation)

class MaxEigenvalDistanceCostModel(CostModel):
    '''
    max eigenvalue of covariance matrix corresponds to variance in most uncertain direction
    penalize this and also near-collisions
    '''
    def __init__(self, lambda_, threshold):
        self.lambda_ = lambda_
        self.threshold = threshold

    def getCost(self, domain, drone, filter_, action):
        max_eig = filter_.maxEigenvalue()

        it = np.nditer(filter_.getBelief(), flags=['multi_index'])
        expectation = 0
        while not it.finished:
            prob = it[0] # value of prob in this bucket
            idx = it.multi_index # index of this bucket in filter (channel, x, y)
            x = (idx[1] + 0.5) * filter_.cellSize
            y = (idx[2] + 0.5) * filter_.cellSize
            
            x_seeker, y_seeker, _ = drone.getPose()
            norm = np.linalg.norm(np.array([x, y]) - np.array([x_seeker, y_seeker]))

            if norm < self.threshold: # if there's a collision for this bin
                expectation += prob

            it.iternext()

        expectation *= self.lambda_

        return -(max_eig + expectation)

class DiscreteProbDistanceCostModel(CostModel):
    '''
    rewards if highest prob above threshold
    penalizes if near collisions above threshold
    '''
    def __init__(self, distance_threshold, entropy_threshold, collision_threshold):
        self.distance_threshold = distance_threshold
        self.entropy_threshold = entropy_threshold
        self.collision_threshold = collision_threshold

    def getCost(self, domain, drone, filter_, action):

        max_prob = filter_.maxProbBucket()
        expectation = 0.0

        F = filter_.getBelief().squeeze()
        i, j = np.nonzero(F)
        x = (j + 0.5) * filter_.cellSize
        y = (i + 0.5) * filter_.cellSize

        centers = np.dstack((x, y)).squeeze()

        x_seeker, y_seeker, _ = drone.getPose()
        pose = np.array([x_seeker, y_seeker])

        if centers.shape == pose.shape:
            norms = np.linalg.norm(centers - pose)
        else:
            norms = np.linalg.norm(centers - pose, axis=1)

        expectation = np.sum(F[i, j][norms < self.distance_threshold])

        reward = 1 if max_prob > self.entropy_threshold else 0
        reward = -1 if expectation > self.collision_threshold else reward
        return reward

class ThresholdProbDistanceCostModel(CostModel):
    '''
    rewards if highest prob above threshold
    penalizes if near collisions above threshold
    '''
    def __init__(self, distance_threshold, entropy_threshold, lambda_):
        self.distance_threshold = distance_threshold
        self.entropy_threshold = entropy_threshold
        self.lambda_ = lambda_

    def getCost(self, domain, drone, filter_, action):

        max_prob = filter_.maxProbBucket()
        expectation = 0.0

        F = filter_.getBelief().squeeze()
        i, j = np.nonzero(F)
        x = (j + 0.5) * filter_.cellSize
        y = (i + 0.5) * filter_.cellSize

        centers = np.dstack((x, y)).squeeze()

        x_seeker, y_seeker, _ = drone.getPose()
        pose = np.array([x_seeker, y_seeker])

        if centers.shape == pose.shape:
            norms = np.linalg.norm(centers - pose)
        else:
            norms = np.linalg.norm(centers - pose, axis=1)

        expectation = np.sum(F[i, j][norms < self.distance_threshold])

        belief_reward = max(float(max_prob - self.entropy_threshold) / float(1 - self.entropy_threshold), 0.0)
        collision_reward = self.lambda_ * expectation
        return belief_reward - collision_reward 
