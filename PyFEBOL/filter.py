'''
filter.py

Cedrick Argueta
cdrckrgt@stanford.edu

filter stuff
'''
import numpy as np
import scipy.stats as stats


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

    def getBelief(self):
        raise Exception("Please instantitate a specific filter!")

class DiscreteFilter(Filter):
    def __init__(self, domain, buckets, sensor):
        self.domain = domain
        self.df = np.ones((buckets, buckets)) / (buckets ** 2) # buckets is num buckets per side
        self.sensor = sensor
        self.cellSize = domain.length / buckets
        self.buckets = buckets

    def getBelief(self):
        return self.df[np.newaxis, :, :] # adding a channel dimension

    def update(self, pose, obs):
        '''
        updates filter with new information (obs)
        '''

        i, j = np.where(self.df > 0) # i is for rows, j is for columns
        x = (j - 0.5) * self.cellSize
        y = (i - 0.5) * self.cellSize
        
        dfUpdate = np.zeros(self.df.shape)
        dfUpdate[i, j] = self.sensor.prob((x, y), pose, obs)
        self.df *= dfUpdate
        self.df /= np.sum(self.df)

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
        m = np.matrix([[c_xx+1e-4, c_xy], [c_xy, c_yy+1e-4]])
        # print("cov mat: ", m)
        # print("np version: ", np.cov(self.df))
        return m

    def entropy(self):
        return stats.entropy(self.df.flatten())

class ParticleFilter(Filter):
    '''
    simple particle filter with simple resampling, performed according to effective N updates
    '''
    def __init__(self, domain, buckets, sensor, policy, nb_particles):
        self.domain = domain
        self.buckets = buckets
        self.sensor = sensor
        self.policy = policy # generative model for updating particle positions
        self.cellSize = domain.length / buckets
        self.nb_particles = nb_particles
        self.x_particles = np.random.uniform(0, domain.length, self.nb_particles)
        self.y_particles = np.random.uniform(0, domain.length, self.nb_particles)
        self.weights = np.ones(self.nb_particles) / self.nb_particles

    def getBelief(self):
        # discretize belief for input into neural net
        f = np.zeros((self.buckets, self.buckets)) / (self.buckets ** 2) # buckets is num buckets per side
        j = np.minimum((self.x_particles // self.cellSize), self.buckets - 1).astype(int)
        i = np.minimum((self.y_particles // self.cellSize), self.buckets - 1).astype(int)
        for idx, pair in enumerate(zip(i, j)):
            idx_i, idx_j = pair
            f[idx_i, idx_j] += self.weights[idx]
        return f[np.newaxis, :, :] # add channel dimension

    def _predictParticles(self):
        moves = self.policy.action(self.nb_particles)
        self.x_particles += moves[:, 0]
        self.y_particles += moves[:, 1]
        self.x_particles = np.clip(self.x_particles, 0, self.domain.length)
        self.y_particles = np.clip(self.y_particles, 0, self.domain.length)
        
    def _updateParticles(self, pose, obs):
        prob = self.sensor.prob((self.x_particles, self.y_particles), pose, obs)
        self.weights *= prob
        self.weights /= self.weights.sum()

    def _stratifiedResample(self):
        positions = (np.random.randn(self.nb_particles) + range(self.nb_particles)) / self.nb_particles
        cumsum = np.cumsum(self.weights)
        idxs = np.zeros(self.nb_particles, dtype=int)
        i, j = 0, 0
        while i < self.nb_particles: # for all subdivisions
            # lazy eval for j, since sometimes our floating points get too close to 1.0
            if (j == self.nb_particles - 1) or (positions[i] < cumsum[j]): 
                idxs[i] = j # choose this particle in subdivision
                i += 1 # move index to next subdivision
            else: # move onto next particle in subdivision
                j += 1
        self.x_particles = self.x_particles[idxs]
        self.y_particles = self.y_particles[idxs]
        self.weights = np.ones(self.nb_particles) / self.nb_particles

    def _resampleParticles(self):
        if ((1. / np.sum(np.square(self.weights))) < (self.nb_particles / 2)):
            self._stratifiedResample()

    def update(self, pose, obs):
        self._predictParticles()
        self._updateParticles(pose, obs)
        self._resampleParticles()

    def entropy(self):
        f = self.getBelief()
        return stats.entropy(f.flatten()) 

    def centroid(self):
        mean_x = np.average(self.x_particles, weights=self.weights)
        mean_y = np.average(self.y_particles, weights=self.weights)
        return mean_x, mean_y
