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

class Particle(object):
    def __init__(self, x, y, weight, policy, domain):
        self.x = x
        self.y = y
        self.weight = weight
        self.policy = policy
        self.domain = domain

    def move(self):
        action = self.policy.action()
        
        newX = self.x + action[0]
        newY = self.y + action[1]
        newX, newY = np.clip([newX, newY], 0, self.domain.length)
        
        self.x, self.y = newX, newY
       
    def getPose(self):
        return self.x, self.y

    def __str__(self):
        return str(self.__dict__)

class ParticleFilter(Filter):
    def __init__(self, domain, buckets, sensor, policy, nb_particles):
        self.domain = domain
        self.buckets = buckets
        self.sensor = sensor
        self.policy = policy # generative model for updating particle positions
        self.cellSize = domain.length / buckets
        self.nb_particles = nb_particles
        self.particles = []
        for i in range(self.nb_particles):
            # create a set of uniformly sampled particles with same weight
            x = np.random.uniform(0, domain.length)
            y = np.random.uniform(0, domain.length)
            assert x < domain.length
            assert y < domain.length
            weight = 1.0 / self.nb_particles
            particle = Particle(x, y, weight, self.policy, self.domain)
            self.particles.append(particle)

    def getBelief(self):
        # discretize belief for input into neural net
        f = np.zeros((self.buckets, self.buckets)) / (self.buckets ** 2) # buckets is num buckets per side
        for particle in self.particles:
            x, y = particle.getPose()
            i = min(int(x // self.cellSize), self.buckets - 1) # prevents out of bounds due to particles clipping at edge of domain
            j = min(int(y // self.cellSize), self.buckets - 1)
            f[i, j] += particle.weight
        return f[np.newaxis, :, :]

    def _predictParticles(self):
        for particle in self.particles:
            particle.move()
            
    def _updateParticles(self, pose, obs):
        total_weight = 0
        for particle in self.particles:
            prob = self.sensor.prob(particle.getPose(), pose, obs)
            particle.weight *= prob # likelihood * prior
            total_weight += particle.weight
        for particle in self.particles:
            particle.weight /= total_weight

    def update(self, pose, obs):
        self._predictParticles()
        self._updateParticles(pose, obs)

    def entropy(self):
        f = self.getBelief()
        return stats.entropy(f.flatten()) 

    def centroid(self):
        x_positions = [particle.x for particle in self.particles]
        y_positions = [particle.y for particle in self.particles]
        weights = [particle.weight for particle in self.particles]
        mean_x = np.average(x_positions, weights=weights)
        mean_y = np.average(y_positions, weights=weights)
        return mean_x, mean_y
