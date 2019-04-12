'''
policy.py

Cedrick Argueta
cdrckrgt@stanford.edu

policy stuff
'''
import numpy as np
from util import getDistance2

class Policy(object):
    def __init__(self):
        raise Exception("please instantiate a specific policy, this is just a base class!")

    def action(self):
        raise Exception("please instantiate a specific policy, this is just a base class!")

    def makeActionList(self, maxStep, numActions, headings):
        actions = []
        angles = np.linspace(0.0, 360. - 360 / numActions, numActions)
        for angle in angles:
            ax = maxStep * np.sin(angle * np.pi / 180)
            ay = maxStep * np.cos(angle * np.pi / 180)
            if headings:
                for heading in headings:
                    actions.append((ax, ay, heading))
            else:
                actions.append((ax, ay, 0))
        # stay action
        actions.append((0.0, 0.0, 0.0))
        return actions

class MeanPolicy(Policy):
    def __init__(self, maxStep, numActions, headings=None):
        self.actions = self.makeActionList(maxStep, numActions, headings)

    def action(self, domain, vehicle, obs, f): # don't use obs here, but will for other policies
        best = (0.0, 0.0, 0.0)
        bestDist = np.inf
        cx, cy = f.centroid()
        for a in self.actions:
            x_new = vehicle.getNewPose(a)
            distToMean = getDistance2((cx, cy), x_new)
            if distToMean < bestDist:
                bestDist = distToMean
                best = a
        return best
