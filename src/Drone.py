'''
drone.py

Cedrick Argueta
cdrckrgt@stanford.edu

the drone and its physics, movement
'''
import numpy as np

class Drone(object):
    def __init__(self, x, y, heading, maxStep, sensor):
        self.x = x
        self.y = y
        self.heading = heading
        self.maxStep = maxStep
        self.sensor = sensor

    def getPose(self):
        return self.x, self.y, self.heading

    def getNewPose(self, pose, action):
        newX = pose[0] + action[0]
        newY = pose[1] + action[1]
        newHeading = (pose[2] + action[2]) % 360. # ensuring that heading remains within 360 degrees
        return newX, newY, newHeading

    def act(self, action):
        self.x, self.y, self.heading = self.getNewPose(self.getPose(), action)

    def observe(self, searchdomain):
        return self.sensor.observe(searchdomain.getTargetLocation(), self.getPose())

if __name__ == '__main__':
    pass 
