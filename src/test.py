from drone import Drone
from sensor import BearingOnlySensor
from searchdomain import SearchDomain
from filter import DiscreteFilter
from policy import MeanPolicy

m = SearchDomain(100)
print("theta: ", m.getTargetLocation())

s = BearingOnlySensor(5.0)

d = Drone(25, 25, 60, 2.0, s)
print("drone pose: ", d.getPose())

f = DiscreteFilter(m, 25, s)
# print("initial belief: ", f.df)

p = MeanPolicy(d.maxStep, 36)  

while p.getDistance2(d.getPose(), m.getTargetLocation()) > 5:
    # observe
    obs = s.observe(m.getTargetLocation(), d.getPose())
    print("sample obs: ", obs)

    # update filter belief
    f.update(d.getPose(), obs)
    # print("updated belief: ", f.df)

    # calculate action
    a = p.action(m, d, obs, f)

    # act
    d.act(a)

    # confirm that it's moved
    print("drone pose, after movement: ", d.getPose())
