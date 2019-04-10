from drone import Drone
from sensor import BearingOnlySensor
from searchdomain import SearchDomain

m = SearchDomain(100)
print("theta: ", m.getTargetLocation())

s = BearingOnlySensor(5.0)

d = Drone(25, 25, 60, 2.0, s)
print("drone pose: ", d.getPose())

# observe
print("sample observation: ", s.observe(m.getTargetLocation(), d.getPose()))

# update filter belief

# calculate action

# act
