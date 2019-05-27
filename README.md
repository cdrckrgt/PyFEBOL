# PyFEBOL

A Python rewrite of [FEBOL.jl](https://github.com/dressel/FEBOL.jl) by Louis Dressel.
It aims to also be a general framework for simulating UAV-based localization of a single target. 
The user can choose from a variety of sensors, filters, and policies (or make his own).
The package was created to better interface with reinforcement learning packages created in Python.

Currently provides:
- particle filter with stratified resampling
- discrete (histogram) filter
- bearing only sensor
- FOV sensor
- various cost models, incorporating entropy, covariance, distance, etc.
- a policy class that allows for creation of seeker and target policies
- a search domain for the seeker and target to live in

Also check out [deep-drone-localization](https://github.com/cdrckrgt/deep-drone-localization) for an implementation of DQN that works with multiple inputs, and a gym environment that uses all the stuff from PyFEBOL.

## Installation

```
pip install PyFEBOL
```
