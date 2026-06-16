"""Real-world trajectory dataset loaders for the MSc thesis (Axis A).

Loaders convert recorded pedestrian (and, for VCI, vehicle) trajectories into
the in-repo conventions (PedestrianState, metres, world frame) so they can be
replayed through the existing observer/predictor/planner pipeline.

Raw data is gitignored and must not be redistributed (see scripts/).
"""
