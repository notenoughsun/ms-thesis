# Notes

Factor graph tricks

Marching the curves: state to state connection + distance + uncertainty

We have some solutions that are out of the allowed region.

We can't directly solve this with differential factors.

Smoothing and mapping: search for best allowed solution from distribution.

### What we can do:

- Add constraints to the problem
- Iteratively recalculate the problem until all conditions will not be satisfied
- Delete trajectories that doesn't fit the model. Delete all connected factors to deleted trajectories.
- Add new/deleted trajectories again to the model.

### Out of region recalculate::

Select on trajectory and with accept/reject method solve/integrate this on static building map. 

Once we have a trajectory known, fix the points which are out of allowed regions: set the covariance of given pts to zero e.g. to not recalculate this trajectory during graph optimisation.

The method is PDR + particle filter + accept/reject method.

### Add border constraints::

The alternative or similar idea. We add the constraints to the task. We add points on border to the map and add the constraints for close points.

Not feasible yet.

### Distance metrics::

We have a need of matching trajectories and solving warping problem not in time, but in state/action space.

If we can limit to one dimensional signals we can solve this step using traditional time-warping techniques.

Alternatively formulating we have all to all matching technique for close near-linear trajectories. 

Same approach was used in two papers on localisation and mapping. (Cimloc and magnetic mapping using chess coverage pattern)

We say that this technique can be used to solve some part of our problem: for straight collinear trajectories. 

This time-warping does not provide us the factors we can implement in factor graphs, but a residuals to current solution. After time warping we have to somehow recalculate graph positions and recalculate the map.

This idea has similarly in several computational methods. We separately optimize likelihood function and the problem itself.

Because of bilinear structure of magnetic field, we doesn't expect the system be able to organise the curves in direction where the field perturbations are small. If we have the corridor, the only direction we detect features is along the corridor direction.

To obtain exact positions of all states we have to condition on other information. 

Our main source of information is a pose graph with loop closures. *Being conditioned on the known building map, it may be used as* 

If we performed trajectory conditioning in orthogonal directions, the uncertainty in normal direction will be reduced. 

Our goal is to solve mapping problem both for normal and tangential uncertainty.