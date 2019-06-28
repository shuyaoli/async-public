Large dataset roughly needs 150s to run. 

Specification:

n = 10000
dim = 3000
epoch = 15.

Note that result won't converge for this much epoch.

# Async_lock_free (199s in total with reading old_mean, or old_z_ik, 202.7 without it)

Roughly takes 0.54s to enter loop.

Each thread takes 200.98s / 203.3s to calculate gradient

Each thread takes 201s / 207 s to calculate dot product in gradient 

Each thread takes 0.436s / 203.3s to calculate delta_z

# Async_lock_mean (200.8s in total)
Lock both mean and z_ik. Mean and z are also stored using atomic
library for comparison. But they will be read into double for
calculation. 

Each thread takes 4s shorter to calcuate the dot product due to
<atomic> overhead.

Interestingly, reading (old) data only takes 0.5s for each thread in
total.

