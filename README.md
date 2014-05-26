ttc_estimation
====================
focus of expansion estimation + time to contact estimation algorithm.

Previous works
--------------

Camus: Iterative search for "mean" location of flow (center of OF mass)

Match filter: matched focus of expansion filter for angular component
        Pros
        + Doesn't depend on magnitude!
        + weighting by participating components improves robustness
        + a radially increasing weighting may improve even further
        + invariant to rotations

        Issues
        + large search space (but can limit to small search space once found)
        + depends heavily on textured environment
