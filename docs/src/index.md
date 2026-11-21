# NonconvexMMA

![JuliaNonconvex logo](https://raw.githubusercontent.com/JuliaNonconvex/NonconvexMMA.jl/master/docs/src/assets/logo.png)

## Description

There are 2 versions of the method of moving asymptotes (MMA) that are available in `NonconvexMMA.jl`:
1. The original MMA algorithm from the [1987 paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620240207).
2. The globally convergent MMA (GCMMA) algorithm from the [2002 paper](https://epubs.siam.org/doi/abs/10.1137/S1052623499362822).

The MMA algorithm only supports inequality constraints. However, the original algorithm was slightly generalized to handle infinite variable bounds.

See [Quick start](quickstart.md) for the first steps, [Options](options.md) for the algorithm options and convergence criteria, [Solver capabilities](capabilities.md) for the model features supported by the MMA algorithms, [Result](result.md) for the fields of the returned result and [Docstrings](docstrings.md) for the full API reference.
