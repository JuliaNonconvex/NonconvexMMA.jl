# Quick start

Given a model `model` and an initial solution `x0`, the following can be used to optimize the model using MMA.
```julia
using NonconvexMMA

alg = MMA87() # or MMA02()
options = MMAOptions()
result = optimize(model, alg, x0, options = options, convcriteria = KKTCriteria())
```

## Construct an instance

To construct an instance of the original MMA algorithm, use:
```julia
alg = MMA87()
```
or alternatively:
```julia
alg = MMA()
```

To construct an instance of the globally convergent MMA algorithm, use:
```julia
alg = MMA02()
```
or alternatively:
```julia
alg = GCMMA()
```
