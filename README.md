# NonconvexMMA

[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://julianonconvex.github.io/NonconvexMMA.jl/stable)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://julianonconvex.github.io/NonconvexMMA.jl/dev)
[![Build Status](https://github.com/JuliaNonconvex/NonconvexMMA.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JuliaNonconvex/NonconvexMMA.jl/actions)
[![Coverage](https://codecov.io/gh/JuliaNonconvex/NonconvexMMA.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaNonconvex/NonconvexMMA.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-Blue-4473c9.svg)](https://github.com/invenia/BlueStyle)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

## Algorithms

- `MMA` / `MMA87`: the original method of moving asymptotes from the
  [1987 paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620240207).
  It builds a first-order convex separable approximation and is fast in
  practice, but is not globally convergent.
- `MMA02` / `GCMMA`: the globally convergent MMA (GCMMA) from the
  [2002 paper](https://epubs.siam.org/doi/abs/10.1137/S1052623499362822).
  It runs inner iterations that increase the lift parameter `rho` until the
  approximation upper-bounds the original functions, which provides the global
  convergence guarantee. `GCMMA` is an alias of `MMA02`.
