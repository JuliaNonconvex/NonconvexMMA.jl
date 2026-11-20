# Solver capabilities

[`MMA87`](@ref) and [`MMA02`](@ref) declare their capabilities through
`solver_capabilities`:

- `linear_ineq = true`, `nonlinear_ineq = true`: inequality constraints of any linearity,
- equality constraints (`linear_eq`, `nonlinear_eq`) are **not** supported — the MMA approximation only handles inequalities,
- `box = true`,
- `continuous = true`,
- `infinite_bounds = true`: the original algorithm was generalized to handle infinite variable bounds,
- semidefinite constraints, binary variables and integer variables are **not** supported.

The declaration is checked against the model via `checksubproblem!` before the solve, so unsupported model features are caught early.
