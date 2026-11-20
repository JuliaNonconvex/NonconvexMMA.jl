# Options

To specify options for the MMA algorithm, you can construct an instance of `MMAOptions` and use keyword arguments.
The `tol` option in MMA can be set to an instance of the `Tolerance` struct.

## Convergence criteria

There are 4 convergence criteria available for the MMA algorithm:
- `GenericCriteria`
- `KKTCriteria`
- `ScaledKKTCriteria`
- `IpoptCriteria`


To specify the convergence criteria, use:
```julia
converiteria = GenericCriteria()
```
replacing `GenericCriteria()` by `KKTCriteria()`, `ScaledKKTCriteria()` or `IpoptCriteria()`.
