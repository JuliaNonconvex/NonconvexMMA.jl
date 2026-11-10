using JET, NonconvexMMA

# Run JET analysis on the package.
JET.test_package(NonconvexMMA; target_modules=(NonconvexMMA,))
