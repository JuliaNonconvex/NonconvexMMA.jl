@testitem "JET" begin
    using JET, NonconvexMMA
    JET.test_package(NonconvexMMA; target_modules=(NonconvexMMA,))
end
