@testitem "Aqua" begin
    using Aqua, NonconvexMMA
    Aqua.test_all(NonconvexMMA)
end
