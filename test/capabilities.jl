using NonconvexMMA, Test

@testset "solver capabilities" begin
    for alg in (MMA02(), MMA87())
        caps = solver_capabilities(alg)
        @test caps isa ConstraintCapabilities
        @test caps.eq === false
        @test caps.ineq === true
        @test caps.sd === false
        @test caps.box === true
        @test caps.integer === false
        @test caps.infinite_bounds === true
    end
end
