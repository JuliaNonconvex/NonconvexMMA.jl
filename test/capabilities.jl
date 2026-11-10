using NonconvexMMA, Test

@testset "solver capabilities" begin
    for alg in (MMA02(), MMA87())
        caps = solver_capabilities(alg)
        @test caps isa ConstraintCapabilities
        @test caps.linear_eq === false
        @test caps.nonlinear_eq === false
        @test caps.linear_ineq === true
        @test caps.nonlinear_ineq === true
        @test caps.linear_sd === false
        @test caps.nonlinear_sd === false
        @test caps.box === true
        @test caps.integer === false
        @test caps.infinite_bounds === true
    end
end
