@testitem "Simple constraints" begin
    using NonconvexMMA, LinearAlgebra, Test, Zygote

    f(x::AbstractVector) = x[2] < 0 ? Inf : sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))

    @testset "MMA $(alg isa MMA87 ? "1987" : "2002")" for alg in (MMA87(), MMA02())
        for convcriteria in (KKTCriteria(), IpoptCriteria())
            options = MMAOptions(; tol=Tolerance(kkt=1e-6, f=0.0), s_init=0.1, convcriteria)
            r = NonconvexMMA.optimize(m, alg, [1.234, 2.345], options=options)
            @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
            @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
        end
    end
end

@testitem "Block constraints" begin
    using NonconvexMMA, LinearAlgebra, Test, Zygote

    f(x::AbstractVector) = x[2] < 0 ? Inf : sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, FunctionWrapper(x -> [g(x, 2, 0), g(x, -1, 1)], 2))

    @testset "MMA $(alg isa MMA87 ? "1987" : "2002")" for alg in (MMA87(), MMA02())
        for convcriteria in (KKTCriteria(), IpoptCriteria())
            options = MMAOptions(; tol=Tolerance(kkt=1e-6, f=0.0), s_init=0.1, convcriteria)
            r = NonconvexMMA.optimize(m, alg, [1.234, 2.345], options=options)
            @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
            @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
        end
    end
end

@testitem "Infinite bounds" begin
    using NonconvexMMA, LinearAlgebra, Test, Zygote

    f(x::AbstractVector) = x[2] < 0 ? Inf : sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    @testset "Infinite upper bound" begin
        m = Model(f)
        addvar!(m, [0.0, 0.0], [Inf, Inf])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        @testset "MMA $(alg isa MMA87 ? "1987" : "2002")" for alg in (MMA87(), MMA02())
            for convcriteria in (KKTCriteria(), IpoptCriteria())
                options = MMAOptions(;
                    tol=Tolerance(kkt=1e-6, f=0.0), s_init=0.1, convcriteria
                )
                r = NonconvexMMA.optimize(m, alg, [1.234, 2.345], options=options)
                @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
                @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
            end
        end
    end
    @testset "Infinite lower bound" begin
        m = Model(f)
        addvar!(m, [-Inf, -Inf], [10, 10])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        @testset "MMA $(alg isa MMA87 ? "1987" : "2002")" for alg in (MMA87(), MMA02())
            for convcriteria in (KKTCriteria(), IpoptCriteria())
                options = MMAOptions(;
                    tol=Tolerance(kkt=1e-6, f=0.0), s_init=0.1, convcriteria
                )
                r = NonconvexMMA.optimize(m, alg, [1.234, 2.345], options=options)
                @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
                @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
            end
        end
    end
    @testset "Infinite upper and lower bound" begin
        m = Model(f)
        addvar!(m, [-Inf, -Inf], [Inf, Inf])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        @testset "MMA $(alg isa MMA87 ? "1987" : "2002")" for alg in (MMA87(), MMA02())
            for convcriteria in (KKTCriteria(), IpoptCriteria())
                options = MMAOptions(;
                    tol=Tolerance(kkt=1e-6, f=0.0), s_init=0.1, convcriteria
                )
                r = NonconvexMMA.optimize(m, alg, [1.234, 2.345], options=options)
                @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
                @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
            end
        end
    end
end

# Extended formulation
# Not great for non-convex problems so we need to start pretty close
# The problem is that an optimum solution to the extended formulation may not be feasible in the original formulation. This could be a local optimum or just due to a subspace of possible optimal solutions along a subspace.
@testitem "Extended formulation" begin
    using NonconvexMMA, LinearAlgebra, Test, Zygote

    f(x::AbstractVector) = x[2] < 0 ? Inf : sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))

    for convcriteria in (KKTCriteria(), IpoptCriteria())
        options = MMAOptions(; tol=Tolerance(kkt=1e-6, f=0.0), s_init=0.1, convcriteria)
        r = NonconvexMMA.optimize(m, MMA02(), [0.4, 0.5], options=options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
    end
end

@testitem "auto_scale reports the correct objective value" begin
    using NonconvexMMA, LinearAlgebra, Test, Zygote

    f(x::AbstractVector) = x[2] < 0 ? Inf : sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))
    options = MMAOptions(;
        tol=Tolerance(kkt=1e-6, f=0.0), s_init=0.1, auto_scale=true, verbose=false
    )
    r = NonconvexMMA.optimize(m, MMA87(), [1.234, 2.345], options=options)
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
    @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
end

@testitem "reset! works" begin
    using NonconvexMMA, LinearAlgebra, Test, Zygote

    m = Model(x -> (x[1] - 1)^2 + (x[2] + 2)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    vm, x0, _ = NonconvexCore.tovecmodel(m)
    w = NonconvexMMA.MMAWorkspace(vm, MMA87(), x0; options=MMAOptions(verbose=false))
    NonconvexCore.reset!(w, [0.5, -1.5])
    r = NonconvexCore.optimize!(w)
    @test norm(r.minimizer - [1.0, -2.0]) < 0.1
end

@testitem "Fixed variables do not crash" begin
    using NonconvexMMA, LinearAlgebra, Test, Zygote

    m = Model(x -> (x[1] - 2)^2 + (x[2] - 3)^2)
    addvar!(m, [2.0, 3.0], [2.0, 3.0])
    r = NonconvexMMA.optimize(m, MMA87(), [2.0, 3.0], options=MMAOptions(verbose=false))
    @test r.minimum < 1e-6
    @test norm(r.minimizer - [2.0, 3.0]) < 1e-6
end

@testitem "optimize without x0" begin
    using NonconvexMMA, LinearAlgebra, Test, Zygote

    m = Model(x -> (x[1] - 1)^2 + (x[2] + 2)^2)
    addvar!(m, [0.0, -3.0], [2.0, -1.0])
    r = NonconvexMMA.optimize(m, MMA87(), options=MMAOptions(verbose=false))
    @test norm(r.minimizer - [1.0, -2.0]) < 1e-2
end

@testitem "scalar MMAApprox getsigma" begin
    using NonconvexMMA, LinearAlgebra, Test, Zygote

    approx = NonconvexMMA.MMAApprox(FunctionWrapper(x -> x[1]^2 + x[2]^2, 1), [1.0, 2.0])
    σ = NonconvexMMA.getσ(approx)
    @test σ isa AbstractVector
    @test length(σ) == 2
    NonconvexMMA.setσ!(approx, 0.5)
    @test all(==(0.5), NonconvexMMA.getσ(approx))
end

@testitem "MMAApproxModel rejects unknown keyword arguments" begin
    using NonconvexMMA, LinearAlgebra, Test

    m = Model(x -> (x[1] - 1)^2 + (x[2] + 2)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    parent = NonconvexCore.tovecmodel(m)[1]
    x = [0.5, -1.5]

    # Unknown keyword arguments must not be silently absorbed
    @test_throws ArgumentError NonconvexMMA.MMAApproxModel(
        parent, x; extended=false, nonsense=1
    )
    @test_throws ArgumentError NonconvexMMA.MMAApproxModel(
        parent, x; extended=true, nonsense=1
    )
    @test_throws ArgumentError NonconvexMMA.MMAApproxModel(parent, x; exended=true)
    # XMMA-only coefficients are unused with extended=false
    @test_throws ArgumentError NonconvexMMA.MMAApproxModel(parent, x; a=fill(1.0, 1))

    # Accepted keyword arguments do not throw and are forwarded
    model = NonconvexMMA.MMAApproxModel(parent, x; extended=false, σ=fill(0.5, 2))
    @test NonconvexMMA.getσ(model) == fill(0.5, 2)
    model = NonconvexMMA.MMAApproxModel(parent, x; extended=true, a=fill(2.0, 1))
    @test model.approx_objective_ineq_constraints isa NonconvexMMA.XMMAApprox
end

@testitem "XMMAApprox rejects scalar-valued MMAApprox" begin
    using NonconvexMMA, LinearAlgebra, Test

    m = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    parent = NonconvexCore.tovecmodel(m)[1]
    obj = NonconvexMMA.getobjective(parent)
    scalar_mma = NonconvexMMA.MMAApprox(obj, [0.5, 0.5])
    @test NonconvexMMA.getfk(scalar_mma) isa Real
    @test_throws ArgumentError NonconvexMMA.XMMAApprox(scalar_mma)
    @test_throws ArgumentError NonconvexMMA.XMMAApprox(scalar_mma, (;))
end

@testitem "store_trace records the solution history" begin
    using NonconvexMMA, LinearAlgebra, Test

    m = Model(x -> (x[1] - 1)^2 + (x[2] + 2)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    vm, x0, _ = NonconvexCore.tovecmodel(m)
    w = NonconvexMMA.MMAWorkspace(
        vm, MMA87(), x0; options=MMAOptions(verbose=false, store_trace=true)
    )
    r = NonconvexCore.optimize!(w)
    @test isfinite(r.minimum)
    @test !isempty(w.trace.tr)
    @test all(entry -> entry.x isa AbstractVector, w.trace.tr)
    # reset! starts a fresh trace
    NonconvexCore.reset!(w, [0.5, -1.5])
    @test isempty(w.trace.tr)
    NonconvexCore.optimize!(w)
    @test !isempty(w.trace.tr)
end

@testitem "maxiter_reached reflects the outer_maxiter exit" begin
    using NonconvexMMA, LinearAlgebra, Test

    m = Model(x -> (x[1] - 1)^2 + (x[2] + 2)^2)
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    options = MMAOptions(;
        verbose=false, maxiter=1000, outer_maxiter=1, tol=Tolerance(f=1e-15, kkt=1e-15)
    )
    r = NonconvexMMA.optimize(m, MMA87(), [0.5, -1.5], options=options)
    @test r.maxiter_reached
end
