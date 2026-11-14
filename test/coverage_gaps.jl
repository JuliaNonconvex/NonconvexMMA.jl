@testitem "XMMA dual solve, plotting, keep-best and stalled iterates" begin
    using NonconvexMMA, NonconvexCore, LinearAlgebra, Test, Random
    import NonconvexMMA: MMAApprox, MMAApproxModel, MMAWorkspace, debugging

    f(x::AbstractVector) = sum(x .^ 2)
    g(x::AbstractVector) = [x[1]^2 - 0.5]
    m = Model(f)
    addvar!(m, [-2.0, -2.0], [2.0, 2.0])
    add_ineq_constraint!(m, g)
    parent = NonconvexCore.tovecmodel(m)[1]
    x0 = [1.0, 1.0]

    @testset "XMMA dual-optimal-x branches" begin
        # The extended (XMMA) dual model maps λ into the primal space through
        # the c[i] - λ[i] < 0 branch; a full solve exercises it.
        r = NonconvexMMA.optimize(
            m, MMA02(), x0; extended=true, options=MMAOptions(maxiter=30)
        )
        @test isfinite(r.minimum)
    end

    @testset "LazyPlottingCallback default path" begin
        # plot_trace=true with show_plot=false constructs the callback without
        # opening a display.
        r = NonconvexMMA.optimize(
            m, MMA87(), x0; plot_trace=true, show_plot=false, options=MMAOptions(maxiter=3)
        )
        @test isfinite(r.minimum)
    end

    @testset "debugging flag and keep-best path" begin
        old = debugging[]
        debugging[] = true
        try
            r = NonconvexMMA.optimize(
                m, MMA87(), x0; options=MMAOptions(keep_best=true, maxiter=5)
            )
            @test isfinite(r.minimum)
        finally
            debugging[] = old
        end
    end

    @testset "stalled iterates keep the move limit" begin
        # A problem that converges on the first iterations makes x == prevx,
        # hitting the σ[j] retention branch of updateσ!.
        r = NonconvexMMA.optimize(
            m,
            MMA87(),
            x0;
            options=MMAOptions(
                maxiter=10, tol=Tolerance(x=1e-2, f=1e-2, kkt=1e-2, infeas=1e-2)
            ),
        )
        @test isfinite(r.minimum)
    end
end

@testitem "MMAApprox dual and pullback branches" begin
    using NonconvexMMA, NonconvexCore, LinearAlgebra, Test
    using ForwardDiff, ChainRulesCore
    import NonconvexMMA: MMAApprox, MMAApproxModel, MMAWorkspace

    f(x::AbstractVector) = sum(x .^ 2)
    g(x::AbstractVector) = [x[1]^2 - 0.5]
    m = Model(f)
    addvar!(m, [-2.0, -2.0], [2.0, 2.0])
    add_ineq_constraint!(m, g)
    parent = NonconvexCore.tovecmodel(m)[1]
    x0 = [0.8, 0.8]

    obj = NonconvexMMA.getobjective(parent)
    approx = MMAApprox(obj, x0; σ=fill(0.4, 2), ρ=0.2)

    # The Dual-eltype branch: evaluating the approximation with duals.
    xd = ForwardDiff.Dual.(x0, [1.0, 0.0])
    out_dual = approx(xd)
    @test all(ForwardDiff.value.(out_dual) .≈ approx(x0))

    # The ChainRules pullback branch with a matrix p.
    _, pb = ChainRulesCore.rrule(approx, x0)
    Δ = [1.0]
    t, grad = pb(Δ)
    @test t === nothing
    @test length(grad) == length(x0)
    @test all(isfinite, grad)
end
