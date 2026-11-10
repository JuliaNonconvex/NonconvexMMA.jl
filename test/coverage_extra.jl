@testitem "Accessors, mutators and helpers coverage" begin
    using NonconvexMMA, LinearAlgebra, Test
    import NonconvexMMA: AbstractMMAApprox, MMAApprox, XMMAApprox, MMAApproxModel

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    _m = Model(f)
    addvar!(_m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(_m, x -> g(x, 2, 0))
    add_ineq_constraint!(_m, x -> g(x, -1, 1))
    parent = NonconvexCore.tovecmodel(_m)[1]

    x = fill(0.5, 2)

    @testset "AbstractMMAApprox throwing fallbacks" begin
        struct DummyApprox <: AbstractMMAApprox end
        d = DummyApprox()
        @test_throws String NonconvexMMA.getasymptotes(d)
        @test_throws String NonconvexMMA.getapproxfg(d)
        @test_throws String NonconvexMMA.getσ(d)
        @test_throws String NonconvexMMA.setσ!(d, zeros(2))
        @test_throws String NonconvexMMA.getρ(d)
        @test_throws String NonconvexMMA.setρ!(d, zeros(2))
        @test_throws String NonconvexMMA.getfk(d)
        @test_throws String NonconvexMMA.setfk!(d, 1.0)
        @test_throws String NonconvexMMA.getxk(d)
        @test_throws String NonconvexMMA.setxk!(d, zeros(2))
        @test_throws String NonconvexMMA.get∇fk(d)
        @test_throws String NonconvexMMA.set∇fk!(d, zeros(2))
    end

    @testset "MMAApprox accessors" begin
        obj = NonconvexMMA.getobjective(parent)
        approx = MMAApprox(obj, x; σ=fill(0.4, 2), ρ=0.2)
        @test NonconvexMMA.getparent(approx) === obj
        @test NonconvexMMA.getdim(approx) == 1
        @test NonconvexMMA.getxk(approx) == x
        @test NonconvexMMA.getfk(approx) isa Real
        @test NonconvexMMA.get∇fk(approx) isa AbstractVecOrMat
        @test NonconvexMMA.getσ(approx) == fill(0.4, 2)
        @test NonconvexMMA.getρ(approx) == 0.2
        L, U = NonconvexMMA.getasymptotes(approx)
        @test length(L) == 2 && length(U) == 2
        fg = NonconvexMMA.getapproxfg(approx)
        @test length(fg) == 1 # the objective only
        # setxk!/setρ! are defined on the raw MMAApprox; setσ! hits the
        # throwing AbstractMMAApprox fallback
        NonconvexMMA.setxk!(approx, fill(0.7, 2))
        @test NonconvexMMA.getxk(approx) == fill(0.7, 2)
        NonconvexMMA.setρ!(approx, 0.9)
        @test NonconvexMMA.getρ(approx) == 0.9
        @test_throws String NonconvexMMA.setσ!(approx, fill(0.6, 2))
    end

    @testset "XMMAApprox construction, accessors and saveout!" begin
        # the objective and constraint functions together: 3 functions
        obj = NonconvexMMA.getobjectiveconstraints(parent)
        mma = MMAApprox(obj, x; σ=fill(0.4, 2), ρ=fill(0.2, 3))
        # keyword constructor
        xm = XMMAApprox(mma; a=fill(1.0, 3), c=fill(1.1, 2), d=fill(1.0, 2))
        @test xm.mma === mma
        # positional NamedTuple constructor
        xm2 = XMMAApprox(mma, (; a=fill(1.0, 3), c=fill(1.1, 2), d=fill(1.0, 2)))
        @test xm2 isa XMMAApprox
        # callable: primal x + y + z concatenated
        xyz = [x; xm.y; xm.z[]]
        out = xm(xyz)
        @test length(out) == 3
        # saveout! caches the output
        NonconvexMMA.saveout!(xm, out)
        @test xm.out == out
        NonconvexMMA.saveout!(mma, NonconvexMMA.getapproxfg(mma))
        @test mma.out == NonconvexMMA.getapproxfg(mma)
        # accessors
        @test NonconvexMMA.getmmaapprox(xm) === mma
        @test NonconvexMMA.getxk(xm) == NonconvexMMA.getxk(mma)
        @test NonconvexMMA.getfk(xm) == NonconvexMMA.getfk(mma)
        @test NonconvexMMA.getσ(xm) == NonconvexMMA.getσ(mma)
        @test NonconvexMMA.getρ(xm) == NonconvexMMA.getρ(mma)
        @test NonconvexMMA.get∇fk(xm) == NonconvexMMA.get∇fk(mma)
        @test NonconvexMMA.getparent(xm) === obj
        @test NonconvexMMA.getdim(xm) == 3
        @test NonconvexMMA.getasymptotes(xm) == NonconvexMMA.getasymptotes(mma)
        @test NonconvexMMA.getapproxfg(xm) isa AbstractVector
        # XMMAApprox-level setters
        NonconvexMMA.setxk!(xm, fill(0.6, 2))
        @test NonconvexMMA.getxk(xm) == fill(0.6, 2)
        NonconvexMMA.setfk!(xm, [2.0, 3.0, 4.0])
        @test NonconvexMMA.getfk(xm) == [2.0, 3.0, 4.0]
        NonconvexMMA.setσ!(xm, fill(0.3, 2))
        @test NonconvexMMA.getσ(xm) == fill(0.3, 2)
        NonconvexMMA.setρ!(xm, [1.0, 2.0, 3.0])
        @test NonconvexMMA.getρ(xm) == [1.0, 2.0, 3.0]
        NonconvexMMA.set∇fk!(xm, ones(3, 2))
        @test NonconvexMMA.get∇fk(xm) == ones(3, 2)
        # updateapprox! for XMMAApprox
        NonconvexMMA.updateapprox!(xm, fill(0.8, 2))
        @test NonconvexMMA.getxk(xm) == fill(0.8, 2)
        NonconvexMMA.updateapprox!(xm)
        @test NonconvexMMA.getapproxfg(xm) isa AbstractVector
    end

    @testset "MMAApproxModel bounds and updates" begin
        for extended in (false, true)
            model = MMAApproxModel(parent, x; extended=extended)
            @test NonconvexMMA.getmin(model) isa AbstractVector
            @test NonconvexMMA.getmax(model) isa AbstractVector
            # updatebounds! shrinks the bounds around x
            NonconvexMMA.updatebounds!(model, fill(0.5, 2))
            @test all(NonconvexMMA.getmin(model) .>= 0.0 .- 1e-8)
            @test all(NonconvexMMA.getmax(model) .<= 10.0 .+ 1e-8)
            # updateapprox! without and with a point, and with f/∇f
            NonconvexMMA.updateapprox!(model)
            @test NonconvexMMA.getapproxfg(model.approx_objective_ineq_constraints) isa
                AbstractVector
            NonconvexMMA.updateapprox!(model, fill(0.6, 2))
            @test NonconvexMMA.getxk(model.approx_objective_ineq_constraints) ==
                fill(0.6, 2)
            fk = NonconvexMMA.getfk(model.approx_objective_ineq_constraints)
            ∇f = NonconvexMMA.get∇fk(model.approx_objective_ineq_constraints)
            NonconvexMMA.updateapprox!(model, fill(0.7, 2), fk, ∇f)
            @test NonconvexMMA.getxk(model.approx_objective_ineq_constraints) ==
                fill(0.7, 2)
            # model-level accessors forward to the approximation
            @test NonconvexMMA.getσ(model) ==
                NonconvexMMA.getσ(model.approx_objective_ineq_constraints)
            NonconvexMMA.setσ!(model, fill(0.5, 2))
            @test NonconvexMMA.getσ(model) == fill(0.5, 2)
            @test NonconvexMMA.getρ(model) ==
                NonconvexMMA.getρ(model.approx_objective_ineq_constraints)
            NonconvexMMA.setρ!(model, NonconvexMMA.getρ(model) .* 0.5)
            @test NonconvexMMA.getρ(model) ==
                NonconvexMMA.getρ(model.approx_objective_ineq_constraints)
        end
    end

    @testset "MMADualObj" begin
        for extended in (false, true)
            model = MMAApproxModel(parent, x; extended=extended)
            dual = NonconvexMMA.MMADualObj(model)
            @test dual isa NonconvexCore.AbstractFunction
            @test NonconvexMMA.getdim(dual) == 1
            λ = fill(
                1.0, length(NonconvexMMA.getfk(model.approx_objective_ineq_constraints)) - 1
            )
            val = dual(λ)
            @test val isa Real
            # the primal solution is cached
            @test dual.x isa AbstractVector
            @test length(dual.approxfg) == length(λ) + 1
        end
    end

    @testset "Trace" begin
        tr = NonconvexMMA.Trace([])
        push!(tr.tr, 1.0)
        push!(tr.tr, 2.0)
        @test tr.tr == [1.0, 2.0]
    end
end

@testitem "MMAOptions keyword coverage" begin
    using NonconvexMMA, Test

    o = NonconvexMMA.MMAOptions(;
        tol=NonconvexCore.Tolerance(; f=1e-6),
        maxiter=100,
        outer_maxiter=200,
        maxinner=50,
        s_init=0.6,
        s_incr=1.2,
        s_decr=0.5,
        store_trace=false,
        show_trace=true,
        auto_scale=false,
        keep_best=true,
        verbose=false,
    )
    @test o.tol isa NonconvexCore.Tolerance
    @test o.maxiter == 100
    @test o.outer_maxiter == 200
    @test o.maxinner == 50
    @test o.s_init == 0.6
    @test o.s_incr == 1.2
    @test o.s_decr == 0.5
    @test !o.store_trace
    @test o.show_trace
    @test !o.auto_scale
    @test o.keep_best
    @test !o.verbose
end
