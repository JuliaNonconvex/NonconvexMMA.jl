using NonconvexMMA, LinearAlgebra, Test, Zygote, FiniteDifferences
const FDM = FiniteDifferences

f(x::AbstractVector) = sqrt(x[2])
g(x::AbstractVector, a, b) = (a*x[1] + b)^3 - x[2]

_m = Model(f)
addvar!(_m, [0.0, 0.0], [10.0, 10.0])
add_ineq_constraint!(_m, x -> g(x, 2, 0))
add_ineq_constraint!(_m, x -> g(x, -1, 1))
m = NonconvexCore.tovecmodel(_m)[1]

x = fill(0.5, 2)

@testset "Gradient and hessian of MMA approximation" begin
    exactf = NonconvexMMA.getobjective(m)
    approxf = NonconvexMMA.MMAApprox(exactf, x)
    val1, grad1 = NonconvexCore.value_gradient(exactf, x)
    val2, grad2 = exactf(x), Zygote.gradient(exactf, x)[1]
    val3, grad3 = NonconvexCore.value_gradient(approxf, x)
    val4, grad4 = approxf(x), Zygote.gradient(approxf, x)[1]
    grad5 = FDM.grad(central_fdm(5, 1), approxf, x)[1]
    @test val1 ≈ val2 ≈ val3 ≈ val4
    @test grad1 ≈ grad2 ≈ grad3 ≈ grad4 ≈ grad5
    for i in 1:100
        y = rand(2)*10
        H = Zygote.hessian(approxf, y)
        for i in 1:2, j in 1:2
            if i == j
                @test H[i, j] >= 0
            else
                @test H[i, j] == 0
            end
        end
    end

    exactf = NonconvexMMA.getobjective(m)
    approxf = NonconvexMMA.MMAApprox(exactf, x)
    val1, grad1 = NonconvexCore.value_gradient(exactf, x)
    val2, grad2 = exactf(x), Zygote.gradient(exactf, x)[1]
    val3, grad3 = NonconvexCore.value_gradient(approxf, x)
    val4, grad4 = approxf(x), Zygote.gradient(approxf, x)[1]
    grad5 = FDM.grad(central_fdm(5, 1), approxf, x)[1]
    @test val1 ≈ val2 ≈ val3 ≈ val4
    @test grad1 ≈ grad2 ≈ grad3 ≈ grad4 ≈ grad5
    for i in 1:100
        y = rand(2)*10
        H = Zygote.hessian(approxf, y)
        for i in 1:2, j in 1:2
            if i == j
                @test H[i, j] >= 0
            else
                @test H[i, j] == 0
            end
        end
    end

    exactf = NonconvexMMA.getineqconstraints(m)
    approxf = NonconvexMMA.MMAApprox(exactf, x)
    val1, jac1 = NonconvexCore.value_jacobian(exactf, x)
    val2 = exactf(x)
    jac2 = vcat([Zygote.gradient(x -> exactf(x)[i], x)[1]' for i in 1:length(val2)]...)
    val3, jac3 = NonconvexCore.value_jacobian(approxf, x)
    val4 = approxf(x)
    jac4 = vcat([Zygote.gradient(x -> approxf(x)[i], x)[1]' for i in 1:length(val2)]...)
    jac5 = vcat([FDM.grad(central_fdm(5, 1), x -> approxf(x)[i], x)[1]' for i in 1:length(val2)]...)

    @test val1 ≈ val2 ≈ val3 ≈ val4
    @test jac1 ≈ jac2 ≈ jac3 ≈ jac4 ≈ jac5
end
@testset "Jacobian of MMA approximation" begin
    approx = NonconvexMMA.MMAApproxModel(m, x)
    exactfs = approx.objective_ineq_constraints
    approxfs = approx.approx_objective_ineq_constraints
    @test exactfs(x) ≈ approxfs(x)

    val1, jac1 = NonconvexCore.value_jacobian(exactfs, x)
    val2, jac2 = NonconvexCore.value_jacobian(approxfs, x)
    @test val1 ≈ val2
    @test jac1 ≈ jac2
end

@testset "Dual objective" begin
    approx = NonconvexMMA.MMAApproxModel(m, x)
    dualmodel = NonconvexMMA.MMADualModel(approx)
    dualobj = dualmodel.obj
    λ = [2.0, 3.0]

    @testset "Value and gradient wrt λ" begin
        val1, grad1 = NonconvexCore.value_gradient(dualobj, λ)
        optimalx = copy(NonconvexMMA.optimizeprimal!(dualobj, λ))
        fg = approx.approx_objective_ineq_constraints
        val2 = fg(optimalx)' * [1.0; λ]
        grad2 = FDM.grad(central_fdm(5, 1), dualobj, λ)[1]
        @test val1 ≈ val2
        @test grad1 ≈ grad2

        grad1 = Zygote.gradient(λ -> dualobj(optimalx, λ), λ)[1]
        grad2 = FDM.grad(central_fdm(5, 1), λ -> dualobj(optimalx, λ), λ)[1]
        @test grad1 ≈ grad2 ≈ fg(optimalx)[2:end]
    end

    @testset "Value and gradient wrt x" begin
        optimalx = copy(NonconvexMMA.optimizeprimal!(dualobj, λ))
        grad1 = Zygote.gradient(x -> dualobj(x, λ), optimalx)[1]
        grad2 = FDM.grad(central_fdm(5, 1), x -> dualobj(x, λ), optimalx)[1]
        @test grad1 ≈ grad2
        @test all(1:length(x)) do j
            abs(grad1[j]) <= 1e-6 || 
                optimalx[j] == 0 && grad1[j] > 0 ||
                optimalx[j] == 10 && grad1[j] < 0
        end

        approxfs = approx.approx_objective_ineq_constraints
        val, jac = NonconvexCore.value_jacobian(approxfs, optimalx)
        @test grad1 ≈ jac' * [1; λ]
    end
end
