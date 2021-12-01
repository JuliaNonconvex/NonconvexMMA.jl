module NonconvexMMA

export MMA87, MMA02, MMA, GCMMA, MMAOptions

import NonconvexCore: addvar!, add_ineq_constraint!, getmax, getmin
import NonconvexCore: get_objective_multiple, set_objective_multiple!
import NonconvexCore: geteqconstraints, getineqconstraints, NoCallback
import NonconvexCore: geteqconstraint, getineqconstraint, getobjective
import NonconvexCore: getobjectiveconstraints, getdim, Solution
import NonconvexCore: assess_convergence!, optimize!

using Reexport, Parameters, ChainRulesCore, ForwardDiff
@reexport using NonconvexCore, Optim
using NonconvexCore: optimize
using NonconvexCore: @params, AbstractFunction, AbstractModel, VecModel
using NonconvexCore: IneqConstraint, value_gradient, value_jacobian
using NonconvexCore: AbstractOptimizer, Solution, ConvergenceCriteria
using NonconvexCore: ConvergenceState, hasconverged, debugging
using NonconvexCore: getnvars, GenericResult

"""
    Solution(dualmodel, λ)

Construct an empty solution for the dual model `dualmodel` given a sample dual solution `λ`.
"""
function Solution(dualmodel, λ)
    prevx = copy(getxk(dualmodel))
    x = copy(prevx)
    λ = copy(λ)
    prevf = Inf
    fg = getfk(dualmodel)
    ∇fg = get∇fk(dualmodel)
    f = fg[1]
    g = fg[2:end]
    ∇f = ∇fg[1, :]
    ∇g = ∇fg[2:end, :]
    convstate = ConvergenceState()
    return Solution(prevx, x, λ, prevf, f, ∇f, g, ∇g, convstate)
end

# Approximation
include("mma_approx.jl")
include("xmma_approx.jl")
include("mma_approx_docs.jl")

# Models
include("mma_model.jl")
include("mmalag_model.jl")
include("dual_model.jl")

# Algorithms
include("mma_algorithm.jl")
include("ammal.jl")

end
