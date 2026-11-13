module NonconvexMMA

export MMA87, MMA02, MMA, GCMMA, MMAOptions

import NonconvexCore: addvar!, add_ineq_constraint!, getmax, getmin
import NonconvexCore: get_objective_multiple, set_objective_multiple!
import NonconvexCore: geteqconstraints, getineqconstraints, NoCallback
import NonconvexCore: geteqconstraint, getineqconstraint, getobjective
import NonconvexCore: getobjectiveconstraints, getdim, Solution
import NonconvexCore: assess_convergence!, optimize!, Workspace, reset!

using Reexport, Parameters, ChainRulesCore, ForwardDiff, Zygote
@reexport using NonconvexCore, Optim
using NonconvexCore: optimize
using NonconvexCore: @params, AbstractFunction, AbstractModel, VecModel
import NonconvexCore: IneqConstraint, value_gradient, value_jacobian
using NonconvexCore: AbstractOptimizer, Solution, ConvergenceCriteria
using NonconvexCore: ConvergenceState, hasconverged, debugging
using NonconvexCore: getnvars, GenericResult, getinit, LazyPlottingCallback

# Approximation
include("mma_approx.jl")
include("xmma_approx.jl")
include("mma_approx_docs.jl")

# Models
include("mma_model.jl")
include("dual_model.jl")

# Typed Solution constructor for the dual model (defined here so the
# MMAApproxModel type is in scope; the typed signature avoids extending
# NonconvexCore.Solution with untyped (broadly pirated) arguments).
function Solution(dualmodel::MMADualModel, λ)
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

# Algorithms
include("mma_algorithm.jl")

end
