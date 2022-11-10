
abstract type AbstractModel end
name(m::AbstractModel) = string(nameof(typeof(m)))
Base.show(io::IO, cm::AbstractModel) = println(io, "`Model` ", name(cm))

"""
$(SIGNATURES)

Returns the `ODEProblem` associated with to `m`.

When `p` is an `AbstractVector`, p is transformed, given `m.mp.st` (bijectors).
When `p` is a Named tuple, no transformation is performed.
"""
function get_prob(m::AbstractModel, u0, tspan, p::AbstractArray)
    # @assert length(u0) == cm.mp.N # this is not necessary true if u0 is a vecor of u0s
    @assert length(p) == get_plength(m)
    p = inverse(get_p_bijector(m))(p) # projecting p in true parameter space
    # transforming in tuple
    p_tuple = get_re(m)(p)
    prob = get_prob(m, u0, tspan, p_tuple)
    return prob
end

function get_prob(m::AbstractModel, u0, tspan, p_tuple::NamedTuple)
    prob = ODEProblem(m, u0, tspan, p_tuple)
    return prob
end

"""
$(SIGNATURES)

Simulate model `m` and returns an `ODESolution`.  
When provided, keyword arguments overwrite default solving options 
in m.
When `p` is an `AbstractVector`, p is transformed, given `m.mp.st` (bijectors).
When `p` is a Named tuple, no transformation is performed.
"""
function simulate(m::AbstractModel; u0 = nothing, tspan=nothing, p = nothing, kwargs...)
    isnothing(u0) ? u0 = get_u0(m) : nothing
    isnothing(tspan) ? tspan = get_tspan(m) : nothing
    isnothing(p) ? p = get_p(m) : nothing
    prob = get_prob(m, u0, tspan, p)
    # kwargs erases get_kwargs(m)
    sol = solve(prob, get_alg(m); get_kwargs(m)..., kwargs...)
    return sol
end

struct ModelParams{P,PST,RE,T,U0,U0ST,A,D,PL,K}
    p::P # Named tuple, trainable parameters. /!\ Not real values, transformed by p_dist!
    p_bij::PST # p bijectors
    re::RE # to reconstruct parameters
    tspan::T # time span
    u0::U0 # u0. /!\ Not real values, transformed by p_dist!
    u0_bij::U0ST # u0 bijector
    alg::A # alg for ODEsolve
    dims::D # dimension of state variable
    plength::PL # parameter length
    kwargs::K # kwargs given to solve fn, e.g., saveat
end

import SciMLBase.remake
remake(mp::ModelParams; p) = ModelParams(p, mp.p_bij, mp.re, mp.tspan, mp.u0, mp.u0_bij, mp.alg, mp.dims, mp.plength, mp.kwargs)
    
# # for the remake fn
# function ModelParams(;p, 
#                     p_bij::PST, 
#                     re, 
#                     tspan, 
#                     u0, 
#                     u0_bij, 
#                     alg, 
#                     dims, 
#                     plength,
#                     kwargs) where PST <: Bijector
#     ModelParams(p, 
#                 p_bij, 
#                 re, 
#                 tspan, 
#                 u0, 
#                 u0_bij, 
#                 alg, 
#                 dims, 
#                 plength,
#                 kwargs)
# end

# model parameters
"""
$(SIGNATURES)

Type containing the details for the numerical simulation of a model.

## Arguments
- `p`: default parameter. Must be a NamedTuple!
- `p_dists`: a tuple with same length as `p`, containing bijectors.
    For distributions from bijectors, one can use:
    - Identity
    - Exp
    - Squared
    - Abs
    Examples: (Abs{0}(),Abs{0}(),Identity{0}(),Abs{0}())
- `tspan`
- `u0`
- `alg` 
- `sensealg`:
- `kwargs`: extra keyword args provided to the `solve` function.
"""
function ModelParams(; p, tspan, u0, alg,
                    p_bij = fill(Identity{0}(),length(p)),
                    u0_bij = Identity{0}(),
                    sensealg = DiffEqSensitivity.ForwardDiffSensitivity(),
                    kwargs...)
    @assert length(p_bij) == length(values(p)) "Each element of `p_dist` should correspond to an entry of `p`"
    @assert eltype(p) <: AbstractArray "The values of `p` must be arrays"
    lp = [0;length.(values(p))...]
    idx_st = [sum(lp[1:i])+1:sum(lp[1:i+1]) for i in 1:length(lp)-1]

    dims = length(u0)
    plength = sum(length.(values(p)))
    _, re = Optimisers.destructure(p)

    ModelParams(p,
                Stacked(p_bij,idx_st),
                re,
                tspan,
                u0,
                u0_bij,
                alg,
                dims,
                plength,
                (;sensealg,kwargs...))
end

get_mp(m::AbstractModel) = m.mp
get_p(m::AbstractModel) = m.mp.p
get_u0(m::AbstractModel) = m.mp.u0
get_alg(m::AbstractModel) = m.mp.alg
get_p_bijector(m::AbstractModel) = m.mp.p_bij
get_u0_bijector(m::AbstractModel) = m.mp.u0_bij
get_re(m::AbstractModel) = m.mp.re
get_tspan(m::AbstractModel) = m.mp.tspan
get_dims(m::AbstractModel) = m.mp.dims
get_plength(m::AbstractModel) = m.mp.plength
get_kwargs(m::AbstractModel) = m.mp.kwargs


"""
$SIGNATURES

Generates the skeleton of the model, a `struct` containing details of the numerical implementation.
"""
macro model(name) 
    expr = quote
        struct $name{MP<:ModelParams} <: AbstractModel
            mp::MP
        end

        $(esc(name))(;mp) = $(esc(name))(mp)
    end
    return expr
end