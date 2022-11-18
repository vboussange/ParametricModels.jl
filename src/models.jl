
abstract type AbstractModel end
name(m::AbstractModel) = string(nameof(typeof(m)))
Base.show(io::IO, cm::AbstractModel) = println(io, "`Model` ", name(cm))

"""
$(SIGNATURES)

Returns the `ODEProblem` associated with to `m`.
"""
function get_prob(m::AbstractModel, u0, tspan, p)
    prob = ODEProblem(m, u0, tspan, p)
    return prob
end

"""
$(SIGNATURES)

Simulate model `m` and returns an `ODESolution`.  
When provided, keyword arguments overwrite default solving options 
in m.
If `apply_bij == true`, `p` is transformed according to `m.mp.p_bij`.
"""
function simulate(m::AbstractModel; u0 = nothing, tspan=nothing, p = nothing, apply_pbij = false, kwargs...)
    isnothing(u0) ? u0 = get_u0(m) : nothing
    isnothing(tspan) ? tspan = get_tspan(m) : nothing
    isnothing(p) ? p = get_p(m) : nothing
    prob = get_prob(m, u0, tspan, p)
    # kwargs erases get_kwargs(m)
    sol = solve(prob, get_alg(m); get_kwargs(m)..., kwargs...)
    return sol
end

struct ModelParams{P,T,U0,A,K}
    p::P # Trainable parameters. /!\ Not real values, transformed by p_dist!
    tspan::T # time span
    u0::U0 # u0. /!\ Not real values, transformed by p_dist!
    alg::A # alg for ODEsolve
    kwargs::K # kwargs given to solve fn, e.g., saveat
end

import SciMLBase.remake
remake(mp::ModelParams; p) = ModelParams(p, mp.tspan, mp.u0, mp.alg, mp.kwargs)
    
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

Structure containing the details for the numerical simulation of a model.

# Arguments
- `p`: default parameter.
- `p_bij`: a bijector from `Bijectors.jl`. 
    If `p` is a `ComponentArray`, can be a tuple of bijectors, each corresponding to an entry of `p`
- `tspan`: time span of the simulation
- `u0`: initial condition of the simulation
- `alg`: numerical solver
- `kwargs`: extra keyword args provided to the `solve` function.

# Example
mp = ModelParams()
"""
function ModelParams(; p, tspan, u0, alg, kwargs...)
    ModelParams(p,
                tspan,
                u0,
                alg,
                kwargs)
end
ModelParams(p, tspan, u0, alg) = ModelParams(p, tspan, u0, alg, ())

get_mp(m::AbstractModel) = m.mp
get_p(m::AbstractModel) = m.mp.p
get_u0(m::AbstractModel) = m.mp.u0
get_alg(m::AbstractModel) = m.mp.alg
get_tspan(m::AbstractModel) = m.mp.tspan
get_kwargs(m::AbstractModel) = m.mp.kwargs
get_dims(m::AbstractModel) = length(get_u0(m))

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