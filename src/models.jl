import Base
function Base.merge(ca::ComponentArray{T}, ca2::ComponentArray{T2}) where {T, T2}
    ax = getaxes(ca)
    ax2 = getaxes(ca2)
    vks = valkeys(ax[1])
    vks2 = valkeys(ax2[1])
    _p = Vector{T}()
    for vk in vks
        if vk in vks2
            _p = vcat(_p, ca2[vk])
        else
            _p = vcat(_p, ca[vk])
        end
    end
    ComponentArray(_p, ax)
end

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
"""
function simulate(m::AbstractModel; u0 = nothing, tspan=nothing, p = nothing, alg = nothing, kwargs...)
    isnothing(u0) ? u0 = get_u0(m) : nothing
    isnothing(tspan) ? tspan = get_tspan(m) : nothing
    if isnothing(p) 
        p = get_p(m) 
    else
        # p can be a sub tuple of the full parameter tuple
        p0 = get_p(m)
        p = merge(p0, p)
    end
    isnothing(alg) ? alg = get_alg(m) : nothing
    prob = get_prob(m, u0, tspan, p)
    # kwargs erases get_kwargs(m)
    sol = solve(prob, alg; get_kwargs(m)..., kwargs...)
    return sol
end

struct ModelParams{P,T,U0,A,K}
    p::P # model parameters; we require dictionary or named tuples or componentarrays
    tspan::T # time span
    u0::U0 # initial conditions
    alg::A # alg for ODEsolve
    kwargs::K # kwargs given to solve fn, e.g., saveat
end

import SciMLBase.remake
function remake(mp::ModelParams; 
        p = mp.p, 
        tspan = mp.tspan, 
        u0 = mp.u0, 
        alg = mp.alg, 
        kwargs = mp.kwargs) 
    ModelParams(p, tspan, u0, alg, kwargs)
end
    
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
- `tspan`: time span of the simulation
- `u0`: initial condition of the simulation
- `alg`: numerical solver
- `kwargs`: extra keyword args provided to the `solve` function.

# Optional
- `p`: default parameter values
# Example
mp = ModelParams()
"""
function ModelParams(; p = nothing, tspan = nothing, u0 = nothing, alg = nothing, kwargs...)
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
"""
$SIGNATURES

Returns the dimension of the state variable
"""
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