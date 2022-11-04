
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
    p = inverse(get_st(m))(p) # projecting p in true parameter space
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

Base.@kwdef struct ModelParams{P,ST,RE,T,U0,A,D,PL,K}
    p::P # Named tuple, trainable parameters
    st::ST # Bijectors
    re::RE # to reconstruct parameters
    tspan::T # time span
    u0::U0 # u0
    alg::A # alg for ODEsolve
    dims::D # dimension of state variable
    plength::PL # parameter length
    kwargs::K # kwargs given to solve fn, e.g., saveat
end

# model parameters
"""
$(SIGNATURES)

Type containing the details for the numerical simulation of a model.

## Arguments
- `p`: default parameter. Must be a NamedTuple!
- `dists`: a tuple with same length as `p`, containing bijectors.
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
function ModelParams(
                    p,
                    dists,
                    tspan,
                    u0,
                    alg;
                    sensealg = DiffEqSensitivity.ForwardDiffSensitivity(),
                    kwargs...)
    @assert length(dists) == length(values(p))
    @assert eltype(p) <: AbstractArray "The values of `p` must be arrays"
    lp = [0;length.(values(p))...]
    idx_st = [sum(lp[1:i])+1:sum(lp[1:i+1]) for i in 1:length(lp)-1]

    dims = length(u0)
    plength = sum(length.(values(p)))
    _, re = Optimisers.destructure(p)

    ModelParams(;p,
                st=Stacked(dists,idx_st),
                re,
                tspan,
                u0,
                alg,
                dims,
                plength,
                kwargs=(;sensealg,kwargs...))
end

ModelParams(p, tspan, u0, alg; kwargs...) = ModelParams(p, fill(Identity{0}(),length(p)), tspan, u0, alg; kwargs...)

# This may be the other option for using remake function from SciMLBase
# see https://github.com/SciML/SciMLBase.jl/blob/d7e3d316a014967414f416efa35e4d1aca5458e5/src/remake.jl#L1-L4
# but for some reason, ,this is making Julia crash, 
# So we keep the other options
# ModelParams(;p, st, re, tspan, u0, alg, dims, plength, kwargs...) = ModelParams(;p, st, re, tspan, u0, alg, dims, plength, kwargs = kwargs)

@generated function struct_as_namedtuple_with_kw(st)
    # function taken from https://github.com/SciML/SciMLBase.jl/blob/d7e3d316a014967414f416efa35e4d1aca5458e5/src/remake.jl#L1-L4
    # but where we consider the field kwargs
    A = (Expr(:(=), n, :(st.$n)) for n in fieldnames(st))
    Expr(:tuple, A...)
end

function remake(mp::ModelParams; kwargs...)
    ModelParams(; struct_as_namedtuple_with_kw(mp)..., kwargs...)
end

get_mp(m::AbstractModel) = m.mp
get_p(m::AbstractModel) = m.mp.p
get_u0(m::AbstractModel) = m.mp.u0
get_alg(m::AbstractModel) = m.mp.alg
get_st(m::AbstractModel) = m.mp.st
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