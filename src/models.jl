#=
Here we define the models as structures, 
which allows to define generic models
which can be declined for systems with a
varying number `N` of entities.

It also allows to store default solving algorithms,
so that they can be used off the shelf with minimum
rewriting.

To define a model, you need the following
## Defining the model
```
struct Modelμonly <: AbstractModel
    mp::ModelParams
end
```

## Defining a constructor, with a `Stacked` of Bijectors
in order to constrain the parameter space for each parameters used
(see https://github.com/TuringLang/Bijectors.jl)
```
function Modelμonly(mp, dists)
    @assert length(dists) == 1
    Modelμonly(remake(mp,st=Stacked(dists,[1:1])))
end
```

## Defining the dynamics
```
function (m::Modelμonly)(du, u, p, t)
    @unpack N, lap = m.mp
    ũ = max.(u, 0f0)
    μ = p[end] |>  m.mp.st.bs[1]
    du .= .- μ .* (lap * ũ) 
    return nothing
end
```

## Defining the number of parameters
Base.length(m::Modelμonly) = 1
=#

abstract type AbstractModel end
name(m::AbstractModel) = string(typeof(m))
Base.show(io::IO, cm::AbstractModel) = println(io, "`Model` ", name(cm))

"""
$(SIGNATURES)

Returns `ODEProblem` associated with `m`.
"""
function get_prob(m::AbstractModel, u0, tspan, p::AbstractArray)
    # @assert length(u0) == cm.mp.N # this is not necessary true if u0 is a vecor of u0s
    @assert length(p) == get_plength(m)
    p = inverse(get_st(m))(p)
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

## Arguments
- `kwargs`: when provided, erases `kwargs` stored in `m`. 
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
$SIGNATURES

To be completed

## Arguments
- `kwargs`: Additional arguments splatted to the ODE solver. Refer to the
[Local Sensitivity Analysis](https://diffeq.sciml.ai/dev/analysis/sensitivity/) and
[Common Solver Arguments](https://diffeq.sciml.ai/dev/basics/common_solver_opts/)
documentation for more details.

## Examples
```julia
logmodel = ModelLog(ModelParams(N=N,
                            lap=lap_mf,
                            p=[rinit; binit],
                            u0 = u0init),
                (Identity{0}(),Identity{0}()))
```

Make sure that when simulting `mymodel`, you use 
`inverse(mymodel.mp.st)(p_init)`

For distributions from bijectors, one can use:
- Identity
- Exp
- Squared
- Abs
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
    @assert eltype(p) <: Array "The values of `p` must be arrays"
    lp = [0;length.(values(p))...]
    idx_st = [sum(lp[1:i])+1:sum(lp[1:i+1]) for i in 1:length(lp)-1]

    dims = length(u0)
    plength = sum(length.(values(p)))
    _, re = Optimisers.destructure(p)

    ModelParams(;
                    p,
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

get_p(m::AbstractModel) = m.mp.p
get_u0(m::AbstractModel) = m.mp.u0
get_alg(m::AbstractModel) = m.mp.alg
get_st(m::AbstractModel) = m.mp.st
get_re(m::AbstractModel) = m.mp.re
get_tspan(m::AbstractModel) = m.mp.tspan
get_dims(m::AbstractModel) = m.mp.dims
get_plength(m::AbstractModel) = m.mp.plength
get_kwargs(m::AbstractModel) = m.mp.kwargs

macro model(name) 
    return :(
        struct $name{MP<:ModelParams} <: AbstractModel
            mp::MP
        end
    )
end