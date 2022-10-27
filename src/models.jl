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


"""
$(SIGNATURES)

A model takes `mp::ModelParams` as arguments, and a tuple of
bijectors `dists` to transform parameters in order to ensure, e.g., 
positivity during inference.

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
abstract type AbstractModel end
name(m::AbstractModel) = string(typeof(m))
Base.show(io::IO, cm::AbstractModel) = println(io, "`Model` ", name(cm))

"""
$(SIGNATURES)

Returns `ODEProblem` associated with `m`.
"""
function get_prob(m::AbstractModel;u0 = m.mp.u0, p=m.mp.p)
    # @assert length(u0) == cm.mp.N # this is not necessary true if u0 is a vecor of u0s
    @assert isnothing(m.mp.tspan) != true
    @assert length(p) == length(m)
    # TODO: inverse to be checked
    prob = ODEProblem(m, u0, m.mp.tspan, inverse(m.mp.st)(p))
    return prob
end

"""
$(SIGNATURES)

Simulate model `m` and returns an `ODESolution`. 
"""
function simulate(m::AbstractModel;u0 = m.mp.u0, p=m.mp.p)
    prob = get_prob(m; u0, p)
    sol = solve(prob, m.mp.alg; sensealg = DiffEqSensitivity.ForwardDiffSensitivity(), m.mp.kwargs_sol...)
    return sol
end

# model parameters
Base.@kwdef struct ModelParams{T,P,U0,G,NN,L,U,ST,A,K<: Dict}
    tspan::T = ()
    p::P = [] # parameter vector
    plabel::Vector{String} = String[] # parameter label vector
    u0::U0 = [] # ICs
    g::G = nothing # graph interactions
    N::NN # nb of products
    lap::L = nothing # laplacian
    uworld::U = nothing # input signal world
    st::ST = Identity{0}()
    alg::A = ()
    kwargs_sol::K = Dict()
end


#=
macro ParametricModel(name, length) 
    return esc(:(
        begin
            Base.@kwdef struct $name{P,MP,U0,NN,ST,A,T,K<: NamedTuple} <: AbstractModel
                p::P = [] # parameter vector
                mp::MP = () # extra model parameters (not fitted)
                plabel::Vector{String} = String[] # parameter label vector
                u0::U0 = [] # ICs
                N::NN # dimension of state variable
                st::ST = Identity{0}() # bijectors
                alg::A = () # solving algorithms
                tspan::T = ()
                kwargs_sol::K = NamedTuple() # key words algorithms
            end

            Base.length(m::$name) = $length
        end
    ))
=#