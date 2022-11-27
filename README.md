# ParametricModels

[![Build Status](https://github.com/vboussange/ParametricModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/vboussange/ParametricModels.jl/actions/workflows/CI.yml?query=branch%3Amain)

**DifferentialEquations.jl** is amazing, but sometimes you want to write an ODE model once for all, and simply play around with the model without bothering further about the details of the numerical solve, etc... If this is the case, **ParametricModels.jl** is for you!

## Getting started

```julia
using ParametricModels, UnPack

# Set the model name
@model LogisticGrowth

# Define the ODE system
function LogisticGrowth(du, u, p, t)
    @unpack r, b = p
    du .=  r .* u .* ( 1. .- b .* u) 
end

tsteps = 1.:0.5:100.5
tspan = (tsteps[1], tsteps[end])
p1 = (r = [0.05, 0.06], b = [0.23, 0.5],)
u0 = ones(2)

# Instantiating the model
mp = ModelParams(p1, 
                tspan,
                u0, 
                BS3(),
                saveat = tsteps, 
                )
model = LogisticGrowth(mp)

# Playing with the model, without worrying about the details!
sol = simulate(model)

p2 = (r = [0.05, 0.06], b = [0.4, 0.6],)
sol2 = simulate(model, p = p2)

u0  = [3., 4.]
sol3 = simulate(model, u0 = u0)
```

## Defining generic models
The macro `@model` defines under the hood a `struct` which inherits AbstractModel.
You can write by hand the `struct` to write generic models, which can e.g. be declined for ODE systems with state variables with arbitrary dimensions, or with exchangeable components.

To define a model, you need the following

```julia
Base.@kwdef struct MyModel{EF} <: AbstractModel
    mp::ModelParams
    extra_field::EF #optional
end
```
And then define the ODE system as usual, where you can use `extra_field` as you wish.

Make sure to use the macro `Base.@kwdef`. This defines a constructor `MyModel(;mp, extra_field)`, which is used internally in ParametricModels.jl.