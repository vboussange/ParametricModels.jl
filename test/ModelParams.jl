using Bijectors
using ParametricModels
using OrdinaryDiffEq
# using SciMLBase

N = 10
tspan = (0., 1.)
tsteps = range(tspan[1], tspan[end], length=10)
p = (r = rand(N), b = rand(N), α = rand(1))
u0 = rand(N)

mp = ModelParams(p,
            tspan,
            u0,
            BS3())

@test ParametricModels.remake(mp; p = p) isa ModelParams
