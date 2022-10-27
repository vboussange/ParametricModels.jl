using ParametricModels
using OrdinaryDiffEq, Test
using Bijectors: Exp, inverse, Identity
using Random; Random.seed!(2)

#=
defining specific models 
to test ComposableModels

Taken from package econobio
=#


"""
    Modelδonly

Only diffusion without logistic term
"""
struct Modelδonly <: AbstractModel
    mp::ModelParams
end

function Modelδonly(mp, dists)
    @assert length(dists) == 1
    Modelδonly(remake(mp,st=Stacked(dists,[1:1])))
end

function (m::Modelδonly)(du, u, p, t)
    @unpack N, uworld = m.mp
    ũ = max.(u, 0f0)
    δ = p[end] |>  m.mp.st.bs[1]
    du .= δ .* (uworld(t) - ũ) 
    return nothing
end
Base.length(m::Modelδonly) = 1

#=
Real tests starting from here

=#

@testset "ComposableModel Modelμonly" begin
    N = 10
    tspan = (0., 1.)
    tsteps = range(tspan[1], tspan[end], length=10)

    u_init = rand(N)
    p_init = [rand(2*N); 0.1; 0.1]

    m1 = Modelα(ModelParams(N=N,
                    uworld = t -> zeros(N), 
                    u0 = u_init,
                    tspan = tspan,
                    alg = BS3(),
                    kwargs_sol = Dict(:saveat => tsteps)),
                    (Identity{0}(),Identity{0}(),Identity{0}()))
    m2 = Modelδonly(ModelParams(N=N,
                    uworld = t -> zeros(N), 
                    u0 = u_init,
                    tspan = tspan,
                    alg = BS3() ),
                    (Identity{0}(),))
    cm = ComposableModel((m1,m2))


    sol_cm = simulate(cm, p = p_init)

    @test !isnothing(Array(sol_cm))
end
