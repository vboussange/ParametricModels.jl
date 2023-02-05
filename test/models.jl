using ParametricModels
using OrdinaryDiffEq, Test, UnPack
using Bijectors
using Random; Random.seed!(2)
using Optimisers, Distributions
using ComponentArrays
#=
Defining specific models to test `AbstractModel`

=#

struct Modelα <: AbstractModel
    mp::ModelParams
end

function (m::Modelα)(du, u, p, t)
    T = eltype(u)
    N = get_dims(m)
    @unpack r, b, α = p 

    ũ = max.(u, 0f0)
    du .= r .* ũ .* (1f0 .- ũ .* b .+ α[] * (sum(ũ) .- ũ) / convert(T,N))
    return nothing
end

#=
Real tests starting here
=#

N = 10
tspan = (0., 1.)
tsteps = range(tspan[1], tspan[end], length=10)

@testset "testing `simulate` with `ComponentArray`s" begin
    p = ComponentArray(r = rand(N), b = rand(N), α = rand(1))
    u0 = rand(N)
    dudt_log = Modelα(ModelParams(;p,
                                    tspan,
                                    u0,
                                    alg=BS3()
                                    ))
    sol = simulate(dudt_log; u0, p)
    @test sol.retcode == :Success
end

@testset "testing `simulate` with `NamedTuple`s" begin
    p = (r = rand(N), b = rand(N), α = rand(1))
    u0 = rand(N)
    dudt_log = Modelα(ModelParams(;p,
                                    tspan,
                                    u0,
                                    alg=BS3()
                                    ))
    sol = simulate(dudt_log; u0, p)
    @test sol.retcode == :Success
end

@testset "testing `simulate` with kwargs" begin
    p = (r = rand(N), b = rand(N), α = rand(1))
    u0 = rand(N)
    tsteps = tspan[1]:0.1:tspan[2]
    dudt_log = Modelα(ModelParams(;p,
                                    tspan,
                                    u0,
                                    alg=BS3(),
                                    saveat = tsteps,
                                    abstol=1e-6
                                    ))
    sol = simulate(dudt_log; u0, p)
    @test sol.retcode == :Success
    @test size(Array(sol),2) == length(tsteps)
end

@testset "testing `simulate` with `p` a subset of model params" begin
    p = (r = rand(N), b = rand(N), α = rand(1))
    u0 = rand(N)
    tsteps = tspan[1]:0.1:tspan[2]
    dudt_log = Modelα(ModelParams(;p,
                                    tspan,
                                    u0,
                                    alg=BS3(),
                                    saveat = tsteps,
                                    abstol=1e-6
                                    ))
    sol = simulate(dudt_log; u0, p = (r = rand(N),))
    @test sol.retcode == :Success
    @test size(Array(sol),2) == length(tsteps)
end