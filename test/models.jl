using OrdinaryDiffEq, Test, UnPack
using Bijectors: Exp, inverse, Identity, Stacked
using Random; Random.seed!(2)
using DocStringExtensions, Optimisers, DiffEqSensitivity
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

dist = (Exp{0}(), Identity{0}(),Identity{0}())

@testset "testing model" begin
    p = (r = rand(N), b = rand(N), α = rand(1))
    u0 = rand(N)
    dudt_log = Modelα(ModelParams(p,
                                    dist,
                                    tspan,
                                    u0,
                                    BS3()
                                    ))
    sol = simulate(dudt_log; u0, p)
    @test sol.retcode == :Success
end


@testset "testing bijections" begin
    p_true = (r = rand(N), b = rand(N), α = rand(1))
    u0 = rand(N)
    model = Modelα(ModelParams(p_true,
                            dist,
                            tspan,
                            u0,
                            BS3()
                            ))
    pflat, _ = Optimisers.destructure(p_true)
    paraminv = inverse(get_st(model))(pflat)
    @test all(paraminv |> get_st(model) .≈ pflat)
end

using MiniBatchInference, LinearAlgebra
# TODO: make sure that the results are coherent
@testset "loglikelihood" begin
    p_true = (r = rand(N), b = rand(N), α = rand(1))
    u0 = rand(N)
    dudt_log = Modelα(ModelParams(p_true,
                                    dist,
                                    tspan,
                                    u0,
                                    BS3();
                                    saveat=tsteps
                                    ))
    ode_data = simulate(dudt_log) |> Array

    group_size = 6
    ranges = get_ranges(group_size, length(tsteps))
    pflat,_ = Optimisers.destructure(p_true)

    p_estimated = pflat .+ 0.02
    u0s = [ode_data[:,rg[1]] for rg in ranges]
    res = InferenceResult(dudt_log, ResultMLE(p_trained = p_estimated[1:get_plength(dudt_log)],
                                            ranges=ranges))
    @test ParametricModels.loglikelihood(res, ode_data, 0.1; u0s = u0s) < 1.0
end