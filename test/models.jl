using OrdinaryDiffEq, Test, UnPack
using Bijectors: Exp, inverse, Identity, Stacked
using Random; Random.seed!(2)

#=
Defining specific models to test `AbstractModel`

=#

# default behavior model
getr(p,m::AbstractModel) = p[1:m.mp.N] .|> m.mp.st.bs[1]
getb(p,m::AbstractModel) = p[m.mp.N+1:2*m.mp.N] .|> m.mp.st.bs[2]
getα(p,::AbstractModel) = NaN
getμ(p,::AbstractModel) = NaN
getδ(p,::AbstractModel) = NaN

struct Modelα <: AbstractModel
    mp::ModelParams
end
getα(p,m::Modelα) = p[end] |>  m.mp.st.bs[3]
function Modelα(mp, dists)
    @assert length(dists) == 3
    Modelα(remake(mp,st=Stacked(dists,[1:mp.N,mp.N+1:2*mp.N,2*mp.N+1])))
end

function (m::Modelα)(du, u, p, t)
    T = eltype(u)
    @unpack N = m.mp

    ũ = max.(u, 0f0)
    r = getr(p, m)
    b = getb(p, m)
    α = getα(p, m)
    du .= r .* ũ .* (1f0 .- ũ .* b .+ α * (sum(ũ) .- ũ) / convert(T,N))
    return nothing
end
Base.length(m::Modelα) = 2 * m.mp.N + 1

#=
Real tests starting here
=#

N = 10
tspan = (0., 1.)
tsteps = range(tspan[1], tspan[end], length=10)

dist = (Exp{0}(), Identity{0}(),Identity{0}())

@testset "testing model" begin
    dudt_log = Modelα(ModelParams(N=N,
                                    uworld = t -> zeros(N), 
                                    u0 = rand(N),
                                    tspan = tspan,
                                    alg = BS3()),
                                    dist)
    sol = simulate(dudt_log, p = [rand(2*N ); 0.1])
    @test sol.retcode == :Success
end


@testset "testing bijections" begin
    model = Modelα(ModelParams(N=N), dist)
    p_true = rand(length(model))
    paraminv = inverse(model.mp.st)(p_true)
    @test all(paraminv |> model.mp.st .≈ p_true)
end

using MiniBatchInference, LinearAlgebra
# TODO: make sure that the results are coherent
@testset "loglikelihood" begin
    p_true = [rand(2*N); 0.1] # all coefficients not used by all model
    dudt_log = Modelα(ModelParams(N=N,
                    uworld = t -> zeros(N), 
                    u0 = rand(N),
                    tspan = tspan,
                    alg = BS3(),
                    kwargs_sol = Dict(:saveat => tsteps,) ),
                    dist)
    ode_data = simulate(dudt_log, p = [rand(2*N); fill(0.1, length(dudt_log) - 2*N)]) |> Array

    group_size = 6
    ranges = get_ranges(group_size, length(tsteps))

    p_estimated = p_true .+ 0.02
    u0s = [ode_data[:,rg[1]] for rg in ranges]
    res = InferenceResult(dudt_log, ResultMLE(p_trained = p_estimated[1:length(dudt_log)],
                                            ranges=ranges))
    @test ParametricModels.loglikelihood(res, ode_data, 0.1; u0s = u0s) < 1.0
end