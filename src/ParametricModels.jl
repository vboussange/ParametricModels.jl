__precompile__(false)

module ParametricModels
    using Statistics
    using UnPack
    using DocStringExtensions
    using SciMLBase
    using Bijectors
    using OrdinaryDiffEq, DiffEqSensitivity
    using Optimisers
    using LinearAlgebra
    using Requires

    include("models.jl")
    # include("composable_model.jl")
    # include("grad_descent.jl")
    # include("inference_result.jl")
    include("utils.jl")

    export AbstractModel, ComposableModel, simulate, ModelParams, @model, name
    # export InferenceResult, construct_result, loglikelihood, estimate_σ,
    #     get_var_covar_matrix, compute_cis, compute_cis_normal, compute_cis_lognormal,
    #     name, R2
    export Squared, Abs, NegAbs, AbsCap
    export get_p, get_u0, get_alg, get_st, get_re, get_tspan, 
        get_dims, get_plength, get_kwargs

end
