# TODO: for now, this file is not working

struct ComposableModel{MS,PS} <: AbstractModel
    models::MS # tuple of abstract models
    param_indices::PS # used for indexing parameters for each submodels
end

""" 
$(SIGNATURES)

A composable model, where the dynamics `du` is the sum of `du_1, ..., du_n`
obtained from `models = (model1,...,modeln)`.

Each model needs to be given initial conditions `u0`, as those are
 used to infer `dutemp`.
"""
function ComposableModel(models::MS)  where {MS <: NTuple{N,AbstractModel} where N} # may be should be called ParametricComposableModel
    @assert all(models[1].mp.N == m.mp.N for m in models)
    shifts = [0;cumsum(length.(models))...]
    param_indices = [shifts[i] + 1:shifts[i+1] for i in 1:length(models)]
    ComposableModel(models, param_indices)
end

name(cm::ComposableModel) = string(typeof.(cm.models))

import Base.show
Base.show(io::IO, cm::ComposableModel) = println(io, "`ComposableModel` with ", name(cm))


function (cm::ComposableModel)(du, u, p, t)
    du .= zero(eltype(du))
    dutemp = similar(du)
    for i in 1:length(cm.models)
        p_i = @view p[cm.param_indices[i]]
        cm.models[i](dutemp, u, p_i, t)
        du .+= dutemp
    end
    return nothing
end

# TODO: define all these
# get_p(m::ParametricModel) = m.p
# get_np(m::ParametricModel) = m.np
# get_p_bijector(m::ParametricModel) = m.st
# get_re(m::ParametricModel) = m.re
# get_tspan(m::ParametricModel) = m.tspan
# get_dims(m::ParametricModel) = m.dims
# get_plength(m::ParametricModel) = m.plength
# get_alg(m::ParametricModel) = m.alg
# get_args(m::ParametricModel) = m.args
# get_kwargs(m::ParametricModel) = m.kwargs