module GatedLinearNetworks

using Flux
using Statistics



"""
Normalize input using a online estimate of mean and variance using Welford's
algorithm.
"""
struct NormalizationLayer{VF <: AbstractVector{<:Real}, VI <: AbstractVector{Int}}
    count::VI # [1]
    mean::VF # [data_dim]
    m2::VF # [data_dim]
end


function NormalizationLayer(n::Int)
    return NormalizationLayer(
        Int[0],
        zeros(Float32, n),
        zeros(Float32, n))
end


"""
x is [data_dim, batch_dim]
"""
function apply(layer::NormalizationLayer, x::AbstractMatrix)
    for j in 1:size(x, 2)
        layer.count .+= 1
        xj = x[:,j]
        delta = xj - layer.mean
        layer.mean .+= delta ./ layer.count
        delta2 = xj - layer.mean
        layer.m2 .+= delta .* delta2
    end

    @assert layer.count[1] > 0

    @show layer.m2
    @show layer.count

    sd = sqrt.(layer.m2 ./ layer.count)
    return (x .- layer.mean) ./ sd
end


# TODO: implement simplistic online gaussian linear regression.

# What's the idea here? We need to produce multiple regression models? Each
# is fit using a supset of the points, or a subset of the dimensions?

abstract type GaussianWeakLearner end


struct BasicGaussianLinearRegression <: GaussianWeakLearner

end


# function forward(bglr::BasicGaussianLinearRegression, X::M, train::Bool) where
#         {M<:AbstractMatrix{<:Real}}

# end


# TODO: Each layer should be it's own type to allow for different numbers of
# neurons.

# Should each neuron be a separate type?

struct GGLNLayer{
        MF <: AbstractMatrix{<:Real},
        VF <: AbstractVector{<:Real},
        VI <: AbstractVector{Int32}}

    input_dim::Int
    output_dim::Int
    context_dim::Int
    predictor_dim::Int
    prediction_dim::Int

    learning_rate::Float64

    # context function parameters
    hyperplanes::MF
    hyperplanes_bias::VF

    # used to compute weight indexes
    bit_offsets::VI
    k_offsets::VI

    weights::MF
end


function GGLNLayer(
        input_dim::Int, output_dim::Int,
        context_dim::Int, predictor_dim::Int, prediction_dim::Int,
        learning_rate::Float64=1e-3)
    hyperplanes = randn(Float32, (output_dim*context_dim, predictor_dim))
    hyperplanes_bias = randn(Float32, (output_dim*context_dim))

    bit_offsets = collect(Int32, 0:context_dim-1)
    k_offsets = collect(Int32, 1:output_dim)

    # TODO:
    # How do I initialize weights? They are in some hypercube, right?
    weights = exp.(randn(Float32, (input_dim, output_dim*(2^context_dim))))

    return GGLNLayer(
        input_dim,
        output_dim,
        context_dim,
        predictor_dim,
        prediction_dim,
        learning_rate,
        hyperplanes,
        hyperplanes_bias,
        bit_offsets,
        k_offsets,
        weights)
end


"""
Apply the context functions for this layer, mapping (standardized) input vectors
to indexes in [1, 2^context_dim].
"""
function context_functions(layer::GGLNLayer, X::AbstractMatrix)
    bits = (layer.hyperplanes * X) .> layer.hyperplanes_bias
    batch_dim = size(X, 2)
    bits_reshape = reshape(bits, (layer.context_dim, layer.output_dim, batch_dim))
    cs = dropdims(sum(bits_reshape .<< layer.bit_offsets, dims=1), dims=1) .+ layer.k_offsets

    return cs
end


"""

input_μ: [input_dim, prediction_dim, batch_dim]
input_σ2: [input_dim, prediction_dim, batch_dim]
z: [predictor_dim, batch_dim]
y: [prediction_dim, batch_dim]
"""
function apply(
        layer::GGLNLayer, input_μ::AbstractArray, input_σ2::AbstractArray,
        z::AbstractMatrix, y::AbstractMatrix, training::Bool)

    # input_μ, input_σ should be [input_dim, batch_dim]
    batch_dim = size(input_μ, 3)

    # [output_dim, batch_dim]
    cs = context_functions(layer, z)

    # [input_dim, output_dim, 1, batch_dim]
    weights = layer.weights[:,cs]

    function predict(weights)
        weights_ = reshape(
            weights, (layer.input_dim, layer.output_dim, 1, batch_dim))

        # [input_dim, 1, prediction_dim, batch_dim]
        input_μ_ = reshape(input_μ, (layer.input_dim, 1, layer.prediction_dim, batch_dim))
        input_σ2_ = reshape(input_σ2, (layer.input_dim, 1, layer.prediction_dim, batch_dim))

        # [output_dim, prediction_dim, batch_dim]
        σ2 = inv.(dropdims(sum(weights_ ./ input_σ2_, dims=1), dims=1))

        # [output_dim, prediction_dim, batch_dim]
        μ = σ2 .* dropdims(sum(weights_ .* input_μ_ ./ input_σ2_, dims=1), dims=1)

        return μ, σ2
    end

    function loss(weights)
        # [output_dim, prediction_dim, batch_dim]
        μ, σ2 = predict(weights)
        σ = sqrt.(σ2)

        @show size(μ)
        @show size(y)

        y_ = reshape(y, (1, layer.prediction_dim, batch_dim))

        # negative normal log-pdf
        return -sum(.- log.(σ) .- 0.5 * log.(2*π) .- 0.5 .* ((y_ .- μ) ./ σ).^2)
    end

    if training
        dloss_dweights = gradient(loss, weights)[1]
        layer.weights[:,cs] = weights - layer.learning_rate * dloss_dweights
        # TODO: paper says we have to project here to keep weights within
        # a valid range. Their code doesn't seem to though.
    end

    return predict(weights)
end


end # module
