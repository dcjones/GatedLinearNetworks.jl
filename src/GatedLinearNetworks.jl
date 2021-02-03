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
function forward(layer::NormalizationLayer, x::AbstractMatrix, training::Bool)
    if training
        for j in 1:size(x, 2)
            layer.count .+= 1
            xj = x[:,j]
            delta = xj - layer.mean
            layer.mean .+= delta ./ layer.count
            delta2 = xj - layer.mean
            layer.m2 .+= delta .* delta2
        end
    end

    @assert layer.count[1] > 0

    sd = sqrt.(layer.m2 ./ layer.count)
    return (x .- layer.mean) ./ sd
end


function inverse(layer::NormalizationLayer, z::AbstractArray)
    sd = sqrt.(layer.m2 ./ layer.count)
    return (z .* sd) .+ layer.mean
end

function inverse(layer::NormalizationLayer, μ::AbstractArray, σ2::AbstractArray)
    sd = sqrt.(layer.m2 ./ layer.count)

    μ_inv = (μ .* sd) .+ layer.mean
    σ2_inv = σ2 .* sd.^2

    return μ_inv, σ2_inv
end


abstract type GaussianWeakLearner end


struct BasicGaussianLinearRegression{
        VF <: AbstractVector{<:Real},
        MF <: AbstractMatrix{<:Real},
        VI <: AbstractVector{Int}} <: GaussianWeakLearner
    # accumulated sufficient statistics
    xy::MF # [predictor_dim, prediction_dim]
    xx::VF # [predictor_dim]
    x::VF # [prediction_dim]
    y::VF # [prediction_dim]
    count::VI # [1]

    # known prior precision
    τ0::VF # [prediction_dim]
    τ::VF # [prediction_dim]
end


function BasicGaussianLinearRegression(predictor_dim::Int, prediction_dim::Int)
    return BasicGaussianLinearRegression(
        zeros(Float32, (predictor_dim, prediction_dim)),
        zeros(Float32, predictor_dim),
        zeros(Float32, predictor_dim),
        zeros(Float32, prediction_dim),
        zeros(Int, 1),
        Float32[1f0],
        Float32[1f0])
end


function forward(
        layer::BasicGaussianLinearRegression,
        x::AbstractMatrix, y::Union{Nothing, AbstractMatrix})

    predictor_dim = size(layer.xy, 1)
    prediction_dim = size(layer.xy, 2)
    batch_dim = size(x, 2)

    @assert size(x, 1) == predictor_dim
    @assert y === nothing || size(y, 1) == prediction_dim
    @assert y === nothing || size(y, 2) == batch_dim

    # [predictor_dim]
    c1 = (layer.τ0 .+ layer.τ .* layer.count) .*
        (layer.τ0 .+ layer.τ .* layer.xx) .- (layer.τ .* layer.x).^2

    # [predictor_dim]
    c2 = layer.τ ./ c1

    # [predictor_dim, prediction_dim]
    uw = (layer.τ0 .+ layer.τ .* layer.count) .* layer.xy

    vw = layer.τ .* reshape(layer.x, (predictor_dim, 1)) .*
        reshape(layer.y, (1, prediction_dim))

    # [predictor_dim, prediction_dim]
    μ_w = c2 .* (uw .- vw)

    # [predictor_dim, prediction_dim]
    ub = (layer.τ0 .+ layer.τ .* reshape(layer.xx, (predictor_dim, 1))) .*
        reshape(layer.y, (1, prediction_dim))

    vb = layer.τ .* layer.x .* layer.xy

    # [predictor_dim, predction_dim]
    μ_b = c2 .* (ub .- vb)

    # [predictors_dim, batch_dim]
    σ2 = inv.(layer.τ) .+ inv.(c1) .* (
        (layer.τ0 .+ layer.τ .* layer.count) .* x.^2 .-
        2f0 .* layer.τ .* x .* layer.x .+
        layer.τ0 .+
        layer.τ .* layer.xx)

    # repeat this across the prediction_dim, since this is an isotropic gaussian

    # [predictior_dim, prediction_dim, batch_dim]
    σ2 = repeat(reshape(σ2, (predictor_dim, 1, batch_dim)), 1, prediction_dim, 1)

    # [predictors_dim, predictions_dim, batch_dim]
    μ = reshape(x, (predictor_dim, 1, batch_dim)) .*
        reshape(μ_w, (predictor_dim, prediction_dim, 1)) .+
        reshape(μ_b, (predictor_dim, prediction_dim, 1))

    if y !== nothing
        layer.xx .+= dropdims(sum(x.*x, dims=2), dims=2)
        layer.xy .+=
            dropdims(sum(reshape(x, (predictor_dim, 1, batch_dim)) .*
            reshape(y, (1, prediction_dim, batch_dim)), dims=3), dims=3)

        layer.x .+= dropdims(sum(x, dims=2), dims=2)
        layer.y .+= dropdims(sum(y, dims=2), dims=2)
        layer.count .+= size(x, 2)
    end

    # [predictor_dim, predicton_dim, batch_dim]
    return μ, σ2
end


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
        learning_rate::Float64=1e-2)
    hyperplanes = randn(Float32, (output_dim*context_dim, predictor_dim))
    hyperplanes_bias = randn(Float32, (output_dim*context_dim))

    bit_offsets = collect(Int32, 0:context_dim-1)
    k_offsets = collect(Int32, 1:output_dim)

    weights = fill(log(1f0/input_dim), (input_dim, output_dim*(2^context_dim)))

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
function forward(
        layer::GGLNLayer, input_μ::AbstractArray, input_σ2::AbstractArray,
        z::AbstractMatrix, y::Union{Nothing, AbstractMatrix})

    # input_μ, input_σ should be [input_dim, batch_dim]
    batch_dim = size(input_μ, 3)

    # [output_dim, batch_dim]
    cs = context_functions(layer, z)

    function predict(weights)
        # [input_dim, output_dim, 1, batch_dim]
        weights_cs = weights[:,cs]
        penalty = 1f-5 * sum(weights_cs.^2)

        weights_ = exp.(reshape(
            weights_cs, (layer.input_dim, layer.output_dim, 1, batch_dim)))

        # [input_dim, 1, prediction_dim, batch_dim]
        input_μ_ = reshape(input_μ, (layer.input_dim, 1, layer.prediction_dim, batch_dim))
        input_σ2_ = reshape(input_σ2, (layer.input_dim, 1, layer.prediction_dim, batch_dim))

        # [output_dim, prediction_dim, batch_dim]
        σ2 = inv.(dropdims(sum(weights_ ./ input_σ2_, dims=1), dims=1))

        # [output_dim, prediction_dim, batch_dim]
        μ = σ2 .* dropdims(sum(weights_ .* input_μ_ ./ input_σ2_, dims=1), dims=1)

        return μ, σ2, penalty
    end

    if y !== nothing
        function loss(weights)
            # [output_dim, prediction_dim, batch_dim]
            μ, σ2, penalty = predict(weights)
            σ = sqrt.(σ2)

            y_ = reshape(y, (1, layer.prediction_dim, batch_dim))

            # negative normal log-pdf
            neg_ll = -sum(.- log.(σ) .- 0.5 * log.(2*π) .- 0.5 .* ((y_ .- μ) ./ σ).^2)

            return neg_ll + penalty

            # return neg_ll
        end

        dloss_dweights = gradient(loss, layer.weights)[1]
        layer.weights .-= layer.learning_rate .* dloss_dweights
        clamp!(layer.weights, -10f0, 10f0)
    end

    return predict(layer.weights)
end


struct GaussianGGLN{AF <: AbstractArray{<:Real,3}}
    x_norm::NormalizationLayer
    y_norm::NormalizationLayer
    weak_learner::BasicGaussianLinearRegression
    layers::Vector{GGLNLayer}

    # [bias_dim, prediction_dim, 1]
    μ_bias::AF
    σ2_bias::AF
end


function GaussianGGLN(
        predictor_dim::Int, prediction_dim::Int,
        num_layers::Int, layer_width::Int, context_dim::Int,
        bias::Float32=5.0f0)

    bias_dim = 2*prediction_dim

    layers = GGLNLayer[]

    if num_layers > 0
        push!(
            layers, GGLNLayer(
                predictor_dim + bias_dim, layer_width, context_dim,
                predictor_dim, prediction_dim))
    end

    for i in 1:num_layers-1
        push!(
            layers, GGLNLayer(
                layer_width + bias_dim, layer_width, context_dim,
                predictor_dim, prediction_dim))
    end

    # compute the bias arrays
    μ_bias = Array{Float32}(undef, (bias_dim, prediction_dim, 1))
    μ_bias = zeros(Float32, (bias_dim, prediction_dim, 1))

    # We want a bias unit for +/- bias for every prediction dimension.
    for i in 1:prediction_dim
        μ_bias[i, i, 1] = -bias
        μ_bias[prediction_dim+i, i, 1] = bias
    end
    σ2_bias = ones(Float32,  (bias_dim, prediction_dim, 1))

    return GaussianGGLN(
        NormalizationLayer(predictor_dim),
        NormalizationLayer(prediction_dim),
        BasicGaussianLinearRegression(predictor_dim, prediction_dim),
        layers, μ_bias, σ2_bias)
end


function forward(ggln::GaussianGGLN, x::AbstractMatrix, y::Union{Nothing, AbstractMatrix})

    batch_dim = size(x, 2)

    @assert y === nothing || size(x, 2) == size(y, 2)
    @assert y === nothing || size(y, 1) == size(ggln.μ_bias, 2)

    xz = forward(ggln.x_norm, x, y !== nothing)
    yz = y === nothing ? nothing : forward(ggln.y_norm, y, true)

    μ, σ2 = forward(ggln.weak_learner, xz, yz)

    for layer in ggln.layers
        # concatenate bias inputs
        μ = cat(repeat(ggln.μ_bias, 1, 1, batch_dim), μ, dims=1)
        σ2 = cat(repeat(ggln.σ2_bias, 1, 1, batch_dim), σ2, dims=1)

        # [output_dim, prediction_dim, batch_dim]
        μ, σ2 = forward(layer, μ, σ2, xz, yz)
    end

    return inverse(
        ggln.y_norm,
        dropdims(mean(μ, dims=1), dims=1),
        dropdims(mean(σ2, dims=1), dims=1))
end

end # module
