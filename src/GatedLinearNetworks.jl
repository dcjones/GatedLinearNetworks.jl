module GatedLinearNetworks

using Flux


"""
A "context function" is just a locality sensitive hash function that maps
normalized data vectors to integer indexes.
"""
struct ContextFunction{
        M <: AbstractMatrix{<:Real},
        V <: AbstractVector{<:Real},
        VI <: AbstractVector{22}}
    C::M # [context dim, data dim]
    b::V # [context dim]
    idx::VI # [context dim]
end


"""
Produce a new random context function where `d` is the dimensionality of the
data and `c` the context dimensionality.
"""
function ContextFunction(T::Real, d::Int, c::Int)
    C = rand(T, (c, d))
    C ./= sum(C.^2, dims=1)

    b = rand(T, c)

    idx = collect(Int32(0):Int32(c-1))

    return ContextFunction(C, b, idx)
end


"""
Apply the context function to data in X, where size(X, 1) is the number of
features, and size(X, 2) is the number of items in the batch.
"""
function (f::ContextFunction{M})(X::M)
    sum((f.C * X .> b) .<< F.idx, dims=1)
end


# TODO: implemnent online mean std estimation
# I think the idea is that we standardize input is it comes to us, then
# feed that into the context functions.

# Do we also standardize the input to the neurons? I don't think there is any
# nead to.


# TODO: implement simplistic online gaussian linear regression.

# What's the idea here? We need to produce multiple regression models? Each
# is fit using a supset of the points, or a subset of the dimensions?

abstract type GaussianWeakLearner end


struct BasicGaussianLinearRegression <: GaussianWeakLearner

end


function forward(bglr::BasicGaussianLinearRegression, X::M, train::Bool) where
        {M<:AbstractMatrix{<:Real}}

end



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
    data_dim::Int

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
        input_dim::Int, output_dim::Int, context_dim::Int, data_dim::Int,
        learning_rate::Float64=1e-3)
    hyperplanes = randn(T, (output_dim*context_dim, data_dim))
    hyperplanes_bias = randn(T, (output_dim*context_dim))

    bit_offsets = collect(Int32, 0:contains-1)
    k_offsets = collect(Int32, 1:output_dim)

    # How do I initialize weights? They are in some hypercube, right?
    weights = randn(T, (input_dim, output_dim*(2^context_dim)))

    return GGLNLayer(
        input_dim,
        output_dim,
        context_dim,
        data_dim,
        learning_rate,
        hyperplanes,
        hyperplanes_bias,
        bit_offsets,
        k_offsets,
        weights)
end


function context_functions(layer::GGLNLayer, X::AbstractMatrix)
    bits = (layer.hyperplanes_bias * X) .> layer.hyperplanes_bias
    batch_dim = size(X, 2)
    bits_reshape = reshape(bits, (layer.context_dim, layer.output_dim, batch_dim))
    cs = dropdims(sum(bits_reshape .<< layer.bit_offsets, dims=1), dims=1) .+ layer.k_offsets

    return cs
end



"""

y should be []
"""
function apply(
        layer::GGLNLayer, input_μ_::AbstractMatrix, input_σ2_::AbstractMatrix,
        y::AbstractMatrix, training::Bool)

    # input_μ, input_σ should be [input_dim, batch_dim]
    batch_dim = size(μ, 2)

    # [output_dim, batch_dim]
    cs = context_functions(layer, X)

    # [input_dim, output_dim, batch_dim]
    weights = layer.weights[:,cs]

    input_μ = reshape(input_μ_, (layer.input_dim, 1, batch_dim))
    input_σ2 = reshape(input_σ2_, (layer.input_dim, 1, batch_dim))

    function predict(weights)
        # [output_dim, batch_dim]
        σ2 = inv.(dropdims(sum(weights ./ input_σ2, dims=1), dims=1))

        # [dim, batch_dim]
        μ = σ2 .* dropdims(sum(weights .* input_μ ./ input_σ2, dims=1), dims=1)

        return μ, σ2
    end

    function loss(weights)
        μ, σ2 = predict(weights)
        σ = sqrt.(σ2)

        # TODO:
        # Now, I'm perplexed. I think μ should have an extra dimension, Since
        # the output dimension can be higher.

        return -sum(.- log.(σ) .- 0.5 * log.(2*π) .- 0.5 .* ((y - μ) / σ)^2)
    end

    if training
        dloss_dweights = gradient(loss, weights)
        layer.weights[:,cs] = weights - layer.learning_rate * dloss_dweights
    end

    return predict(weights)
end


end # module
