
"""
Aitchison GLN layer with arbitrary number of units.
"""
struct AGLNLayer{
        MF <: AbstractMatrix{<:Real},
        VF <: AbstractVector{<:Real},
        VI <: AbstractVector{Int32}}
    input_dim::Int
    output_dim::Int
    context_dim::Int
    predictor_dim::Int
    prediction_dim::Int

    learning_rate::Float64

    lsh::LocalitySensitiveHash{MF, VF, VI}

    weights::MF
end


"""
Aitchison geometry closure function.
"""
function closure(p::AbstractArray, simplex_dim::Int)
    return p ./ sum(p, dims=simplex_dim)
end


function closure!(p::AbstractArray, simplex_dim::Int)
    p ./= sum(p, dims=simplex_dim)
end


"""
Aitchison geometry powering function.
"""
function power(p::AbstractArray, w::AbstractArray, simplex_dim::Int)
    @show size(p)
    @show size(w)
    return closure(p .^ w, simplex_dim)
end


"""
Aitchison geometry perturbation function.
"""
function perturb(p::AbstractArray, perturb_dim::Int, simplex_dim::Int)
    return dropdims(closure(prod(p, dims=perturb_dim), simplex_dim), dims=perturb_dim)
end


"""
Construct a single AGLN layer.

  * `input_dim`: number of inputs (i.e. number of units in the prev layer)
  * `output_dim`: number of outputs (i.e. number of units in this layer)
  * `context_dim`: number of context functions (inducing 2^context_dim weight vectors)
  * `predictor_dim`: dimensionality of predictors
  * `prediction_dim`: dimensionality of predictions
  * `learning_rate`: controls step size
"""
function AGLNLayer(
        input_dim::Int, output_dim::Int,
        context_dim::Int, predictor_dim::Int, prediction_dim::Int,
        learning_rate::Float64=1e-2)

    lsh = LocalitySensitiveHash(output_dim, context_dim, predictor_dim)
    weights = zeros(Float32,(input_dim, output_dim*(2^context_dim)))

    return AGLNLayer(
        input_dim,
        output_dim,
        context_dim,
        predictor_dim,
        prediction_dim,
        learning_rate,
        lsh,
        weights)
end


"""
Forward pass for Aitchison GLN layer. Combine input prediction probability
vectors into a hopefully better prediction probability vector.
"""
function forward(
        layer::AGLNLayer, p::AbstractArray,
        z::AbstractMatrix, y::Union{Nothing, AbstractMatrix})

    batch_dim = size(p, 3)

    # [output_dim, batch_dim]
    cs = layer.lsh(z)

    function predict(weights)
        # [input_dim, output_dim, 1, batch_dim]
        weights_cs = weights[:,cs]
        penalty = 1f-5 * sum(weights_cs.^2)

        weights_ = exp.(reshape(
            weights_cs, (layer.input_dim, layer.output_dim, 1, batch_dim)))

        # [input_dim, 1, prediction_dim, batch_dim]
        p_ = reshape(p, (layer.input_dim, 1, layer.prediction_dim, batch_dim))

        # weight and combine inputs p_ by powering then perturbation

        # [output_dim, prediction_dim, batch_dim]
        q = perturb(power(p_, weights, 3), 1, 3)

        return q, penalty
    end

    if y !== nothing
        function loss(weights)
            # [output_dim, prediction_dim, batch_dim]
            q, penalty = predict(weights)

            y_ = reshape(y, (1, layer.prediction_dim, batch_dim))

            neg_ll = -sum(log.(q .* y_))

            return neg_ll + penalty
        end

        dloss_dweights = gradient(loss, layer.weights)[1]
        layer.weights .-= layer.learning_rate .* dloss_dweights
        clamp!(layer.weights, -10f0, 10f0)
    end

    return predict(layer.weights)
end


abstract type WeakClassifier end


"""
Simple logistic regression used as the weak learner input into the AGLN.
"""
struct BasicLogisticRegression{AF <: AbstractArray{<:Real, 3}} <: WeakClassifier
    # weights [predictor_dim, prediction_dim, 1]
    w::AF

    # bias [predictor_dim, prediction_dim, 1]
    b::AF

    # hyperparameters for the N(0, σ) prior on weights and biases
    σ_w::Float32
    σ_b::Float32

    # gradience descent learning rate
    learning_rate::Float32
end


"""
Construct simple logistic regression that makes `predictor_dim` predictions
seperately using each of the the predictor dimensions.
"""
function BasicLogisticRegression(
        predictor_dim::Int, prediction_dim::Int;
        σ_w=5.0, σ_b=5.0, learning_rate=1f-2)
    return BasicLogisticRegression(
        1f-3 * randn(Float32, (predictor_dim, prediction_dim, 1)),
        zeros(Float32, (predictor_dim, prediction_dim, 1)),
        Float32(σ_w), Float32(σ_b), Float32(learning_rate))
end


# x: [predictor_dim, batch_dim]
# y: [prediction_dim, batch_dim]
"""
Logistic regression forward pass. If `y` is not nothing, the w
"""
function forward(
        layer::BasicLogisticRegression,
        x::AbstractMatrix, y::Union{Nothing, AbstractMatrix})

    predictor_dim = size(layer.w, 1)
    prediction_dim = size(layer.w, 2)
    batch_dim = size(x, 2)

    @assert size(x, 1) == predictor_dim
    @assert y === nothing || size(y, 1) == prediction_dim
    @assert y === nothing || size(y, 2) == batch_dim

    function predict(w, b)
        # [predictor_dim, prediction_dim, batch_dim]
        logits = b .+ w .* reshape(x, (predictor_dim, 1, batch_dim))

        p = softmax(logits, dims=2)
        @show extrema(p)

        return p
    end

    if y !== nothing
        function loss(w, b)
            p = predict(w, b)

            y_ = reshape(y, 1, prediction_dim, batch_dim)
            true_label_likelihood = dropdims(sum(y_ .* p, dims=2), dims=2)

            ll = sum(log.(true_label_likelihood))
            lprior = sum((w ./ layer.σ_w).^2) + sum((b ./ layer.σ_b).^2)

            lp = ll + lprior

            @show lp

            return -lp
        end

        dloss_dw, dloss_db = gradient(loss, layer.w, layer.b)

        layer.w .-= layer.learning_rate .* dloss_dw
        layer.b .-= layer.learning_rate .* dloss_db
    end

    return predict(layer.w, layer.b)
end


struct AitchisonGLN{AF <: AbstractArray{<:Real,3}}
    x_norm::Union{Nothing, NormalizationLayer}
    weak_learner::BasicLogisticRegression
    layers::Vector{AGLNLayer}

    # [bias_dim, prediction_dim, 1]
    p_bias::AF
end


"""
Construct an untrained Aitchison gated linear network.

  * `predictor_dim`: dimensionality of predictors (x)
  * `prediction_dim`: dimensionality of predictions (y)
  * `num_layers`: number of layers in the model
  * `layer_width`: width of each layer in the model
  * `context_dim`: number of context functions (inducing 2^context_dim weight vectors)
  * `bias`: values for `bias` units.
"""
function AitchisonGLN(
        predictor_dim::Int, prediction_dim::Int,
        num_layers::Int, layer_width::Int, context_dim::Int;
        standardize::Bool=true, bias::Float32=5.0f0)

    bias_dim = prediction_dim - 1

    layers = AGLNLayer[]

    if num_layers > 0
        push!(
            layers, AGLNLayer(
                predictor_dim + bias_dim, layer_width, context_dim,
                predictor_dim, prediction_dim))
    end

    for i in 1:num_layers-1
        push!(
            layers, AGLNLayer(
                layer_width + bias_dim, layer_width, context_dim,
                predictor_dim, prediction_dim))
    end

    bias = zeros(Float32, (bias_dim, prediction_dim, 1))
    sbp_basis!(bias)

    return AitchisonGLN(
        standardize ? NormalizationLayer(predictor_dim) : nothing,
        BasicLogisticRegression(predictor_dim, prediction_dim),
        layers, bias)
end


"""
Construct a simplical basis using sequential binary partitioning.
"""
function sbp_basis!(basis::AbstractArray)
    i = Ref(1)
    sbp_basis!(basis, i, 1, size(basis, 2))
    map!(exp, basis, basis)
    closure!(basis, 2)
end


function sbp_basis!(basis::AbstractArray, i::Ref{Int}, from::Int, to::Int)
    num_leaves = to - from + 1
    if num_leaves <= 1
        return
    end

    mid = div(from + to, 2)

    left_subtree = from:mid
    right_subtree = mid+1:to

    len = length(from:to)
    left_len = length(left_subtree)
    right_len = length(right_subtree)

    basis[i[], from:mid, 1] .= sqrt(right_len) / sqrt(left_len * len)
    basis[i[], mid+1:to, 1] .= -sqrt(left_len) / sqrt(right_len * len)

    i[] += 1

    sbp_basis!(basis, i, from, mid)
    sbp_basis!(basis, i, mid+1, to)
end


"""
Forward pass for the AGLN, training if y is given, predicting otherwise.
"""
function forward(agln::AitchisonGLN, x::AbstractMatrix, y::Union{Nothing, AbstractMatrix})

    batch_dim = size(x, 2)

    @assert y === nothing || size(x, 2) == size(y, 2)
    @assert y === nothing || size(y, 1) == size(agln.p_bias, 2)

    xz = agln.x_norm === nothing ? x : forward(agln.x_norm, x, y !== nothing)

    p = forward(agln.weak_learner, xz, y)

    for layer in agln.layers
        # concatenate bias inputs
        p = cat(repeat(agln.p_bias, 1, 1, batch_dim), p, dims=1)

        # [output_dim, prediction_dim, batch_dim]
        p = forward(layer, p, xz, y)
    end

    return dropdims(mean(p, dims=1), dims=1)
end


"""
Train the model one step using a the batch of predictors `x` and predictions
`y`.

`x` should have shape [predictors dimension, batch size]
`y` should have shape [predictions dimension, batch size]
"""
function train!(agln::AitchisonGLN, x::AbstractMatrix, y::AbstractMatrix)
    return forward(agln, x, y)
end


"""
Make predictions using a trained model for predictors `x`.

`x` should have shape [predictors dimension, batch size]
"""
function predict(agln::AitchisonGLN, x::AbstractMatrix)
    return forward(agln, x, nothing)
end