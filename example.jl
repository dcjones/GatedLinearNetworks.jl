
using GatedLinearNetworks
using Dapple


batch_dim = 200

x = 10f0 * rand(Float32, (1, batch_dim))

y = Array{Float32}(undef, (2, batch_dim))
for j in 1:batch_dim
    y[1,j] = x[1,j] * cos(x[1,j]) + 4f-1 * randn(Float32)
    y[2,j] = x[1,j] * sin(x[1,j]) + 4f-1 * randn(Float32)
end

ggln = GatedLinearNetworks.GaussianGGLN(1, 2, 2, 64, 16)

ntrain_steps = 400
for step_num in 1:ntrain_steps
    @show step_num
    train!(ggln, x, y)
end

nsteps = 200
xsteps = reshape(
    collect(Float32, range(minimum(x), maximum(x), length=nsteps)),
    (1, nsteps))
μ, σ2 = predict(ggln, xsteps)

pl = plot(
    points(x=y[1,:], y=y[2,:], color=x[1,:]),
    lines(x=μ[1,:], y=μ[2,:]))
pl |> SVG("example.svg", 4inch, 4inch)