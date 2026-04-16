using Pkg
Pkg.activate(".")

using Flux
using CUDA

# Check GPU
device = CUDA.functional() ? gpu : cpu

# Dummy dataset
x = rand(Float32, 10, 1000)
y = sum(x, dims=1)

x = device(x)
y = device(y)

# Model
model = Chain(
    Dense(10, 64, relu),
    Dense(64, 1)
) |> device

loss(x, y) = Flux.mse(model(x), y)

opt = Adam(1e-3)

# Training loop
for epoch in 1:50
    grads = gradient(() -> loss(x, y), Flux.params(model))
    Flux.Optimise.update!(opt, Flux.params(model), grads)

    println("Epoch $epoch | Loss: $(loss(x, y))")
end