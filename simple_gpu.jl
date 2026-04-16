using Pkg
Pkg.activate(".")

ENV["CUDA_VISIBLE_DEVICES"] = "0"
ENV["JULIA_CUDA_USE_BINARYBUILDER"] = "true"

using Flux
using CUDA
CUDA.set_runtime_version!(v"12.2")


device = CUDA.functional() ? gpu : cpu

x = rand(Float32, 10, 1000) |> device
y = sum(x, dims=1) |> device

model = Chain(
    Dense(10, 64, relu),
    Dense(64, 1)
) |> device

loss(m, x, y) = Flux.mse(m(x), y)

opt = Adam(1e-3)

#  NEW required state object
state = Flux.setup(opt, model)

for epoch in 1:50

    #  NEW gradient style (NO Params)
    grads = gradient(model) do m
        loss(m, x, y)
    end

    #  correct update call
    Flux.update!(state, model, grads[1])

    println("epoch $epoch loss = ", loss(model, x, y))
end