using Pkg
Pkg.activate(".")

ENV["CUDA_VISIBLE_DEVICES"] = "0"

using CUDA

println("CUDA functional: ", CUDA.functional())
println("Device: ", CUDA.device())




device = CUDA.functional() ? gpu : cpu

using Flux

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