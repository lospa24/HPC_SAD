using Pkg
Pkg.activate(".")

ENV["CUDA_VISIBLE_DEVICES"] = "0"

using CUDA

println("CUDA functional: ", CUDA.functional())
println("Device: ", CUDA.device())


using Flux
device = CUDA.functional() ? gpu : cpu

using JLD2: @load
using LinearAlgebra
using Random
using Statistics

using ArgParse


function parse_commandline()
    args = ArgParseSettings()
    @add_arg_table args begin

        "--kappa", "-k"
            arg_type = Float64

        "--lambda", "-l"
            arg_type = Float64

        "--DType", "-T"
            arg_type = String
            default = "Float32"


        "--epochs", "-e"
            arg_type = Int
            default = 100

        "--batchsize", "-b"
            arg_type = Int 
            default = 64 
        
        "--depth", "-d"
            arg_type = Int  
            default = 1

        "--nodes", "-n"
            arg_type = Int  
            default = 1

        "--learning_rate"
            arg_type = Float64
            default = 1e-3

        "--split"
            arg_type = Float64
            default = 0.75

        "--model_name"
            arg_type = String
            default = "MLP"

        
    end

    return parse_args(args)
end

###

L1 =32
L2 = 8
b = 8
T = pargs["T"]
kappa= pargs["kappa"]
lambda= pargs["lambda"]
max_epochs = pargs["epochs"]

depth = pargs["depth"]
nodes = pargs["nodes"]
split = pargs["split"]
lr = pargs["learning_rate"]
model_name = pargs["model_name"]


batchsize = pargs["batchsize"]



D = L1*L2

@load "./2d_l$(lambda)_k$(kappa)_L_$(L1)_$(L2).jld2" pics 
pics = convert(Array{T,4}, pics)
N = size(pics,4)

maxiter =  div(N, batchsize) -1

train_indices = randperm(N)[1:Int(floor(split * N))]
test_indices = setdiff(1:N, train_indices)

prior = pics[:, :, :, train_indices] |> device
prior_test = pics[:, :, :, test_indices] |> device
N_train = size(prior, 4)
N_test = size(prior_test, 4)


###

model = Chain(
    x -> reshape(x, (D, size(x, 4))),
    Dense(D, D, relu),
    Dense(D, D),
    x -> reshape(x, (L1, L2, b, size(x, 2)))
) |> device

###

loss(m, x, y) = Flux.mse(m(x), y)

opt = Adam(1e-3)

#  NEW required state object
state = Flux.setup(opt, model)

for epoch in 1:50

    #  NEW gradient style (NO Params)
    grads = gradient(model) do m
        loss(m, prior, prior)
    end

    #  correct update call
    Flux.update!(state, model, grads[1])

    println("epoch $epoch loss = ", loss(model, prior, prior))
end