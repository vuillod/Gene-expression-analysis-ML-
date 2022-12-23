### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 476fa2e0-77c3-11ed-38cd-35798ab207b6
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

# ╔═╡ 23c162c2-93fd-4cd6-8bf3-43ce14e2a67f
using DataFrames, CSV, MLJ, MLJLinearModels, Optim, MLCourse, Distributions, Plots,  Random, OpenML, Statistics, Serialization, MLJFlux, Flux

# ╔═╡ f1b881c4-5798-4043-af58-b0cbd1c7096e
begin 
	clean_data = deserialize("clean_data_train.dat")
	coerce!(clean_data, :labels => Multiclass)
end

# ╔═╡ 509646d3-018e-4f51-992f-0ad1aee2b49e
test_data = deserialize("clean_data_test.dat")

# ╔═╡ e954a626-e3e5-43b6-ba44-44a66f1b3f7e
# Builder for NeuralNetworkClassifier with two hidden layers :

begin
	mutable struct MyBuilder <:  MLJFlux.Builder
    	n_hidden1::Int
    	n_hidden2::Int     # if zero use geometric mean of input/output
    	dropout::Float64
    	σ
	end
	MyBuilder(; n_hidden1=10,n_hidden2=10, dropout=0.5, σ=Flux.relu) = MyBuilder(n_hidden1,n_hidden2, dropout, σ)
	function MLJFlux.build(builder::MyBuilder, rng, n_in, n_out)
		Random.seed!(0)
    	init=Flux.glorot_uniform(rng)
    	return Flux.Chain(
        	Flux.Dense(n_in, builder.n_hidden1, builder.σ, init=init),
        	Flux.Dropout(builder.dropout),
        	Flux.Dense(builder.n_hidden1, builder.n_hidden2, builder.σ, init=init),
        	Flux.Dropout(builder.dropout),
        	Flux.Dense(builder.n_hidden2, n_out, init=init))
	end
end


# ╔═╡ 00f753e5-53f7-4e27-b5f8-4bc120cc72f0
md"
```julia
#Two layers NN :

Here is the list of the range we tried for every hyperparameter for the NN with two hidden layer :

	- epochs : from 10 to 1000.
	- dropout : from 0.5 to 1. 
	- n_hidden1 : from 30 to 400. 
	- n_hidden2 : from 20 to 80 
	- lambda : from  1e-5 to 1e-15 (scale = :log10)
```
"

# ╔═╡ 9eb88425-48ad-45af-b564-031bcbd111e1

begin
	model2 = NeuralNetworkClassifier(builder = MyBuilder(σ=Flux.relu),
    								optimiser = ADAM(),
                             		batch_size = 32, 
									alpha = 1)
                       
	tuned_model = TunedModel(model = model2,
							  	resampling = Holdout(fraction_train = 0.8),
								tuning = Grid(goal=5),                      
								range = [
									range(model2,
									:(builder.n_hidden1), values=[250]),
									range(model2,
									:(builder.n_hidden2), values=[40]),
						            range(model2, :lambda, values = [1.87382e-9]),
								    range(model2,
										  :epochs,
									    values = [30]),
									range(model2, :(builder.dropout), values=[0.7])
										],
	                          measure = accuracy)
	neuron_mach = machine(tuned_model,
	                     select(clean_data, Not(:labels)), clean_data.labels)
	fit!(neuron_mach, verbosity = 2)
end


# ╔═╡ c9e826ca-716e-4b19-a4b2-e0e87fa7b9a3
md"
```julia
#One layer NN :

Here is the list of the range we tried for every hyperparameter for the NN with two hidden layer :

	- epochs : from 10 to 1000.
	- dropout : from 0.5 to 1.
	- n_hidden1 : from 100 to 400.
	- lambda : from  1e-5 to 1e-15 (scale = :log10)
```
"

# ╔═╡ 83c54600-2bcb-441d-ab0b-54b67e5372c3
md"
```julia
#Neural Network with one hidden layer, with some tuning values.

begin 
	model2 = NeuralNetworkClassifier(builder =  MLJFlux.Short(dropout = 0.7,
                                                      σ = relu),
    								optimiser = ADAM(),
                             		batch_size = 50, 
									alpha = 1)
	
	tuned_model = TunedModel(model = model2,
							  	resampling = Holdout(fraction_train = 0.8),
								tuning = Grid(goal=2),
								range = [
										range (model2, :(builder.n_hidden),     
 										values = [100, 200, 300, 400]),
										range(model2, :lambda, scale = :log10,   
 										lower = 1e-9, upper = 1e-8),
								        range(model2, :epochs,
									    values = [50, 100, 150])
										],
	                          measure = accuracy)
	neuron_mach = machine(tuned_model,
	                     select(clean_data, Not(:labels)), clean_data.labels)
	fit!(neuron_mach, verbosity = 2)
end
```
"

# ╔═╡ b41d3380-e157-4587-9a44-1cbc56ce5939
neuron_res = predict_mode(neuron_mach, test_data)

# ╔═╡ 6044a3c6-33b3-40af-9c8d-66c9ee14b24e
report(neuron_mach)

# ╔═╡ a610da8c-d0ee-40e7-92c4-f8e0d0f245ac
plot(neuron_mach)

# ╔═╡ 7b31c8ec-1293-48fb-8aba-80a947a974c0
#KAGGLE SUBMISSION

# ╔═╡ 72975521-b08e-46db-bdff-a13cf93e752d
begin
	index = []
	for i in 1:3093
		push!(index,i)
	end
	kaggle_neuron = DataFrame(id=index[:], prediction = neuron_res)
	CSV.write(pwd()*"\\res_predictions_neurons.csv", kaggle_neuron)
end

# ╔═╡ Cell order:
# ╠═476fa2e0-77c3-11ed-38cd-35798ab207b6
# ╠═23c162c2-93fd-4cd6-8bf3-43ce14e2a67f
# ╠═f1b881c4-5798-4043-af58-b0cbd1c7096e
# ╠═509646d3-018e-4f51-992f-0ad1aee2b49e
# ╠═e954a626-e3e5-43b6-ba44-44a66f1b3f7e
# ╟─00f753e5-53f7-4e27-b5f8-4bc120cc72f0
# ╠═9eb88425-48ad-45af-b564-031bcbd111e1
# ╟─c9e826ca-716e-4b19-a4b2-e0e87fa7b9a3
# ╟─83c54600-2bcb-441d-ab0b-54b67e5372c3
# ╠═b41d3380-e157-4587-9a44-1cbc56ce5939
# ╠═6044a3c6-33b3-40af-9c8d-66c9ee14b24e
# ╠═a610da8c-d0ee-40e7-92c4-f8e0d0f245ac
# ╠═7b31c8ec-1293-48fb-8aba-80a947a974c0
# ╠═72975521-b08e-46db-bdff-a13cf93e752d
