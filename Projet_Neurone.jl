### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 476fa2e0-77c3-11ed-38cd-35798ab207b6
begin 
	using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
	import Pkg; Pkg.add("Optim")
end

# ╔═╡ 23c162c2-93fd-4cd6-8bf3-43ce14e2a67f
using DataFrames, CSV, MLJ, MLJLinearModels, Optim, MLCourse, Distributions, Plots,  Random, OpenML, Statistics, Serialization, MLJFlux, Flux, OpenML

# ╔═╡ f1b881c4-5798-4043-af58-b0cbd1c7096e
begin 
	clean_data = deserialize("clean_data_train.dat")
	coerce!(clean_data, :labels => Multiclass)
end

# ╔═╡ 509646d3-018e-4f51-992f-0ad1aee2b49e
test_data = deserialize("clean_data_test.dat")

# ╔═╡ 3778363e-9b22-4197-8aec-943f1224f0a7
size(test_data)

# ╔═╡ 9eb88425-48ad-45af-b564-031bcbd111e1
begin
	model2 = NeuralNetworkClassifier(builder = MLJFlux.Short(σ = relu),
    								optimiser = ADAM(),
                             		batch_size = 200)
                       
	tuned_model = TunedModel(model = model2,
							  resampling = CV(nfolds = 10),
	                          range = [range(model2,
						                :(neural_network_classifier.builder.dropout),
									    values = [0, .1, .2]),
								       range(model2,
									     :(neural_network_classifier.epochs),
									     values = [500, 1000]),
							  		range(model2, 
									:(neural_network_classifier.builder.n_hidden), values = [50, 150])],
	                          measure = accuracy)
	neuron_mach = fit!(machine(tuned_model,
	                     select(clean_data, Not(:labels)), verbosity = 0))
end

# ╔═╡ b41d3380-e157-4587-9a44-1cbc56ce5939
neuron_res = predict_mode(neuron_mach, test_data)

# ╔═╡ 6044a3c6-33b3-40af-9c8d-66c9ee14b24e
plot(neuron_res)

# ╔═╡ 7b31c8ec-1293-48fb-8aba-80a947a974c0
#KAGGLE SUBMISSION

# ╔═╡ 72975521-b08e-46db-bdff-a13cf93e752d
begin
	index = []
	for i in 1:3093
		push!(index,i)
	end
	kaggle_neuron = DataFrame(id=index[:], prediction = ridge_predict)
	CSV.write(pwd()*"\\res_predictions_neuron.csv",kaggle_neuron)
end

# ╔═╡ 3010ebcd-5bab-4413-b0ca-773daf7a6504


# ╔═╡ Cell order:
# ╠═476fa2e0-77c3-11ed-38cd-35798ab207b6
# ╠═23c162c2-93fd-4cd6-8bf3-43ce14e2a67f
# ╠═f1b881c4-5798-4043-af58-b0cbd1c7096e
# ╠═509646d3-018e-4f51-992f-0ad1aee2b49e
# ╠═3778363e-9b22-4197-8aec-943f1224f0a7
# ╠═9eb88425-48ad-45af-b564-031bcbd111e1
# ╠═b41d3380-e157-4587-9a44-1cbc56ce5939
# ╠═6044a3c6-33b3-40af-9c8d-66c9ee14b24e
# ╠═7b31c8ec-1293-48fb-8aba-80a947a974c0
# ╠═72975521-b08e-46db-bdff-a13cf93e752d
# ╠═3010ebcd-5bab-4413-b0ca-773daf7a6504
