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
using DataFrames, CSV, MLJ, MLJLinearModels, Optim, MLCourse, Distributions, Plots,  Random, OpenML, Statistics, Serialization 

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
function tune_model_non_linear(model, data,)
	tuned_model = TunedModel(model = model,
	                         resampling = CV(nfolds = 10),
	                         tuning = Grid(goal=16),
	                         range = [range(model, :lambda,
									       scale = :log10,
									       lower = 1e-20, upper = 1e-1),    ],
	                         measure = accuracy)
	self_tuned_mach = machine(tuned_model, select(data, Not(:labels)), data.labels)
	fit!(self_tuned_mach, verbosity = 2)
end

# ╔═╡ 7b31c8ec-1293-48fb-8aba-80a947a974c0
#KAGGLE SUBMISSION

# ╔═╡ 72975521-b08e-46db-bdff-a13cf93e752d
begin
	index = []
	for i in 1:3093
		push!(index,i)
	end
end

# ╔═╡ 3010ebcd-5bab-4413-b0ca-773daf7a6504


# ╔═╡ Cell order:
# ╠═476fa2e0-77c3-11ed-38cd-35798ab207b6
# ╠═23c162c2-93fd-4cd6-8bf3-43ce14e2a67f
# ╠═f1b881c4-5798-4043-af58-b0cbd1c7096e
# ╠═509646d3-018e-4f51-992f-0ad1aee2b49e
# ╠═3778363e-9b22-4197-8aec-943f1224f0a7
# ╠═9eb88425-48ad-45af-b564-031bcbd111e1
# ╠═7b31c8ec-1293-48fb-8aba-80a947a974c0
# ╠═72975521-b08e-46db-bdff-a13cf93e752d
# ╠═3010ebcd-5bab-4413-b0ca-773daf7a6504
