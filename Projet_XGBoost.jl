### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 37c64d4d-44d6-4ed8-88db-5bdac05bbece
using Pkg;Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

# ╔═╡ 4a6be16e-810f-11ed-1d33-c3430b52b428
using OpenML, MLJ, MLJXGBoostInterface, DataFrames, MLJLinearModels, MLJDecisionTreeInterface, Serialization, CSV, MLCourse, Distributions, Plots, Random, Statistics, Distributions, Distances, LinearAlgebra, CategoricalArrays


# ╔═╡ 6c88f9db-8d5c-452a-a7e8-1e3d609c9d5b
begin
	clean_data = deserialize("clean_data_train.dat")
	coerce!(clean_data, :labels => Multiclass)
end

# ╔═╡ 44dc7ef1-3211-4ad6-b9a5-13098f782a6d
test_data = deserialize("clean_data_test.dat")

# ╔═╡ cdfd7f58-07f7-4f5d-aa9d-fc39e59b3cf3
md"
```julia
#XGBoost

Here is the list of the range we tried for every hyperparameter for the XGBoost :

	- eta : from 0 to 1.
	- num_round : from 1 to 100.
	- max_depth : from 2 to 50.

We also tried to tune the gamma parameter, but it has no impact on our predictions.
```
"

# ╔═╡ c9c6585f-81b0-4fa3-a3dc-c5c72a0d6f75
begin
	model = XGBoostClassifier(booster = "gblinear")
	self_tuning_XGB_model = TunedModel(model = model,
	                            resampling = Holdout(fraction_train=0.8),
	                            tuning = Grid(goal = 8),
								range = [
								range(model, :eta, values=[0.1, 0.05]),
								range(model, :num_round, values=[1, 5]),
								range(model, :max_depth, values=[10, 15])
								], 
								measure = accuracy)
	
	self_tune_mach = machine(self_tuning_XGB_model, select(clean_data, Not(:labels)), 						clean_data.labels)
	
	fit!(self_tune_mach, verbosity = 0)
end

# ╔═╡ e65d5410-eef6-497d-94c1-d70dc3b53615
report(self_tune_mach)

# ╔═╡ 338d3c05-834c-4d61-9027-53ea9025ecd9
plot(self_tune_mach)

# ╔═╡ d9278177-2427-496e-92e8-0f58a04e686e
XGB_prediction = predict_mode(self_tune_mach, test_data)

# ╔═╡ a4b28159-5367-49ca-b89d-652116197b3f
#KAGGLE :
begin 
	index = []
	for i in 1:3093
		push!(index,i)
	end
	kaggle_XGBoost = DataFrame((id = index, prediction = XGB_prediction))
	CSV.write(pwd()*"\\res_prediction_XGBoost.csv", kaggle_XGBoost)
end

# ╔═╡ Cell order:
# ╠═37c64d4d-44d6-4ed8-88db-5bdac05bbece
# ╠═4a6be16e-810f-11ed-1d33-c3430b52b428
# ╠═6c88f9db-8d5c-452a-a7e8-1e3d609c9d5b
# ╠═44dc7ef1-3211-4ad6-b9a5-13098f782a6d
# ╟─cdfd7f58-07f7-4f5d-aa9d-fc39e59b3cf3
# ╠═c9c6585f-81b0-4fa3-a3dc-c5c72a0d6f75
# ╠═e65d5410-eef6-497d-94c1-d70dc3b53615
# ╠═338d3c05-834c-4d61-9027-53ea9025ecd9
# ╠═d9278177-2427-496e-92e8-0f58a04e686e
# ╠═a4b28159-5367-49ca-b89d-652116197b3f
