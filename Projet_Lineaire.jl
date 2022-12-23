### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ a001cc30-771e-11ed-0bc5-a37734fc6a0e
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

# ╔═╡ e26ffe18-fe92-48ba-859e-6e7fd5c1af11
using DataFrames, CSV, MLJ, MLJLinearModels, MLCourse, Distributions, Plots,  Random, OpenML, Statistics, Serialization 

# ╔═╡ ff7be794-4b9b-4192-bc00-272548c19a3d
begin 
	clean_data = deserialize("clean_data_train.dat")
	coerce!(clean_data, :labels => Multiclass)
end

# ╔═╡ 5d06d671-0832-4339-91d9-ec830076a2f6
test_data = deserialize("clean_data_test.dat")

# ╔═╡ fc83ce73-bd06-44e3-aa19-6ea46a834f0d
md"
```julia
#First step to find the best lambda.

function tune_model_labels(model, data,)
	tuned_model = TunedModel(model = model,
	                         resampling = CV(nfolds = 10),
	                         tuning = Grid(goal=20),
	                         range = range(model, :lambda, scale = :log10,
														upper = 1e-1, lower = 1e-20),
	                         measure = accuracy)
	self_tuned_mach = machine(tuned_model, select(data, Not(:labels)), data.labels)
	fit!(self_tuned_mach, verbosity = 2)
end
```
"

# ╔═╡ 75fbcf05-dcec-48f9-978c-77f4918995bb
md"
```julia
#Second step to find the lambda, by using boundaries closer from the value found above

function tune_model_labels(model, data,)
	tuned_model = TunedModel(model = model,
	                         resampling = CV(nfolds = 10),
	                         tuning = Grid(goal=25),
	                         range = range(model, :lambda, scale = :log10,
														upper = 1e-6, lower = 1e-8),
	                         measure = accuracy)
	self_tuned_mach = machine(tuned_model, select(data, Not(:labels)), data.labels)
	fit!(self_tuned_mach, verbosity = 2)
end

#Repeat it again and again...

```
"

# ╔═╡ 899754cf-7d00-4633-a83f-fcc973e8fc0e
#Model with the best lambda found

function tune_model_labels(model, data,)
	tuned_model = TunedModel(model = model,
	                         resampling = CV(nfolds = 10),
	                         tuning = Grid(goal=1),
	                         range = range(model, :lambda, values= [9.5875e-7]),
	                         measure = accuracy)
	self_tuned_mach = machine(tuned_model, select(data, Not(:labels)), data.labels)
	fit!(self_tuned_mach, verbosity = 2)
end

# ╔═╡ 78fba11f-db27-4282-895a-f528b54980d1
md"
```julia
#Ridge classifier
#Good to precise that the lambda was tuned for the Lasso only

ridge_res = tune_model_labels(LogisticClassifier(penalty = :l2), clean_data)
```
"

# ╔═╡ 37ec27ef-2489-4832-9ce9-0aa192513058
md"
```julia
fitted_params(ridge_res)
```
"




# ╔═╡ 60fa9f19-afad-42b2-8232-28d2844c4726
md"
```julia
plot(ridge_res)
```
"



# ╔═╡ e9b7a7a2-4d2e-4b14-8e4e-456fdbc98497
md"
```julia 
#KAGGLE :
begin 
	ridge_predict = predict_mode(ridge_res,test_data)
	kaggle_ridge = DataFrame(id=index[:], prediction = ridge_predict)
	CSV.write(pwd()*\"res_predictions_ridge.csv\", kaggle_ridge)
end
```
"

# ╔═╡ 377ca4d4-a65b-4737-8cbb-4ce9c42e7f16
#LASSO CLASSIFIER :

# ╔═╡ 1cf6a8ae-0d21-44f2-a90f-bb48790af092
lasso_res = tune_model_labels(LogisticClassifier(penalty = :l1), clean_data)

# ╔═╡ bd710df3-15aa-4d4e-9790-17651f8742a3
plot(lasso_res)

# ╔═╡ 79641c51-0d48-4735-83ce-ffd97457d2c4
fitted_params(lasso_res)

# ╔═╡ 8e1bf001-ea47-42df-8f74-87c70b96eba5
report(lasso_res)

# ╔═╡ 782ebfd3-34fd-4583-adf6-4d1e14c8d782
#KAGGLE :
begin
	index = []
	for i in 1:3093
		push!(index,i)
	end
end

# ╔═╡ 521c3dc7-7d33-4fb5-9b6d-3b1e88cde101
begin 
	lasso_predict = predict_mode(lasso_res,test_data)
	kaggle_lasso = DataFrame(id=index[:], prediction = lasso_predict)
	CSV.write(pwd()*"\\res_predictions_lasso.csv",kaggle_lasso)
end

# ╔═╡ Cell order:
# ╠═a001cc30-771e-11ed-0bc5-a37734fc6a0e
# ╠═e26ffe18-fe92-48ba-859e-6e7fd5c1af11
# ╠═ff7be794-4b9b-4192-bc00-272548c19a3d
# ╠═5d06d671-0832-4339-91d9-ec830076a2f6
# ╟─fc83ce73-bd06-44e3-aa19-6ea46a834f0d
# ╟─75fbcf05-dcec-48f9-978c-77f4918995bb
# ╠═899754cf-7d00-4633-a83f-fcc973e8fc0e
# ╟─78fba11f-db27-4282-895a-f528b54980d1
# ╟─37ec27ef-2489-4832-9ce9-0aa192513058
# ╟─60fa9f19-afad-42b2-8232-28d2844c4726
# ╟─e9b7a7a2-4d2e-4b14-8e4e-456fdbc98497
# ╠═377ca4d4-a65b-4737-8cbb-4ce9c42e7f16
# ╠═1cf6a8ae-0d21-44f2-a90f-bb48790af092
# ╠═bd710df3-15aa-4d4e-9790-17651f8742a3
# ╠═79641c51-0d48-4735-83ce-ffd97457d2c4
# ╠═8e1bf001-ea47-42df-8f74-87c70b96eba5
# ╠═782ebfd3-34fd-4583-adf6-4d1e14c8d782
# ╠═521c3dc7-7d33-4fb5-9b6d-3b1e88cde101
