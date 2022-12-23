### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ a1495970-7bd6-11ed-2e9f-4561ff921b5c
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

# ╔═╡ 7b2c39e7-3018-4a5e-8abd-23e740999b79
using DataFrames, CSV, MLJ, MLJLinearModels, Optim, MLCourse, Distributions, Plots,  Random, OpenML, Statistics, Serialization ,MLJDecisionTreeInterface, MLJXGBoostInterface

# ╔═╡ 90af01bf-9cf8-4674-aa9e-e7dc5057e8af
begin
	test_data = deserialize("clean_data_test.dat")
	clean_data = deserialize("clean_data_train.dat")
	coerce!(clean_data, :labels => Multiclass)
end

# ╔═╡ d9eb6601-f300-4650-af5a-02af48b30f37
function tune_model_labels(model, data,)
	Random.seed!(0)
	tuned_model = TunedModel(model = model,
							tuning = Grid(goal = 147),
	                         resampling = Holdout(fraction_train = 0.8),
	                         range  = [
									range(model,:max_depth,values=[18,20,22,24,26,28,30]), 
									range(model,:sampling_fraction,values=[0.48,0.52,0.56,0.60,0.64,0.68,0.72]), 
									range(model,:n_trees,values=[100, 500, 1000]),
									range(model,:min_samples_split ,values=[6])
							 ],
	                         measure = accuracy)
	self_tuned_mach = machine(tuned_model, select(data, Not(:labels)), data.labels)
	fit!(self_tuned_mach, verbosity = 2)
end

# ╔═╡ a7407962-4a7a-461c-b743-d2f3abeb28db
tune_model_forest = tune_model_labels(MLJDecisionTreeInterface.RandomForestClassifier(), clean_data)

# ╔═╡ b3b3e483-6991-4c2d-8f3f-ca6dd0e51dd7
report(tune_model_forest)

# ╔═╡ 919beb15-5974-40b4-9f0b-5e6cadc4a574
plot(tune_model_forest)

# ╔═╡ 9bfd5f81-41c0-43db-a579-0b0ef13707ed
#KAGGLE

begin
	index = []
	for i in 1:3093
		push!(index,i)
	end
	forest_predict = predict_mode(tune_model_forest,test_data)
	kaggle_forest = DataFrame(id=index[:], prediction = forest_predict)
	CSV.write(pwd()*"\\res_predictions_forest.csv",kaggle_forest)
end

# ╔═╡ 41cafe71-7e1c-48e5-b2f9-d1d446cfcefa


# ╔═╡ Cell order:
# ╠═a1495970-7bd6-11ed-2e9f-4561ff921b5c
# ╠═7b2c39e7-3018-4a5e-8abd-23e740999b79
# ╠═90af01bf-9cf8-4674-aa9e-e7dc5057e8af
# ╠═d9eb6601-f300-4650-af5a-02af48b30f37
# ╠═a7407962-4a7a-461c-b743-d2f3abeb28db
# ╠═b3b3e483-6991-4c2d-8f3f-ca6dd0e51dd7
# ╠═919beb15-5974-40b4-9f0b-5e6cadc4a574
# ╠═9bfd5f81-41c0-43db-a579-0b0ef13707ed
# ╠═41cafe71-7e1c-48e5-b2f9-d1d446cfcefa
