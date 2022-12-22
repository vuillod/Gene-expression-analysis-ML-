### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ a1495970-7bd6-11ed-2e9f-4561ff921b5c
begin 
	using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
	import Pkg; Pkg.add("Optim")
end

# ╔═╡ 7b2c39e7-3018-4a5e-8abd-23e740999b79
using DataFrames, CSV, MLJ, MLJLinearModels, Optim, MLCourse, Distributions, Plots,  Random, OpenML, Statistics, Serialization ,MLJDecisionTreeInterface, MLJXGBoostInterface

# ╔═╡ 90af01bf-9cf8-4674-aa9e-e7dc5057e8af
begin
	test_data = deserialize("clean_data_test.dat")
	clean_data = deserialize("clean_data_train.dat")
	coerce!(clean_data, :labels => Multiclass)
end

# ╔═╡ ac6ec1a2-4a8b-4653-a876-85581c54fd82
solver = MLJLinearModels.LBFGS(optim_options = Optim.Options(time_limit = 100))


# ╔═╡ f58d2f5e-88ef-4e23-b68f-969364b8fdde

#function tune_model_labels(model, data,)
#	tuned_model = TunedModel(model = build_forest(labels, features,
 #                        n_subfeatures,
  #                        n_trees,
   #                       partial_sampling,
    #                      max_depth,
     #                     min_samples_leaf,
      #                    min_samples_split,
       #                   min_purity_increase),
	    #                     resampling = CV(nfolds = 10),
	     #                    measure = accuracy)
	#self_tuned_mach = machine(tuned_model, select(data, Not(:labels)), data.labels)
	
#fit!(self_tuned_mach, verbosity = 2)
#begin
#	modelForest= MLJDecisionTreeInterface.RandomForestClassifier(builder= build_forest(labels, features,
 #                        n_subfeatures,
  #                        n_trees,
   #                       partial_sampling,
    #                      max_depth,
     #                     min_samples_leaf,
      #                    min_samples_split,
       #                   min_purity_increase))
#end
#range(model,:max_depth ,scale =:linear, lower = 1, upper = 50), 
#range(model,:sampling_fraction,scale =:linear,lower = 0.1, upper = 0.5), 
#range(model,:n_trees,scale=:linear, lower = 40, upper = 50),
#range(model,:n_subfeatures,scale=:linear, lower = 1, upper = 30),
#range(model,:min_purity_increase ,scale=:linear, lower = 1, upper = 30),
#range(model,:sampling_fraction ,scale=:linear, lower = 1, upper = 30),
#range(model,:min_samples_split ,scale=:linear, lower = 1, upper = 30)
#resampling = CV(nfolds = 10),


# ╔═╡ d9eb6601-f300-4650-af5a-02af48b30f37
"""
function tune_model_labels(model, data,)
	tuned_model = TunedModel(model = model,
							tuning = Grid(goal = 49),
	                         resampling = Holdout(fraction_train = 0.8),
	                         range  = [
	
range(model,:max_depth,values=[18,20,22,24,26,28,30]), 
range(model,:sampling_fraction,values=[0.48,0.52,0.56,0.60,0.64,0.68,0.72]), 
range(model,:n_trees,values=[1000]),
range(model,:min_samples_split ,values=[6])
								 
							 ],
	                         measure = accuracy)
	self_tuned_mach = machine(tuned_model, select(data, Not(:labels)), data.labels)
	fit!(self_tuned_mach, verbosity = 2)
end
"""

# ╔═╡ a7407962-4a7a-461c-b743-d2f3abeb28db
"""
tune_model_forest = tune_model_labels(MLJDecisionTreeInterface.RandomForestClassifier(), clean_data)
"""

# ╔═╡ b3b3e483-6991-4c2d-8f3f-ca6dd0e51dd7
#report(tune_model_forest)

# ╔═╡ 919beb15-5974-40b4-9f0b-5e6cadc4a574
#plot(tune_model_forest)

# ╔═╡ 9194583c-2de2-4118-abd6-6d0ceb08c3f9
Random.seed!(0)

# ╔═╡ edcc9e45-e3b4-4a89-8f2f-ae6ad20505cc
begin
	model = MLJXGBoostInterface.XGBoostClassifier(booster = "gblinear")
	self_tuning_XGB_model = TunedModel(model = model,
	                            resampling = Holdout(fraction_train=0.8),
	                            tuning = Grid(goal = 4),
								range = [
								range(model, :eta, values=[0.8]),
								range(model, :num_round, values=[25]),
								range(model, :max_depth, values=[5]),
								range(model, :gamma, values=[5])
								], 
								measure = accuracy)
	#VALEUR DE GAMMA A CHANGER !!!
	self_tune_mach = machine(self_tuning_XGB_model, select(clean_data, Not(:labels)), 						clean_data.labels)
	fit!(self_tune_mach, verbosity = 0)
end

# ╔═╡ 22c92a9e-57f0-4ebf-8e45-259a1610d7c1
report(self_tune_mach)

# ╔═╡ 1c644d6e-62ae-4808-be1c-7b4d69422ae5
plot(self_tune_mach)

# ╔═╡ 013cd07a-9e31-444e-b3f3-85747c37df36
XGB_prediction = predict_mode(self_tune_mach, test_data)

# ╔═╡ 9bfd5f81-41c0-43db-a579-0b0ef13707ed
begin
	index = []
	for i in 1:3093
		push!(index,i)
	end
	kaggle_XGBoost = DataFrame((id = index, prediction = XGB_prediction))
	CSV.write(pwd()*"\\res_prediction_XGBoost.csv", kaggle_XGBoost)
end

# ╔═╡ 41cafe71-7e1c-48e5-b2f9-d1d446cfcefa
"""
begin
	forest_predict = predict_mode(tune_model_forest,test_data)
	kaggle_forest = DataFrame(id=index[:], prediction = forest_predict)
	CSV.write(pwd()*"\\res_predictions_1000arbres_forest.csv",kaggle_forest)
end
"""

# ╔═╡ 45635387-6a39-404f-bbc8-7ab68bd27702
"""
begin
	boost_predict = predict_mode(m2,test_data)
	kaggle_boost = DataFrame(id=index[:], prediction = forest_predict)
	CSV.write(pwd()*"\\res_predictions_forest.csv",kaggle_forest)
end
"""

# ╔═╡ fcf9a243-29ba-45c7-b295-70f714a4d37f
#report(m2)

# ╔═╡ a2de52e2-2ac6-4159-abd1-95c89a2afc42
#plot(m2)

# ╔═╡ Cell order:
# ╠═a1495970-7bd6-11ed-2e9f-4561ff921b5c
# ╠═7b2c39e7-3018-4a5e-8abd-23e740999b79
# ╠═90af01bf-9cf8-4674-aa9e-e7dc5057e8af
# ╠═ac6ec1a2-4a8b-4653-a876-85581c54fd82
# ╠═f58d2f5e-88ef-4e23-b68f-969364b8fdde
# ╠═d9eb6601-f300-4650-af5a-02af48b30f37
# ╠═a7407962-4a7a-461c-b743-d2f3abeb28db
# ╠═b3b3e483-6991-4c2d-8f3f-ca6dd0e51dd7
# ╠═919beb15-5974-40b4-9f0b-5e6cadc4a574
# ╠═9194583c-2de2-4118-abd6-6d0ceb08c3f9
# ╠═edcc9e45-e3b4-4a89-8f2f-ae6ad20505cc
# ╠═22c92a9e-57f0-4ebf-8e45-259a1610d7c1
# ╠═1c644d6e-62ae-4808-be1c-7b4d69422ae5
# ╠═013cd07a-9e31-444e-b3f3-85747c37df36
# ╠═9bfd5f81-41c0-43db-a579-0b0ef13707ed
# ╠═41cafe71-7e1c-48e5-b2f9-d1d446cfcefa
# ╠═45635387-6a39-404f-bbc8-7ab68bd27702
# ╠═fcf9a243-29ba-45c7-b295-70f714a4d37f
# ╠═a2de52e2-2ac6-4159-abd1-95c89a2afc42
