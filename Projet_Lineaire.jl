### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ a001cc30-771e-11ed-0bc5-a37734fc6a0e
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

# ╔═╡ e26ffe18-fe92-48ba-859e-6e7fd5c1af11
using DataFrames, CSV, MLJ, MLJLinearModels, Optim, MLCourse, Distributions, Plots,  Random, OpenML, Statistics, Serialization

# ╔═╡ ff7be794-4b9b-4192-bc00-272548c19a3d
clean_data = deserialize("intermediate_result.dat")

# ╔═╡ 4aed2913-3743-4280-99b6-6ca841969770
begin
	clean_data_train = clean_data[1:4000, :]
	clean_data_test = clean_data[4001:end, :]
end

# ╔═╡ 3aaeeaa0-4fce-4c7f-8549-051b88d57393
begin
	solver = MLJLinearModels.LBFGS(optim_options = Optim.Options(time_limit = 60))
	model1 = LogisticClassifier(penalty = :l2, lambda = 1e-4, solver = solver)
	clean_data_train.labels = categorical(clean_data_train.labels, levels=["KAT5","eGFP","CBP"], ordered = true)
    m = machine(model1, select(clean_data_train,Not(:labels)), clean_data_train.labels)
    fit!(m, verbosity = 0);
end

# ╔═╡ 075a79cb-78ff-4f30-b66c-97916599794d
begin
	clean_data_test.labels = categorical(clean_data_test.labels, levels=["KAT5","eGFP","CBP"], ordered = true)
	confusion_matrix(predict_mode(m, select(clean_data_test, Not(:labels))), clean_data_test.labels)
end

# ╔═╡ 62347da9-2b93-47e4-9789-b47c3fd0b002
function losses(machine, input, response)
    (negative_loglikelihood = sum(log_loss(predict(machine, input), response)),
     misclassification_rate = mean(predict_mode(machine, input) .!= response),
     accuracy = accuracy(predict_mode(machine, input), response),
     auc = auc(predict(machine, input), response)
	)
end;

# ╔═╡ 2d777b75-0fcf-407b-8cf3-76599ca91275
let
	fprs1, tprs1, _ = roc_curve(predict(m), clean_data_train.labels)
    fprs2, tprs2, _ = roc_curve(predict(m, select(clean_data_test, Not(:labels))), 		clean_data_test.labels)
	plot(fprs1, tprs1, label = "training ROC")
	plot!(fprs2, tprs2, label = "test ROC", legend = :bottomright)
end

# ╔═╡ b3151845-b17c-4e01-b89f-0143f152aea9
losses(m, select(clean_data_train, Not(:labels)), clean_data_train.labels)

# ╔═╡ fe0eaa78-7836-4913-9cd0-1597cbdba8e6
losses(m, select(clean_data_test, Not(:labels)), clean_data_test.labels)

# ╔═╡ a4789350-e349-47ff-8b38-2363eb49aa6d
begin
    model = RidgeRegressor()
    self_tuning_model = TunedModel(model = model, # the model to be tuned
                                   resampling = CV(nfolds = 100), # how to evaluate
                                   range = range(model, :lambda, scale = :log10,
									       lower = 1e-30, upper = 1e-1), #seebelow
                                   measure = accuracy) # evaluation measure
	
    self_tuning_mach = machine(self_tuning_model, select(clean_data_train, 			 
    Not(:labels)), 	clean_data_train.labels)
	
	fit!(self_tuning_mach, verbosity = 0)
end

# ╔═╡ Cell order:
# ╠═a001cc30-771e-11ed-0bc5-a37734fc6a0e
# ╠═e26ffe18-fe92-48ba-859e-6e7fd5c1af11
# ╠═ff7be794-4b9b-4192-bc00-272548c19a3d
# ╠═4aed2913-3743-4280-99b6-6ca841969770
# ╠═3aaeeaa0-4fce-4c7f-8549-051b88d57393
# ╠═075a79cb-78ff-4f30-b66c-97916599794d
# ╠═62347da9-2b93-47e4-9789-b47c3fd0b002
# ╠═2d777b75-0fcf-407b-8cf3-76599ca91275
# ╠═b3151845-b17c-4e01-b89f-0143f152aea9
# ╠═fe0eaa78-7836-4913-9cd0-1597cbdba8e6
# ╠═a4789350-e349-47ff-8b38-2363eb49aa6d
