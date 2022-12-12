### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ a001cc30-771e-11ed-0bc5-a37734fc6a0e
begin 
	using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
	import Pkg; Pkg.add("Optim")
end

# ╔═╡ e26ffe18-fe92-48ba-859e-6e7fd5c1af11
using DataFrames, CSV, MLJ, MLJLinearModels, Optim, MLCourse, Distributions, Plots,  Random, OpenML, Statistics, Serialization

# ╔═╡ ff7be794-4b9b-4192-bc00-272548c19a3d
begin 
	clean_data = deserialize("clean_data_train.dat")
	coerce!(clean_data, :labels => Multiclass)
end

# ╔═╡ 5d06d671-0832-4339-91d9-ec830076a2f6
test_data = deserialize("clean_data_test.dat")

# ╔═╡ 4aed2913-3743-4280-99b6-6ca841969770
begin
	clean_data_train = clean_data[1:4000, :]
	clean_data_test = clean_data[4001:end, :]
end

# ╔═╡ 3aaeeaa0-4fce-4c7f-8549-051b88d57393
begin
	solver = MLJLinearModels.LBFGS(optim_options = Optim.Options(time_limit = 20))
	model1 = LogisticClassifier(penalty = :l2, lambda = 1e-4, solver = solver)
	clean_data_train.labels = categorical(clean_data_train.labels, levels=["KAT5","eGFP","CBP"], ordered = true)
    m = machine(model1, select(clean_data_train,Not(:labels)),clean_data_train.labels)
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

# ╔═╡ 899754cf-7d00-4633-a83f-fcc973e8fc0e
function tune_model_labels(model, data,)
	tuned_model = TunedModel(model = model,
	                         resampling = CV(nfolds = 10),
	                         tuning = Grid(goal=10),
	                         range = range(model, :lambda,
									       scale = :log10,
									       lower = 1e-30, upper = 1e-1),
	                         measure = accuracy)
	self_tuned_mach = machine(tuned_model, select(data, Not(:labels)), data.labels)
	fit!(self_tuned_mach, verbosity = 2)
end

# ╔═╡ 78fba11f-db27-4282-895a-f528b54980d1
ridge_res = tune_model_labels(LogisticClassifier(penalty = :l2, solver = solver), clean_data)

# ╔═╡ 37ec27ef-2489-4832-9ce9-0aa192513058
fitted_params(ridge_res)

# ╔═╡ 782ebfd3-34fd-4583-adf6-4d1e14c8d782
begin
	index = []
	for i in 1:3093
		push!(index,i)
	end
end

# ╔═╡ 2cdd3568-f8a4-4e0c-b580-ede0eb3d8e49
	size(index)


# ╔═╡ 60fa9f19-afad-42b2-8232-28d2844c4726
plot(ridge_res)

# ╔═╡ 0a806c9c-ae2e-4e7f-af36-e094cd713550


# ╔═╡ 91aced6a-e7eb-4695-aab0-5817e8b5a94a
begin 
	ridge_predict = predict_mode(ridge_res,test_data)
	kaggle_ridge = DataFrame(id=index[:], prediction = ridge_predict)
	CSV.write(pwd()*"\\res_predictions.csv",kaggle_ridge)
end

# ╔═╡ Cell order:
# ╠═a001cc30-771e-11ed-0bc5-a37734fc6a0e
# ╠═e26ffe18-fe92-48ba-859e-6e7fd5c1af11
# ╠═ff7be794-4b9b-4192-bc00-272548c19a3d
# ╠═5d06d671-0832-4339-91d9-ec830076a2f6
# ╠═4aed2913-3743-4280-99b6-6ca841969770
# ╠═3aaeeaa0-4fce-4c7f-8549-051b88d57393
# ╠═075a79cb-78ff-4f30-b66c-97916599794d
# ╠═62347da9-2b93-47e4-9789-b47c3fd0b002
# ╠═2d777b75-0fcf-407b-8cf3-76599ca91275
# ╠═b3151845-b17c-4e01-b89f-0143f152aea9
# ╠═fe0eaa78-7836-4913-9cd0-1597cbdba8e6
# ╠═899754cf-7d00-4633-a83f-fcc973e8fc0e
# ╠═78fba11f-db27-4282-895a-f528b54980d1
# ╠═37ec27ef-2489-4832-9ce9-0aa192513058
# ╠═782ebfd3-34fd-4583-adf6-4d1e14c8d782
# ╠═2cdd3568-f8a4-4e0c-b580-ede0eb3d8e49
# ╠═60fa9f19-afad-42b2-8232-28d2844c4726
# ╠═0a806c9c-ae2e-4e7f-af36-e094cd713550
# ╠═91aced6a-e7eb-4695-aab0-5817e8b5a94a
