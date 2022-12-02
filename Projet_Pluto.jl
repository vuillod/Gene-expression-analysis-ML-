### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ d40d9730-7092-11ed-337f-dff4b950f6df
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

# ╔═╡ 0105ef37-4764-409f-a8cb-1b6731af49ca
using DataFrames, CSV, MLJ, MLJLinearModels, MLCourse, Distributions, Plots,  Random, OpenML, Statistics

# ╔═╡ fe75abdd-33a2-444a-86f6-23f1884f091c
data_init = CSV.read("C:\\Users\\nicol\\Desktop\\Project_ML\\train.csv", DataFrame)

# ╔═╡ bc909244-39e8-4cac-aecb-fadccc341d53
data_clean = data_init[:, std.(eachcol(data_init)) .!= 0]

# ╔═╡ 13da3afc-fe7b-4897-8deb-d87ec7fa796a
findall(≈(1), cor(Matrix(data_init))) |> 
idxs -> filter(x -> x[1] > x[2], idxs)

# ╔═╡ 796ecca2-4363-474c-b449-93247a5ed697
begin
	data_init_train1 = data_init[1:4000, :]
	data_init_train2 = data_init[4001:end, :]
end

# ╔═╡ 676e6c06-c207-450d-b8b5-a8a50a4b39ca
begin
	data_init_train1.labels = categorical(data_init_train1.labels, levels=["KAT5","eGFP","CBP"], ordered = true)
    m = machine(LogisticClassifier(penalty = :none),
                 select(data_init_train1, Not(:labels)),
                 data_init_train1.labels)
    fit!(m, verbosity = 0);
end

# ╔═╡ d8943fd5-8553-47a3-b202-3e755e3bf2bb
begin
	data_init_train2.labels = categorical(data_init_train2.labels, levels=["KAT5","eGFP","CBP"], ordered = true)
	confusion_matrix(predict_mode(m, select(data_init_train2, Not(:labels))), data_init_train2.labels)
end

# ╔═╡ 9fcbf92b-8c60-4417-8093-8d7ac5bde6b9
function losses(machine, input, response)
    (negative_loglikelihood = sum(log_loss(predict(machine, input), response)),
     misclassification_rate = mean(predict_mode(machine, input) .!= response),
     accuracy = accuracy(predict_mode(machine, input), response),
     auc = auc(predict(machine, input), response)
	)
end;

# ╔═╡ b8a31745-8eee-44d9-b3fb-ac6671f5d796
let
	fprs1, tprs1, _ = roc_curve(predict(m), data_init_train1.labels)
    fprs2, tprs2, _ = roc_curve(predict(m, select(data_init_train2, Not(:labels))), data_init_train2.labels)
	plot(fprs1, tprs1, label = "training ROC")
	plot!(fprs2, tprs2, label = "test ROC", legend = :bottomright)
end

# ╔═╡ 6a3485e3-37bc-4674-9707-96301146719e
losses(m, select(data_init_train1, Not(:labels)), data_init_train1.labels)

# ╔═╡ 195eea5a-df79-47c7-be3d-6ee3c0ccd5d0
losses(m, select(data_init_train2, Not(:labels)), data_init_train2.labels)

# ╔═╡ Cell order:
# ╠═d40d9730-7092-11ed-337f-dff4b950f6df
# ╠═0105ef37-4764-409f-a8cb-1b6731af49ca
# ╠═fe75abdd-33a2-444a-86f6-23f1884f091c
# ╠═bc909244-39e8-4cac-aecb-fadccc341d53
# ╠═13da3afc-fe7b-4897-8deb-d87ec7fa796a
# ╠═796ecca2-4363-474c-b449-93247a5ed697
# ╠═676e6c06-c207-450d-b8b5-a8a50a4b39ca
# ╠═d8943fd5-8553-47a3-b202-3e755e3bf2bb
# ╠═9fcbf92b-8c60-4417-8093-8d7ac5bde6b9
# ╠═b8a31745-8eee-44d9-b3fb-ac6671f5d796
# ╠═6a3485e3-37bc-4674-9707-96301146719e
# ╠═195eea5a-df79-47c7-be3d-6ee3c0ccd5d0
