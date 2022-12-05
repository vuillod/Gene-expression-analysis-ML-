### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ d40d9730-7092-11ed-337f-dff4b950f6df
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

# ╔═╡ 0105ef37-4764-409f-a8cb-1b6731af49ca
using DataFrames, CSV, MLJ, MLJLinearModels, MLCourse, Distributions, Plots,  Random, OpenML, Statistics

# ╔═╡ fe75abdd-33a2-444a-86f6-23f1884f091c
data_init = CSV.read(pwd() * "\\train.csv", DataFrame)

# ╔═╡ 17978cb9-6c8c-4001-b817-1c74e1d5ceda
#---------------------------------------------------------------------------
#CLEAN THE DATA

# ╔═╡ 43a437a7-310c-46c1-93ca-fc522405d864
begin
	data_without_labels = select(data_init, Not(:labels))
	labels = select(data_init, :labels)
end

# ╔═╡ bc909244-39e8-4cac-aecb-fadccc341d53
new_data_without_labels = data_without_labels[:, std.(eachcol(data_without_labels)) .!= 0]

# ╔═╡ 9c306784-9b0a-4f87-a478-87370409783d
size(new_data_without_labels)

# ╔═╡ 13da3afc-fe7b-4897-8deb-d87ec7fa796a
tab = findall(≈(1), cor(Matrix(new_data_without_labels))) |> 
idxs -> filter(x -> x[1] > x[2], idxs)

# ╔═╡ bddd5feb-cda1-44fd-bd1d-40e2833c122e
size(tab)

# ╔═╡ 53c3b744-d272-430b-8745-fc0cd2e4ae03
begin
	new_t=[]
	for elem in tab
		push!(new_t, elem[1])
	end
	#mtn qu'on a les index, faut enlever les colonnes correspondantes
end

# ╔═╡ ef96207b-2bf5-4433-a538-006a7d0e2e0b
new_data_without_labels2 = select(new_data_without_labels, Not(unique(new_t)))

# ╔═╡ 4829f515-7bec-4ef2-a5cd-50637ee2c711
clean_data = hcat(new_data_without_labels2, labels)

# ╔═╡ 77692985-ce7f-4b1a-acf1-9b25ad5c13f3
#-------------------------------------------------------------------------------
#A modifier pour prendre les data clean

# ╔═╡ 796ecca2-4363-474c-b449-93247a5ed697
begin
	clean_data_train = clean_data[1:4000, :]
	clean_data_test = clean_data[4001:end, :]
end

# ╔═╡ 676e6c06-c207-450d-b8b5-a8a50a4b39ca
begin
	clean_data_train.labels = categorical(clean_data_train.labels, levels=["KAT5","eGFP","CBP"], ordered = true)
    m = machine(LogisticClassifier(penalty = :none),
                 select(clean_data_train, Not(:labels)),
                 clean_data_train.labels)
    fit!(m, verbosity = 0);
end

# ╔═╡ d8943fd5-8553-47a3-b202-3e755e3bf2bb
begin
	clean_data_test.labels = categorical(clean_data_test.labels, levels=["KAT5","eGFP","CBP"], ordered = true)
	confusion_matrix(predict_mode(m, select(clean_data_test, Not(:labels))), clean_data_test.labels)
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
	fprs1, tprs1, _ = roc_curve(predict(m), clean_data_train.labels)
    fprs2, tprs2, _ = roc_curve(predict(m, select(clean_data_test, Not(:labels))), 		clean_data_test.labels)
	plot(fprs1, tprs1, label = "training ROC")
	plot!(fprs2, tprs2, label = "test ROC", legend = :bottomright)
end

# ╔═╡ 6a3485e3-37bc-4674-9707-96301146719e
losses(m, select(clean_data_train, Not(:labels)), clean_data_train.labels)

# ╔═╡ 195eea5a-df79-47c7-be3d-6ee3c0ccd5d0
losses(m, select(clean_data_test, Not(:labels)), clean_data_test.labels)

# ╔═╡ Cell order:
# ╠═d40d9730-7092-11ed-337f-dff4b950f6df
# ╠═0105ef37-4764-409f-a8cb-1b6731af49ca
# ╠═fe75abdd-33a2-444a-86f6-23f1884f091c
# ╠═17978cb9-6c8c-4001-b817-1c74e1d5ceda
# ╠═43a437a7-310c-46c1-93ca-fc522405d864
# ╠═bc909244-39e8-4cac-aecb-fadccc341d53
# ╠═9c306784-9b0a-4f87-a478-87370409783d
# ╠═13da3afc-fe7b-4897-8deb-d87ec7fa796a
# ╠═bddd5feb-cda1-44fd-bd1d-40e2833c122e
# ╠═53c3b744-d272-430b-8745-fc0cd2e4ae03
# ╠═ef96207b-2bf5-4433-a538-006a7d0e2e0b
# ╠═4829f515-7bec-4ef2-a5cd-50637ee2c711
# ╠═77692985-ce7f-4b1a-acf1-9b25ad5c13f3
# ╠═796ecca2-4363-474c-b449-93247a5ed697
# ╠═676e6c06-c207-450d-b8b5-a8a50a4b39ca
# ╠═d8943fd5-8553-47a3-b202-3e755e3bf2bb
# ╠═9fcbf92b-8c60-4417-8093-8d7ac5bde6b9
# ╠═b8a31745-8eee-44d9-b3fb-ac6671f5d796
# ╠═6a3485e3-37bc-4674-9707-96301146719e
# ╠═195eea5a-df79-47c7-be3d-6ee3c0ccd5d0
