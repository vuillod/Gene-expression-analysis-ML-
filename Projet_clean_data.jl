### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ d40d9730-7092-11ed-337f-dff4b950f6df
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

# ╔═╡ 0105ef37-4764-409f-a8cb-1b6731af49ca
using DataFrames, CSV, MLJ, MLJLinearModels, MLCourse, Distributions, Plots,  Random, OpenML, Statistics, Serialization, MLJMultivariateStatsInterface

# ╔═╡ fe75abdd-33a2-444a-86f6-23f1884f091c
data_init = CSV.read(pwd() * "\\train.csv", DataFrame)

# ╔═╡ 80136d53-d343-4f6a-bd67-d2824ef6d70e
test_init = CSV.read(pwd() * "\\test.csv", DataFrame)

# ╔═╡ 17978cb9-6c8c-4001-b817-1c74e1d5ceda
#---------------------------------------------------------------------------
#CLEAN THE DATA

# ╔═╡ 43a437a7-310c-46c1-93ca-fc522405d864
begin
	train_data_without_labels = select(data_init, Not(:labels))
	train_labels = select(data_init, :labels)
end

# ╔═╡ bc909244-39e8-4cac-aecb-fadccc341d53
#CONSTANT PREDICTORS :
new_train_data_without_labels = train_data_without_labels[:, std.(eachcol(train_data_without_labels)) .!= 0]

# ╔═╡ 006a81e3-4a27-4789-8c0b-999d01283865
clean_data_test1 = test_init[:, std.(eachcol(train_data_without_labels)) .!= 0]

# ╔═╡ 13da3afc-fe7b-4897-8deb-d87ec7fa796a
#CORRELATE PREDICTORS :
tab = findall(≈(1), cor(Matrix(new_train_data_without_labels))) |> 
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
begin
	new_train_data_without_labels2 = select(new_train_data_without_labels, Not(unique(new_t)))
	clean_data_test2 = select(clean_data_test1, Not(unique(new_t)))
end

# ╔═╡ c3ff62b1-2fb9-4604-83ef-90eaf70f0606
#---------------------------------------------------------------------------------
#PCA

# ╔═╡ ebf40c1e-4f3d-442f-972e-74f177e0188a
begin
	pca_machine_train_data = fit!(machine(PCA(variance_ratio = 0.975), new_train_data_without_labels2), verbosity = 0)
end

# ╔═╡ 2f8d1e13-2c7e-48f5-9ee0-9b95f78250b9
data_train_without_label_pca = MLJ.transform(pca_machine_train_data, new_train_data_without_labels2)

# ╔═╡ e38f5a19-62bc-48b9-aa1b-2576e17489ba
size(data_train_without_label_pca)

# ╔═╡ 83793f4e-182c-4f73-aa52-11d6933111f6
names(data_train_without_label_pca)

# ╔═╡ 5f6f4c17-794d-4e11-bc8c-2a339083f5d5
data_test_pca= MLJ.transform(pca_machine_train_data, clean_data_test2)

# ╔═╡ 320db0b0-28ec-4dea-a72a-c3df1599ed5c
size(data_test_pca)

# ╔═╡ 4829f515-7bec-4ef2-a5cd-50637ee2c711
clean_data_train = hcat(data_train_without_label_pca, train_labels)

# ╔═╡ be344bfb-75d9-4d34-b46e-b6b770afff1f
begin
	serialize("clean_data_train.dat", clean_data_train)
	serialize("clean_data_test.dat", data_test_pca)
end

# ╔═╡ Cell order:
# ╠═d40d9730-7092-11ed-337f-dff4b950f6df
# ╠═0105ef37-4764-409f-a8cb-1b6731af49ca
# ╠═fe75abdd-33a2-444a-86f6-23f1884f091c
# ╠═80136d53-d343-4f6a-bd67-d2824ef6d70e
# ╠═17978cb9-6c8c-4001-b817-1c74e1d5ceda
# ╠═43a437a7-310c-46c1-93ca-fc522405d864
# ╠═bc909244-39e8-4cac-aecb-fadccc341d53
# ╠═006a81e3-4a27-4789-8c0b-999d01283865
# ╠═13da3afc-fe7b-4897-8deb-d87ec7fa796a
# ╠═bddd5feb-cda1-44fd-bd1d-40e2833c122e
# ╠═53c3b744-d272-430b-8745-fc0cd2e4ae03
# ╠═ef96207b-2bf5-4433-a538-006a7d0e2e0b
# ╠═c3ff62b1-2fb9-4604-83ef-90eaf70f0606
# ╠═ebf40c1e-4f3d-442f-972e-74f177e0188a
# ╠═2f8d1e13-2c7e-48f5-9ee0-9b95f78250b9
# ╠═e38f5a19-62bc-48b9-aa1b-2576e17489ba
# ╠═83793f4e-182c-4f73-aa52-11d6933111f6
# ╠═5f6f4c17-794d-4e11-bc8c-2a339083f5d5
# ╠═320db0b0-28ec-4dea-a72a-c3df1599ed5c
# ╠═4829f515-7bec-4ef2-a5cd-50637ee2c711
# ╠═be344bfb-75d9-4d34-b46e-b6b770afff1f
