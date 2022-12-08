### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ d40d9730-7092-11ed-337f-dff4b950f6df
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

# ╔═╡ 0105ef37-4764-409f-a8cb-1b6731af49ca
using DataFrames, CSV, MLJ, MLJLinearModels, MLCourse, Distributions, Plots,  Random, OpenML, Statistics, Serialization

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

# ╔═╡ be344bfb-75d9-4d34-b46e-b6b770afff1f
serialize("intermediate_result.dat", clean_data)

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
# ╠═be344bfb-75d9-4d34-b46e-b6b770afff1f
