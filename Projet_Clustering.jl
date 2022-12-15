### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ bd9df9b0-7c9d-11ed-33ca-f317e29085fa
using Pkg;Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

# ╔═╡ 9400c714-064d-450b-bc0e-90b84a597f89
using MLCourse, MLJ, DataFrames, MLJMultivariateStatsInterface, OpenML,
          LinearAlgebra, Statistics, Random, MLJClusteringInterface, StatsPlots,
          Distributions, Distances, Serialization

# ╔═╡ cd5f716a-5122-426a-8998-709313a519f4
begin 
	clean_data = deserialize("clean_data_train.dat")
	coerce!(clean_data, :labels => Multiclass)
end

# ╔═╡ 9f8540f4-db37-4d4b-adf5-0262ec58fccd
test_data = deserialize("clean_data_test.dat")

# ╔═╡ 390f32b4-d1bf-4a9b-9209-fc02c6df31ab
#KMeans :

begin
	KMeans_mach = machine(KMeans(k = 3), select(clean_data, Not(:labels)), clean_data.labels)
    fit!(KMeans_mach, verbosity = 0)
end

# ╔═╡ e8c8d460-4b9d-4a99-9c7a-6712ef4035be
KMeans_predict = predict_mode(KMeans_mach, test_data)

# ╔═╡ 551a5788-2553-4b1e-ac65-fe355e9a5b9c
#Kaggle KMeans :
begin
	index = []
	for i in 1:3093
		push!(index,i)
	end
	kaggle_KMeans = DataFrame(id=index[:], prediction = KMeans_predict)
	CSV.write(pwd()*"\\res_predictions_KMeans.csv", kaggle_KMeans)
end

# ╔═╡ Cell order:
# ╠═bd9df9b0-7c9d-11ed-33ca-f317e29085fa
# ╠═9400c714-064d-450b-bc0e-90b84a597f89
# ╠═cd5f716a-5122-426a-8998-709313a519f4
# ╠═9f8540f4-db37-4d4b-adf5-0262ec58fccd
# ╠═390f32b4-d1bf-4a9b-9209-fc02c6df31ab
# ╠═e8c8d460-4b9d-4a99-9c7a-6712ef4035be
# ╠═551a5788-2553-4b1e-ac65-fe355e9a5b9c
