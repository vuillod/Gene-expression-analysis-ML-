### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ bd9df9b0-7c9d-11ed-33ca-f317e29085fa
using Pkg;Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

# ╔═╡ 9400c714-064d-450b-bc0e-90b84a597f89
using DataFrames, CSV, MLJ, MLJLinearModels, Optim, MLCourse, Distributions, Plots, Random, OpenML, Statistics, Serialization, MLJClusteringInterface, StatsPlots, Distributions, Distances, LinearAlgebra, TSne, MLJMultivariateStatsInterface


# ╔═╡ cfebb63f-5f38-4de7-8c91-7a6579a6b723
begin
	clean_data = deserialize("clean_data_train.dat")
	coerce!(clean_data, :labels => Multiclass)
end

# ╔═╡ 9f8540f4-db37-4d4b-adf5-0262ec58fccd
test_data = deserialize("clean_data_test.dat")

# ╔═╡ a851159d-b4a1-492c-a229-9c5cc29a8cb1
#TSne visualisation :
begin
	tsne_proj = tsne(Array(select(clean_data, Not(:labels))), 2, 0, 2000, 50.0, progress = false)
	scatter(tsne_proj[:, 1], tsne_proj[:, 2],
        c = Int.(int(clean_data.labels)),
        xlabel = "TSne 1", ylabel = "TSne 2",
        legend = false)

end

# ╔═╡ bc7c438c-1af4-4377-b810-88f962a5d573
#KMeans :
begin
	KMeans_mach = machine(KMeans(k = 3), select(clean_data, Not(:labels)))
    fit!(KMeans_mach, verbosity = 0)
end

# ╔═╡ e8c8d460-4b9d-4a99-9c7a-6712ef4035be
KMeans_predict = MLJ.predict(KMeans_mach, select(clean_data, Not(:labels)))

# ╔═╡ 82f1d02f-86eb-4212-9dc4-e1aaedcc2c9c
confusion_matrix(KMeans_predict, clean_data.labels)

# ╔═╡ 76764df6-e5ef-4864-8c25-4d29401ecd3e
#DBSCAN
begin 
	dbscan_mach = machine(DBSCAN(min_cluster_size = 50, radius = 0.5))
	DBSCAN_pred = predict(dbscan_mach, select(clean_data, Not(:labels)))
end

# ╔═╡ 166d703b-17c0-41b0-814b-5eaa2161608b
confusion_matrix(DBSCAN_pred, clean_data.labels)

# ╔═╡ Cell order:
# ╠═bd9df9b0-7c9d-11ed-33ca-f317e29085fa
# ╠═9400c714-064d-450b-bc0e-90b84a597f89
# ╠═cfebb63f-5f38-4de7-8c91-7a6579a6b723
# ╠═9f8540f4-db37-4d4b-adf5-0262ec58fccd
# ╠═a851159d-b4a1-492c-a229-9c5cc29a8cb1
# ╠═bc7c438c-1af4-4377-b810-88f962a5d573
# ╠═e8c8d460-4b9d-4a99-9c7a-6712ef4035be
# ╠═82f1d02f-86eb-4212-9dc4-e1aaedcc2c9c
# ╠═76764df6-e5ef-4864-8c25-4d29401ecd3e
# ╠═166d703b-17c0-41b0-814b-5eaa2161608b
