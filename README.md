# ML-project

**First manipulation to set up the environnement**

You can download the test data and the train data with these two links  :

https://lcnwww.epfl.ch/bio322/project2022/train.csv.gz
https://lcnwww.epfl.ch/bio322/project2022/test.csv.gz

Firstly, make sur that all the documents are in the same folder (train.csv.gz, test.csv.gz and the 6 Pluto notebook). 

Note :  - we will upload the cleaned data (test and train) in two .dat files in this folder after    
          running the correspondant notebook.
        - the prevision will be upload in a .csv file in this folder and the end of each notebook.


**Projet_clean_data.jl :**

We clean the data in this notebook, we remove constant and correlate predictors and we make a PCA with 95% cumulative proportion of variance preserved. 
Just let the notebook run and both test data and train data will be cleaned and save in your folder, as clean_data_test.dat and clean_data_train.dat respectively. 

**ALL THE NOTEBOOKS USE THE CLEANED VALUES (.dat) TO MAKE THE PREDICTION**

**Projet_Lineaire.jl :** 

We perform the linear classification in this notebook, based on our cleaned data. 
We only let the Lasso run (best linear model), in order to save time. 
(Ridge classification is present in the code, but does not run). 
We already create a machine with the best lambda, but you can see the steps we perform in comment cells above. If you want to run the tuning function, you can uncomment it, but it can take a moment to tuned the model.
Let the notebook run and the predicitons will be saved in a file named res_predictions_lasso.csv.


**Projet_arbres.jl :**

The notebook will run and a file named res_predictions_forest.csv is saved in the folder. The notebook will run a big tuning model (goal = 147), but it is not necessary to run it, because the predicitons are not relevant (best result = 60%), we just let it here to show our work. 


**Projet_Neurone.jl :**

We perform to kind of Neuronal Network classification in this note book : one with a single hidden layer and one with two hidden layers. 
We obtain the best result with the two hidden layers, that is why we let run only this one, but the other one is still present in disable cells.
Let the notebook run and the predicitons will be saved in a file named res_predictions_neurone.csv

**Projet_XGBoost.jl :**

We perform a Gradient Boost methods in this notebook. In a comment cell, you can see the different values we used to tune our model.
Let the notebook run and the predicitons will be saved in a file named res_predictions_xgboost.csv.

**Projet_Clustering.jl :**

This notebook is more to have a better comprehension of the data, because we perform some unsupervised methods on the train data in order to visualize easily the data. 
The notebook is subdivised in three part : TSne visualization, KMeans prediction and DBSCAN prediction. 
The TSne cell run and plot the data in a 2D graph, by trying to rearrange the data such that all point clouds are still distinguishable in the lower-dimensional space 
The KMeans prediction is set with k=3 to distribute the train data in 3 clusters. Then we show the confusion matrix of the clustering. 
The DBSCAN prediction is set with a minimum cluster size = 50 and radius = 0.5 and then apply to the train data. Then we show the confusion matrix of the clustering (could take a moment to run because of the radius' size is small).


**Packages :**
You will need to install those packages to run our code :

DataFrames 
CSV
MLJ
MLJLinearModels
Optim
MLCourse
Distributions
Plots
Random
OpenML
Statistics
Serialization
MLJFlux
Flux
MLJMultivariateStatsInterface
MLJDecisionTreeInterface
MLJXGBoostInterface
Distances
LinearAlgebra
CategoricalArrays
MLJClusteringInterface
TSne
StatsPlots