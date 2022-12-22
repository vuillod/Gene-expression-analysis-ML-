# ML-project

Firstly, make sur that all the documents are in the same folder (train.csv, test.csv and the 6 Pluto notebook). 
You can download the data with the command :
download("https://lcnwww.epfl.ch/bio322/project2022/train.csv.gz",
         joinpath(YOUR_FOLDER, "train.csv.gz"))

We will upload the cleaned data in two .dat files in this folder and the prevision in a 
.csv file in this folder too.


**Projet_clean_data.jl :**

We clean the data in this notebook, we remove constant and correlate predictors and we make a PCA with 97,5% cumulative proportion of variance preserved. 
Just let the notebook run and both test data and train data will be cleaned and save in your folder, as clean_data_test.dat and clean_data_train.dat respectively. 


**Projet_Lineaire.jl :** 

We perform the linear classification in this notebook, based on our cleaned data. 
We only let the Lasso run, in order to save time, because it is the methods that works the best. (Ridge classification is present but does not run). 
Let the notebook run and the prediciton will be saved in a file named res_predictions_lasso.csv.


**Projet_Forrest.jl :**

The notebook will run and a file named res_predictions_forrest.csv is saved in the folder. 
But it is not necessary to run it, because the predicitons are not relevant, we just let it here to show our work. 


**Projet_Neurone.jl :**

We perform to kind of Neuronal Network classification in this note book : one with a single hidden layer and one with two hidden layers. 
We obtain the best result with ...... that is why we let run this methods, but the other one is still present in disable cells.
Let the notebook run and the prediciton will be saved in a file named res_predictions_neurone.csv

**Projet_XGBoost.jl :**

We perform a Gradient Boost methods in this notebook.
Let the notebook run and the prediciton will be saved in a file named res_predictions_xgboost.csv.

**Projet_Clustering.jl :**

This notebook is more to have a better comprehension of the data, because we perform some unsupervised methods on the train data in order to visualize easily the data. 
The notebook is subdivised in three part : TSne visualization, KMeans prediction and DBSCAN prediction. 
The TSne cell run and plot the data in a 2D graph, by trying to rearrange the data such that all point clouds are still distinguishable in the lower-dimensional space 
The KMeans prediction is set with k=3 to distribute the train data in 3 clusters. Then we show the confusion matrix of the clustering. 
The DBSCAN prediction is set with a minimum cluster size = 50 and radius = 0.5 and then apply to the train data. Then we show the confusion matrix of the clustering.


**Packages :**

.....