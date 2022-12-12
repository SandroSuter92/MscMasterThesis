# MscMasterThesis

This repository contains the data, the codes and the output files with the required information for my Master Thesis 
"The Prediction Accuracy of Combined Tree-Boosting and Gaussian Process Models".

In the folder 'Code' there is a ".py-file" for each data set, which models the corresponding data set with the methods 
"Gaussian Process", "Gaussian Process Boosting", "Boosting" and "Random Forest" in a 4-fold cross-validation. The required 
outputs are automatically stored by the code in the models subfolder as ".csv-file". In addition to the required KPIs, this 
also contains the respective tuning parameters and the "random states" belonging to the model. If the results are to be 
reproduced exactly, the corresponding "random states" must be entered in the respective lines of code. In addition, the 
path to the data must be adjusted for each code file after importing the required libraries.

The data files used are stored in the 'Data' folder.
