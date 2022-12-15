# The Prediction Accuracy of Combined Tree-Boosting and Gaussian Process Models 

This repository contains the underlying data for the master thesis of Sandro Suter.

It contains the

* [Codes](/Code/)
* [Underlying data](/data/)
* [Calculated Outputs](/Code/models/)
* [Calculation of the Friedman statistics](/Calculation_Friedman_Statistic.ipynb)

Since the goal of the thesis is the modeling of different data sets in the area of regressions and binary classifications, below is an overview of which data was used for the regressions and which for the classifications:

**Regressions**

* Student Marks
* Possum
* Car Price
* Housing
* Energy
* Insurance
* Abalone
* Wine White
* diamonds
* KCHouseData

**Classifications**

* Sonar
* Cancer
* Titanic
* Uefa Champions Leage
* Diabetes
* Banknote authentication
* Pumpkin Seeds
* Water Quality
* Churn
* Smoking

For the codes, the .py files were saved with the prefix 'Cla_' or 'Reg_'. If the codes are to be run by themselves, the path must be adjusted accordingly at the beginning of each file. Furthermore, there is a Requirements.txt file in the folder, which lists the used packages. This must be installed in the respective
programming environment.
