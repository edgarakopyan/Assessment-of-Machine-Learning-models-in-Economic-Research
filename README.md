# Assessment-of-Machine-Learning-models-in-Economic-Research

Code for "Assessment of Machine Learning models in Economic Research" Master Thesis. 

## Models assessed: 

1. Bayesian Additive Regression Trees from bartCause package (R) 
URL: https://cran.r-project.org/web/packages/bartCause/index.html
2. Double Machine Learning, Doubly Robust Learning, Causal Forest from EconML package (Python)
URL: https://econml.azurewebsites.net
3. Generative Adversarial Nets for inference of Individualized Treat- ment Effects (GANITE) from GANITE package (Python)
URL: https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/ganite

## How to Run 

Code is separated into different folders. The code that needs to be run is Python code, R scripts are used within Python files. As a prelimiary run Install_Required_Packages.R file in Rstudio to make sure that all R packages are installed and install packages from requirements.txt file. 

To recreate main results from the paper proceed with following steps:

1. Begin with Preparations folder: run Folder_Setup.py, Data_Setup_IHDP.py and Data_Setup_ACIC.py in this sequence. 
2. Continue with Analysis folder: run ACIC.py and then IHDP.py.
3. Finally, to create graphs and tables: run Results_prep.py.

All model results will be stored in Results folder. Tables and Graphs will be in Graphs_and_Tables folder. 

GitHub repo: https://github.com/edgarakopyan/Assessment-of-Machine-Learning-models-in-Economic-Research
