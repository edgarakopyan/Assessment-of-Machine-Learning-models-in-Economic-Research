import gdown
import tarfile
import subprocess
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import shutil

dir = os.getcwd()

# First download ACIC data
file_id = '0B7pG5PPgj6A3N09ibmFwNWE1djA'
destination = dir + "/data_cf_all.tar.gz"
gdown.download(id=file_id, output=destination, quiet=False)

tar = tarfile.open(destination, "r:gz")
tar.extractall(dir)
tar.close()

# Prepare ACIC data with the correct outcome
command = "Rscript"
path2script = dir + "/R files/ACIC_data_add_outcome.R"
try:
    subprocess.call([command, path2script, dir])
except subprocess.CalledProcessError as e:
    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))


# Prepare Covariates
X = pd.read_csv(dir + "/data_cf_all/x.csv")

feature_list = []
for cols_ in X.columns:
    if type(X.loc[X.index[0], cols_]) not in [np.int64, np.float64]:

        enc = OneHotEncoder(drop="first")

        enc.fit(np.array(X[[cols_]]).reshape((-1, 1)))

        for k in range(len(list(enc.get_feature_names_out()))):
            X[cols_ + list(enc.get_feature_names_out())[k]] = enc.transform(
                np.array(X[[cols_]]).reshape((-1, 1))
            ).toarray()[:, k]

        feature_list.append(cols_)

X.drop(feature_list, axis=1, inplace=True)

X.to_csv(dir + "/ACIC/x.csv", index = False)

shutil.rmtree(dir + "/data_cf_all", ignore_errors = True)


