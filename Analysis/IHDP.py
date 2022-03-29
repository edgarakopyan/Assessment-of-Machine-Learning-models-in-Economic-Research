#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
import ganite as gn
import numpy as np
import econml.dml as dml
import econml.grf as grf
import econml.dr as dr
import pandas as pd
import tqdm
import warnings
from sklearn.preprocessing import PolynomialFeatures
import os
import subprocess
warnings.filterwarnings("ignore")
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

dir = os.getcwd()

df_train = np.load(dir + "/ihdp_npci_1-100.train.npz")
df_test = np.load(dir + "/ihdp_npci_1-100.test.npz")

mu1_test = df_test['mu1']
mu0_test = df_test['mu0']

mu1_train = df_train['mu1']
mu0_train = df_train['mu0']

cate_test = mu1_test - mu0_test
cate_train = mu1_train - mu0_train

ates = np.ones(100) * df_train["ate"]

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# Run Models

# Causal Forest with DML

model = dml.CausalForestDML(n_estimators=2000, max_features=25, random_state=123, discrete_treatment=True,
                            criterion="het", subforest_size=200, max_samples=0.5, cv=3)

pehe_out_cfdml_ihdp = []
missed_out_cfdml_ihdp = []
coverage_out_cfdml_ihdp = []
pehe_in_cfdml_ihdp = []
missed_in_cfdml_ihdp = []
coverage_in_cfdml_ihdp = []
ate_cfdml_ihdp = []

for i in tqdm.tqdm(range(100)):
    # out sample
    X_train, W_train, Y_train= df_train['x'][:,:,i], df_train['t'][:,i], df_train["yf"][:,i]
    model.tune(Y_train, W_train, X = X_train)
    model.fit(Y_train, W_train, X = X_train)
    subtraction = np.subtract(cate_test[:,i], np.reshape(model.effect(X = df_test['x'][:,:,i]), (75,)))
    subtraction = subtraction**2
    pehe_out_cfdml_ihdp.append(np.sqrt(sum(subtraction)/75))
    intervals = model.effect_interval(X = df_test['x'][:,:,i])
    coverage_out_cfdml_ihdp.append(np.mean((intervals[0] < cate_test[:, i]) & (cate_test[:, i] < intervals[1])))
    missed_out_cfdml_ihdp.append(np.mean((intervals[0] < 0) & (0 < intervals[1])))

    # in sample
    subtraction = np.subtract(cate_train[:,i], np.reshape(model.effect(X = X_train), (672,)))
    subtraction = subtraction**2
    pehe_in_cfdml_ihdp.append(np.sqrt(sum(subtraction)/672))
    intervals = model.effect_interval(X = X_train)
    ci_low = np.reshape(intervals[0], (672,))
    ci_high = np.reshape(intervals[1], (672,))
    coverage_in_cfdml_ihdp.append(np.mean((ci_low < cate_train[:, i]) & (cate_train[:, i] < ci_high)))
    missed_in_cfdml_ihdp.append(np.mean((ci_low < 0) & (0 < ci_high)))

    # in sample ATE and its coverage
    ate_cfdml_ihdp.append(model.ate_[0])


# Regular Causal Forest

model = grf.CausalForest(n_estimators=2000, random_state=123,  max_features=25, criterion="het", subforest_size=200,
                         max_samples=0.5)

pehe_out_cf_ihdp = []
missed_out_cf_ihdp = []
coverage_out_cf_ihdp = []
pehe_in_cf_ihdp = []
missed_in_cf_ihdp = []
coverage_in_cf_ihdp = []
ate_cf_ihdp = []

for i in tqdm.tqdm(range(100)):
    X_train, W_train, Y_train= df_train['x'][:,:,i], df_train['t'][:,i], df_train["yf"][:,i]
    model.fit(y = Y_train, T = W_train, X = X_train)
    subtraction = np.subtract(cate_test[:,i], np.reshape(model.predict(X = df_test['x'][:,:,i]), (75,)))
    subtraction = subtraction**2
    pehe_out_cf_ihdp.append(np.sqrt(sum(subtraction)/75))
    intervals = model.predict_interval(X = df_test['x'][:,:,i])
    ci_low = np.reshape(intervals[0], (75,))
    ci_high = np.reshape(intervals[1], (75,))
    coverage_out_cf_ihdp.append(np.mean((ci_low < cate_test[:, i]) & (cate_test[:, i] < ci_high)))
    missed_out_cf_ihdp.append(np.mean((ci_low < 0) & (0 < ci_high)))

    # in sample
    subtraction = np.subtract(cate_train[:,i], np.reshape(model.predict(X = X_train), (672,)))
    subtraction = subtraction**2
    pehe_in_cf_ihdp.append(np.sqrt(sum(subtraction)/672))
    intervals = model.predict_interval(X = X_train)
    ci_low = np.reshape(intervals[0], (672,))
    ci_high = np.reshape(intervals[1], (672,))
    coverage_in_cf_ihdp.append(np.mean((ci_low < cate_train[:, i]) & (cate_train[:, i] < ci_high)))
    missed_in_cf_ihdp.append(np.mean((ci_low < 0) & (0 < ci_high)))

    # in sample ATE
    ate_cf_ihdp.append(np.mean(model.predict(X = X_train)))


# Linear DML

model = dml.LinearDML(linear_first_stages = False, random_state=123, featurizer=PolynomialFeatures(degree = 2,
                                                                                                   include_bias=False))

pehe_out_linear_ihdp = []
missed_out_linear_ihdp = []
coverage_out_linear_ihdp = []
pehe_in_linear_ihdp = []
missed_in_linear_ihdp = []
coverage_in_linear_ihdp = []
ate_linear_ihdp = []

for i in tqdm.tqdm(range(100)):
    # out sample
    X_train, W_train, Y_train= df_train['x'][:,:,i], df_train['t'][:,i], df_train["yf"][:,i]
    model.fit(Y_train, W_train, X = X_train)
    subtraction = np.subtract(cate_test[:,i], np.reshape(model.effect(X = df_test['x'][:,:,i]), (75,)))
    subtraction = subtraction**2
    pehe_out_linear_ihdp.append(np.sqrt(sum(subtraction)/75))
    intervals = model.effect_interval(X = df_test['x'][:,:,i])
    ci_low = np.reshape(intervals[0], (75,))
    ci_high = np.reshape(intervals[1], (75,))
    coverage_out_linear_ihdp.append(np.mean((ci_low < cate_test[:, i]) & (cate_test[:, i] < ci_high)))
    missed_out_linear_ihdp.append(np.mean((ci_low < 0) & (0 < ci_high)))

    # in sample
    subtraction = np.subtract(cate_train[:,i], np.reshape(model.effect(X = X_train), (672,)))
    subtraction = subtraction**2
    pehe_in_linear_ihdp.append(np.sqrt(sum(subtraction)/672))
    intervals = model.effect_interval(X = X_train)
    ci_low = np.reshape(intervals[0], (672,))
    ci_high = np.reshape(intervals[1], (672,))
    coverage_in_linear_ihdp.append(np.mean((ci_low < cate_train[:, i]) & (cate_train[:, i] < ci_high)))
    missed_in_linear_ihdp.append(np.mean((ci_low < 0) & (0 < ci_high)))

    # In sample ATE
    ate_linear_ihdp.append(model.ate(X = X_train))


# Sparse Linear DML

model = dml.SparseLinearDML(linear_first_stages = False, random_state=123,
                            featurizer=PolynomialFeatures(degree = 1, include_bias=False))

pehe_out_sparse_ihdp = []
missed_out_sparse_ihdp = []
coverage_out_sparse_ihdp = []
pehe_in_sparse_ihdp = []
missed_in_sparse_ihdp = []
coverage_in_sparse_ihdp = []
ate_sparse_ihdp = []

for i in tqdm.tqdm(range(100)):
    X_train, W_train, Y_train= df_train['x'][:,:,i], df_train['t'][:,i], df_train["yf"][:,i]
    model.fit(Y_train, W_train, X = X_train)
    subtraction = np.subtract(cate_test[:,i], np.reshape(model.effect(X = df_test['x'][:,:,i]), (75,)))
    subtraction = subtraction**2
    pehe_out_sparse_ihdp.append(np.sqrt(sum(subtraction)/75))
    intervals = model.effect_interval(X = df_test['x'][:,:,i])
    ci_low = np.reshape(intervals[0], (75,))
    ci_high = np.reshape(intervals[1], (75,))
    coverage_out_sparse_ihdp.append(np.mean((ci_low < cate_test[:, i]) & (cate_test[:, i] < ci_high)))
    missed_out_sparse_ihdp.append(np.mean((ci_low < 0) & (0 < ci_high)))

    # in sample
    subtraction = np.subtract(cate_train[:,i], np.reshape(model.effect(X = X_train), (672,)))
    subtraction = subtraction**2
    pehe_in_sparse_ihdp.append(np.sqrt(sum(subtraction)/672))
    intervals = model.effect_interval(X = X_train)
    ci_low = np.reshape(intervals[0], (672,))
    ci_high = np.reshape(intervals[1], (672,))
    coverage_in_sparse_ihdp.append(np.mean((ci_low < cate_train[:, i]) & (cate_train[:, i] < ci_high)))
    missed_in_sparse_ihdp.append(np.mean((ci_low < 0) & (0 < ci_high)))
    # In sample ATE
    ate_sparse_ihdp.append(model.ate(X = X_train))


# Generative Adversarial Network

pehe_in_gan_ihdp = []
pehe_out_gan_ihdp = []
ate_gan_ihdp = []

for i in tqdm.tqdm(range(100)):
    # out sample
    X_train, W_train, Y_train = df_train['x'][:,:,i], df_train['t'][:,i], df_train["yf"][:,i]
    X_test =  df_test['x'][:,:,i]
    model = gn.Ganite(X_train, W_train, Y_train, alpha=2, beta=5, dim_hidden=25, depth=5, minibatch_size=32,
                      num_iterations=1000)
    pred = model(X_test).numpy()
    subtraction = np.subtract(cate_test[:,i], pred)
    subtraction = subtraction**2
    print(np.sqrt(np.mean(subtraction)))
    pehe_out_gan_ihdp.append(np.sqrt(np.mean(subtraction)))

    # in sample
    predictions = model(X_train).numpy()
    subtraction = np.subtract(cate_train[:,i], predictions)
    subtraction = subtraction**2
    pehe_in_gan_ihdp.append(np.sqrt(sum(subtraction)/672))

    # In sample ATE
    ate_gan_ihdp.append(np.mean(predictions))


# DR Forest

model = dr.ForestDRLearner( n_estimators=2000,  random_state=123, subforest_size=200, max_samples=0.5)

pehe_out_drf_ihdp = []
pehe_in_drf_ihdp = []
coverage_out_drf_ihdp = []
missed_out_drf_ihdp = []
coverage_in_drf_ihdp = []
missed_in_drf_ihdp = []
ate_drf_ihdp = []

for i in tqdm.tqdm(range(100)):
    # out sample
    X_train, W_train, Y_train= df_train['x'][:,:,i], df_train['t'][:,i], df_train["yf"][:,i]
    model.fit(Y_train, W_train, X = X_train)
    subtraction = np.subtract(cate_test[:,i], np.reshape(model.effect(X = df_test['x'][:,:,i]), (75,)))
    subtraction = subtraction**2
    pehe_out_drf_ihdp.append(np.sqrt(sum(subtraction)/75))
    intervals = model.effect_interval(X = df_test['x'][:,:,i])
    ci_low = np.reshape(intervals[0], (75,))
    ci_high = np.reshape(intervals[1], (75,))
    coverage_out_drf_ihdp.append(np.mean((ci_low < cate_test[:, i]) & (cate_test[:, i] < ci_high)))
    missed_out_drf_ihdp.append(np.mean((ci_low < 0) & (0 < ci_high)))

    # in sample
    subtraction = np.subtract(cate_train[:,i], np.reshape(model.effect(X = X_train), (672,)))
    subtraction = subtraction**2
    pehe_in_drf_ihdp.append(np.sqrt(sum(subtraction)/672))
    intervals = model.effect_interval(X = X_train)
    ci_low = np.reshape(intervals[0], (672,))
    ci_high = np.reshape(intervals[1], (672,))
    coverage_in_drf_ihdp.append(np.mean((ci_low < cate_train[:, i]) & (cate_train[:, i] < ci_high)))
    missed_in_drf_ihdp.append(np.mean((ci_low < 0) & (0 < ci_high)))

    # In sample ATE
    ate_drf_ihdp.append(model.ate(X = X_train))


# DR Linear


pehe_out_drlin_ihdp = []
pehe_in_drlin_ihdp = []
coverage_out_drlin_ihdp = []
missed_out_drlin_ihdp = []
coverage_in_drlin_ihdp = []
missed_in_drlin_ihdp = []
ate_drlin_ihdp = []

model = dr.LinearDRLearner( random_state=123, featurizer=PolynomialFeatures(degree = 2, include_bias=False), cv=5)


for i in tqdm.tqdm(range(100)):
    X_train, W_train, Y_train= df_train['x'][:,:,i], df_train['t'][:,i], df_train["yf"][:,i]
    model.fit(Y_train, W_train, X = X_train)
    subtraction = np.subtract(cate_test[:,i], np.reshape(model.effect(X = df_test['x'][:,:,i]), (75,)))
    subtraction = subtraction**2
    pehe_out_drlin_ihdp.append(np.sqrt(sum(subtraction)/75))
    intervals = model.effect_interval(X = df_test['x'][:,:,i])
    ci_low = np.reshape(intervals[0], (75,))
    ci_high = np.reshape(intervals[1], (75,))
    coverage_out_drlin_ihdp.append(np.mean((ci_low < cate_test[:, i]) & (cate_test[:, i] < ci_high)))
    missed_out_drlin_ihdp.append(np.mean((ci_low < 0) & (0 < ci_high)))

    # in sample
    subtraction = np.subtract(cate_train[:,i], np.reshape(model.effect(X = X_train), (672,)))
    subtraction = subtraction**2
    pehe_in_drlin_ihdp.append(np.sqrt(sum(subtraction)/672))
    intervals = model.effect_interval(X = X_train)
    ci_low = np.reshape(intervals[0], (672,))
    ci_high = np.reshape(intervals[1], (672,))
    coverage_in_drlin_ihdp.append(np.mean((ci_low < cate_train[:, i]) & (cate_train[:, i] < ci_high)))
    missed_in_drlin_ihdp.append(np.mean((ci_low < 0) & (0 < ci_high)))
    # In sample ATE
    ate_drlin_ihdp.append(model.ate(X = X_train))



# DR Sparse Linear
pehe_out_drsparse_ihdp = []
pehe_in_drsparse_ihdp = []
coverage_out_drsparse_ihdp = []
missed_out_drsparse_ihdp = []
coverage_in_drsparse_ihdp = []
missed_in_drsparse_ihdp = []
ate_drsparse_ihdp = []


for i in tqdm.tqdm(range(100)):
    X_train, W_train, Y_train= df_train['x'][:,:,i], df_train['t'][:,i], df_train["yf"][:,i]
    model.fit(Y_train, W_train, X = X_train)
    subtraction = np.subtract(cate_test[:,i], np.reshape(model.effect(X = df_test['x'][:,:,i]), (75,)))
    subtraction = subtraction**2
    pehe_out_drsparse_ihdp.append(np.sqrt(sum(subtraction)/75))
    intervals = model.effect_interval(X = df_test['x'][:,:,i])
    ci_low = np.reshape(intervals[0], (75,))
    ci_high = np.reshape(intervals[1], (75,))
    coverage_out_drsparse_ihdp.append(np.mean((ci_low < cate_test[:, i]) & (cate_test[:, i] < ci_high)))
    missed_out_drsparse_ihdp.append(np.mean((ci_low < 0) & (0 < ci_high)))

    # in sample
    subtraction = np.subtract(cate_train[:,i], np.reshape(model.effect(X = X_train), (672,)))
    subtraction = subtraction**2
    pehe_in_drsparse_ihdp.append(np.sqrt(sum(subtraction)/672))
    intervals = model.effect_interval(X = X_train)
    ci_low = np.reshape(intervals[0], (672,))
    ci_high = np.reshape(intervals[1], (672,))
    coverage_in_drsparse_ihdp.append(np.mean((ci_low < cate_train[:, i]) & (cate_train[:, i] < ci_high)))
    missed_in_drsparse_ihdp.append(np.mean((ci_low < 0) & (0 < ci_high)))
    # In sample ATE
    ate_drsparse_ihdp.append(model.ate(X = X_train))


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# Finally, save the resulting dataframe

final_data_drf = pd.DataFrame(
    np.array([pehe_out_drf_ihdp, pehe_in_drf_ihdp, coverage_out_drf_ihdp,
              missed_out_drf_ihdp, coverage_in_drf_ihdp, missed_in_drf_ihdp,
              ate_drf_ihdp]).T,
    columns =["pehe_out_drf_ihdp", "pehe_in_drf_ihdp", "coverage_out_drf_ihdp",
              "missed_out_drf_ihdp", "coverage_in_drf_ihdp", "missed_in_drf_ihdp",
              "ate_drf_ihdp" ]
)

final_data_drlin = pd.DataFrame(
    np.array([pehe_out_drlin_ihdp, pehe_in_drlin_ihdp, coverage_out_drlin_ihdp,
              missed_out_drlin_ihdp, coverage_in_drlin_ihdp, missed_in_drlin_ihdp,
              ate_drlin_ihdp]).T,
    columns =["pehe_out_drlin_ihdp", "pehe_in_drlin_ihdp", "coverage_out_drlin_ihdp",
              "missed_out_drlin_ihdp", "coverage_in_drlin_ihdp", "missed_in_drlin_ihdp",
              "ate_drlin_ihdp" ]
)

final_data_drsparse = pd.DataFrame(
    np.array([pehe_out_drsparse_ihdp, pehe_in_drsparse_ihdp, coverage_out_drsparse_ihdp,
              missed_out_drsparse_ihdp, coverage_in_drsparse_ihdp, missed_in_drsparse_ihdp,
              ate_drsparse_ihdp]).T,
    columns =["pehe_out_drsparse_ihdp", "pehe_in_drsparse_ihdp", "coverage_out_drsparse_ihdp",
              "missed_out_drsparse_ihdp", "coverage_in_drsparse_ihdp", "missed_in_drsparse_ihdp",
              "ate_drsparse_ihdp" ]
)

final_data_ganite = pd.DataFrame(
    np.array([pehe_in_gan_ihdp, pehe_out_gan_ihdp, ate_gan_ihdp ]).T,
    columns =["pehe_in_gan_ihdp", "pehe_out_gan_ihdp", "ate_gan_ihdp" ]
)

final_data_sparse = pd.DataFrame(
    np.array([pehe_out_sparse_ihdp, pehe_in_sparse_ihdp, coverage_out_sparse_ihdp, missed_out_sparse_ihdp,
              coverage_in_sparse_ihdp, missed_in_sparse_ihdp, ate_sparse_ihdp]).T,
    columns =["pehe_out_sparse_ihdp", "pehe_in_sparse_ihdp", "coverage_out_sparse_ihdp", "missed_out_sparse_ihdp",
              "coverage_in_sparse_ihdp", "missed_in_sparse_ihdp", "ate_sparse_ihdp" ]
)

final_data_linear = pd.DataFrame(
    np.array([pehe_out_linear_ihdp, pehe_in_linear_ihdp, coverage_out_linear_ihdp, missed_out_linear_ihdp,
              coverage_in_linear_ihdp, missed_in_linear_ihdp, ate_linear_ihdp]).T,
    columns =["pehe_out_linear_ihdp", "pehe_in_linear_ihdp", "coverage_out_linear_ihdp", "missed_out_linear_ihdp",
              "coverage_in_linear_ihdp", "missed_in_linear_ihdp", "ate_linear_ihdp" ]
)

final_data_cf = pd.DataFrame(
    np.array([pehe_out_cf_ihdp, pehe_in_cf_ihdp, coverage_out_cf_ihdp, missed_out_cf_ihdp, ate_cf_ihdp,
              coverage_in_cf_ihdp, missed_in_cf_ihdp ]).T,
    columns =["pehe_out_cf_ihdp", "pehe_in_cf_ihdp", "coverage_out_cf_ihdp", "missed_out_cf_ihdp", "ate_cf_ihdp",
              "coverage_in_cf_ihdp", "missed_in_cf_ihdp" ]
)

final_data_cfdml = pd.DataFrame(
    np.array([pehe_out_cfdml_ihdp, pehe_in_cfdml_ihdp, coverage_out_cfdml_ihdp, missed_out_cfdml_ihdp,
              coverage_in_cfdml_ihdp, missed_in_cfdml_ihdp, ate_cfdml_ihdp ]).T,
    columns =["pehe_out_cfdml_ihdp", "pehe_in_cfdml_ihdp", "coverage_out_cfdml_ihdp", "missed_out_cfdml_ihdp",
              "coverage_in_cfdml_ihdp", "missed_in_cfdml_ihdp", "ate_cfdml_ihdp" ]
)

final = pd.concat(
    [final_data_drf,final_data_drlin, final_data_drsparse, final_data_sparse, final_data_linear, final_data_cf,
     final_data_cfdml, final_data_ganite, pd.DataFrame(ates, columns = ["ates"]) ],
    axis = 1
)

final.to_csv(dir + "/Results/ihdp_acic_models.csv", index = False)

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# We also run BART model but save it separately

path2script = dir + "/R files/IHDP_BART.R"
try:
    subprocess.call(['Rscript', path2script, dir])
except subprocess.CalledProcessError as e:
    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
