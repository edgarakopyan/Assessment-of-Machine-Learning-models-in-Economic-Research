#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
import os
import pandas as pd
import ganite as gn
import numpy as np
import econml.dml as dml
import econml.grf as grf
from tqdm import tqdm
import econml.dr as dr
from sklearn.preprocessing import PolynomialFeatures
import warnings
import subprocess
warnings.filterwarnings("ignore")
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

Knobs = list([2, 3, 4, 7, 8, 11, 12, 16, 25, 26, 32, 56])

dir = os.getcwd()

# Load Covariates
X = pd.read_csv(dir + "/ACIC/x.csv")

for k in Knobs:
    # Set the directory to the ACIC folder
    directory_in_str = dir + "/ACIC/" + str(k)
    directory = os.fsencode(directory_in_str)

    # Load all datas into one list to avoid loading later
    datas = []
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        df = pd.read_csv(directory_in_str + "/" + filename)
        datas.append(df)


    # calculate ATEs
    ates = []
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        df = pd.read_csv(directory_in_str + "/" + filename)
        df_train = df[0:4000]
        cate = df_train['mu1'] - df_train['mu0']
        ates.append(np.mean(cate))

    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
    # Run Models

    # Causal Forest DML

    model = dml.CausalForestDML(n_estimators=2000, max_features=int(np.sqrt(X.shape[1]) + 20), random_state=123,
                                discrete_treatment=True, criterion="het", subforest_size=200, max_samples=0.5, cv=3)

    pehe_out_cfdml_acic = []
    pehe_in_cfdml_acic = []
    coverage_out_cfdml_acic = []
    missed_out_cfdml_acic = []
    coverage_in_cfdml_acic = []
    missed_in_cfdml_acic = []
    ate_cfdml_acic = []

    for i in tqdm(datas):
        # Out Sample PEHE
        df = i
        df_train = df[0:4000]
        df_test = df[4000:]
        cate = df_test['mu1'] - df_test['mu0']
        y = df_train['y']
        x = X[0:4000]
        model.fit(y, df_train[['z']], X=x)
        subtraction = np.subtract(cate, np.reshape(model.effect(X = X[4000:]), (802,)))
        subtraction = subtraction**2
        pehe_out_cfdml_acic.append(np.sqrt(sum(subtraction)/802))

        # Coverage and Missed
        intervals = model.effect_interval(X = X[4000:])
        ci_low = np.reshape(intervals[0], (802,))
        ci_high = np.reshape(intervals[1], (802,))
        coverage_out_cfdml_acic.append(np.mean((intervals[0] < cate) & (cate < intervals[1])))
        missed_out_cfdml_acic.append(np.mean((intervals[0] < 0) & (0 < intervals[1])))

        # In Sample PEHE
        cate = df_train['mu1'] - df_train['mu0']

        # in sample ATE and its coverage
        ate_cfdml_acic.extend(model.ate_)

        subtraction = np.subtract(cate, np.reshape(model.effect(X = x), (4000,)))
        subtraction = subtraction**2
        pehe_in_cfdml_acic.append(np.sqrt(sum(subtraction)/4000))

        # Coverage and Missed
        intervals = model.effect_interval(X = x)
        ci_low = np.reshape(intervals[0], (4000,))
        ci_high = np.reshape(intervals[1], (4000,))
        coverage_in_cfdml_acic.append(np.mean((intervals[0] < cate) & (cate < intervals[1])))
        missed_in_cfdml_acic.append(np.mean((intervals[0] < 0) & (0 < intervals[1])))


    # Regular Causal Forest

    model = grf.CausalForest(n_estimators=2000, random_state=123,  max_features=int(np.sqrt(X.shape[1]) + 20),
                             criterion="het", subforest_size=200, max_samples=0.5)

    pehe_out_cf_acic = []
    pehe_in_cf_acic = []
    coverage_out_cf_acic = []
    missed_out_cf_acic = []
    ate_cf_acic = []
    coverage_in_cf_acic = []
    missed_in_cf_acic = []

    for i in tqdm(datas):
        df = i
        df_train = df[0:4000]
        df_test = df[4000:]
        cate = df_test['mu1'] - df_test['mu0']
        y = df_train['y']
        x = X[0:4000]
        model.fit(y = y, T = df_train[['z']], X = x )
        subtraction = np.subtract(cate, np.reshape(model.predict(X = X[4000:]), (802,)))
        subtraction = subtraction**2
        pehe_out_cf_acic.append(np.sqrt(sum(subtraction)/802))

        intervals = model.predict_interval(X = X[4000:])
        ci_low = np.reshape(intervals[0], (802,))
        ci_high = np.reshape(intervals[1], (802,))
        coverage_out_cf_acic.append(np.mean((ci_low < cate) & (cate < ci_high)))
        missed_out_cf_acic.append(np.mean((ci_low < 0) & (0 < ci_high)))

        cate = df_train['mu1'] - df_train['mu0']

        # in sample ATE
        ate_cf_acic.append(np.mean(model.predict(X = x)))


        subtraction = np.subtract(cate, np.reshape(model.predict(X = x), (4000,)))
        subtraction = subtraction**2
        pehe_in_cf_acic.append(np.sqrt(sum(subtraction)/4000))

        intervals = model.predict_interval(X = x)
        ci_low = np.reshape(intervals[0], (4000,))
        ci_high = np.reshape(intervals[1], (4000,))
        coverage_in_cf_acic.append(np.mean((ci_low < cate) & (cate < ci_high)))
        missed_in_cf_acic.append(np.mean((ci_low < 0) & (0 < ci_high)))


    # Linear DML

    model = dml.LinearDML(linear_first_stages = False, random_state=123, featurizer=PolynomialFeatures(degree = 2,
                                                                                                       include_bias=False))

    pehe_out_linear_acic = []
    pehe_in_linear_acic = []
    coverage_out_linear_acic = []
    missed_out_linear_acic = []
    coverage_in_linear_acic = []
    missed_in_linear_acic = []
    ate_linear_acic = []
    ate_coverage_linear_acic = []

    for i in tqdm(datas):
        df = i
        df_train = df[0:4000]
        df_test = df[4000:]
        cate = df_test['mu1'] - df_test['mu0']
        y = df_train['y']
        x = X[0:4000]
        model.fit(Y = y, T = df_train[['z']], X = x)
        subtraction = np.subtract(cate, np.reshape(model.effect(X = X[4000:]), (802,)))
        subtraction = subtraction**2
        pehe_out_linear_acic.append(np.sqrt(sum(subtraction)/802))

        intervals = model.effect_interval(X = X[4000:])
        ci_low = np.reshape(intervals[0], (802,))
        ci_high = np.reshape(intervals[1], (802,))
        coverage_out_linear_acic.append(np.mean((ci_low < cate) & (cate < ci_high)))
        missed_out_linear_acic.append(np.mean((ci_low < 0) & (0 < ci_high)))

        cate = df_train['mu1'] - df_train['mu0']

        # In sample ATE
        ate_linear_acic.append(model.ate(X = x))

        subtraction = np.subtract(cate, np.reshape(model.effect(X = x), (4000,)))
        subtraction = subtraction**2
        pehe_in_linear_acic.append(np.sqrt(sum(subtraction)/4000))

        intervals = model.effect_interval(X = x)
        ci_low = np.reshape(intervals[0], (4000,))
        ci_high = np.reshape(intervals[1], (4000,))
        coverage_in_linear_acic.append(np.mean((ci_low < cate) & (cate < ci_high)))
        missed_in_linear_acic.append(np.mean((ci_low < 0) & (0 < ci_high)))

    # Sparse Linear DML

    model = dml.SparseLinearDML(linear_first_stages = False, random_state=123,
                                featurizer=PolynomialFeatures(degree = 1, include_bias=False), cv=5)

    pehe_out_sparse_acic = []
    pehe_in_sparse_acic = []
    coverage_out_sparse_acic = []
    missed_out_sparse_acic = []
    coverage_in_sparse_acic = []
    missed_in_sparse_acic = []
    ate_sparse_acic = []

    for i in tqdm(datas):
        df = i
        df_train = df[0:4000]
        df_test = df[4000:]
        cate = df_test['mu1'] - df_test['mu0']
        y = df_train['y']
        x = X[0:4000]
        model.fit(Y = y.to_numpy(), T = df_train['z'].to_numpy(), X = x)
        subtraction = np.subtract(cate, np.reshape(model.effect(X = X[4000:]), (802,)))
        subtraction = subtraction**2
        pehe_out_sparse_acic.append(np.sqrt(sum(subtraction)/802))

        intervals = model.effect_interval(X = X[4000:].to_numpy())
        ci_low = np.reshape(intervals[0], (802,))
        ci_high = np.reshape(intervals[1], (802,))
        coverage_out_sparse_acic.append(np.mean((ci_low < cate) & (cate < ci_high)))
        missed_out_sparse_acic.append(np.mean((ci_low < 0) & (0 < ci_high)))

        cate = df_train['mu1'] - df_train['mu0']

        # In sample ATE
        ate_sparse_acic.append(model.ate(X = x))


        subtraction = np.subtract(cate, np.reshape(model.effect(X = x), (4000,)))
        subtraction = subtraction**2
        pehe_in_sparse_acic.append(np.sqrt(sum(subtraction)/4000))

        intervals = model.effect_interval(X = x)
        ci_low = np.reshape(intervals[0], (4000,))
        ci_high = np.reshape(intervals[1], (4000,))
        coverage_in_sparse_acic.append(np.mean((ci_low < cate) & (cate < ci_high)))
        missed_in_sparse_acic.append(np.mean((ci_low < 0) & (0 < ci_high)))


    #  Generative Adversarial Network

    pehe_in_gan_acic = []
    pehe_out_gan_acic = []
    ate_gan_acic = []

    for i in tqdm(datas):
        df =i
        df_train = df[0:4000]
        df_test = df[4000:]
        cate = df_test['mu1'] - df_test['mu0']
        y = df_train['y']
        x = X[0:4000]
        model = gn.Ganite( X= x, Treatments=df_train['z'], Y=y, alpha=2, beta=5, dim_hidden=79,
                           depth=5, minibatch_size=16, num_iterations=1000)
        subtraction = np.subtract(cate, model(X[4000:]).numpy())
        subtraction = subtraction**2
        pehe_out_gan_acic.append(np.sqrt(sum(subtraction)/802))

        # In Sample PEHE
        cate = df_train['mu1'] - df_train['mu0']
        y = df_train['y']
        predictions = model(x).numpy()
        subtraction = np.subtract(cate, predictions)
        subtraction = subtraction**2
        pehe_in_gan_acic.append(np.sqrt(sum(subtraction)/4000))

        # In sample ATE
        ate_gan_acic.append(np.mean(predictions))



    # DR Forest
    model = dr.ForestDRLearner( n_estimators=2000,  random_state=123, subforest_size=200,
                                              max_samples=0.5, max_features=int(np.sqrt(X.shape[1]) + 20))

    pehe_out_drf_acic = []
    pehe_in_drf_acic = []
    coverage_out_drf_acic = []
    missed_out_drf_acic = []
    coverage_in_drf_acic = []
    missed_in_drf_acic = []
    ate_drf_acic = []

    for i in tqdm(datas):
        df = i
        df_train = df[0:4000]
        df_test = df[4000:]
        cate = df_test['mu1'] - df_test['mu0']
        y = df_train['y']
        x = X[0:4000]
        model.fit(Y = y, T = df_train['z'], X = x)
        subtraction = np.subtract(cate, np.reshape(model.effect(X = X[4000:]), (802,)))
        subtraction = subtraction**2
        pehe_out_drf_acic.append(np.sqrt(sum(subtraction)/802))

        intervals = model.effect_interval(X = X[4000:])
        ci_low = np.reshape(intervals[0], (802,))
        ci_high = np.reshape(intervals[1], (802,))
        coverage_out_drf_acic.append(np.mean((ci_low < cate) & (cate < ci_high)))
        missed_out_drf_acic.append(np.mean((ci_low < 0) & (0 < ci_high)))

        cate = df_train['mu1'] - df_train['mu0']

        # In sample ATE
        ate_drf_acic.append(model.ate(X = x))

        subtraction = np.subtract(cate, np.reshape(model.effect(X = x), (4000,)))
        subtraction = subtraction**2
        pehe_in_drf_acic.append(np.sqrt(sum(subtraction)/4000))

        intervals = model.effect_interval(X = x)
        ci_low = np.reshape(intervals[0], (4000,))
        ci_high = np.reshape(intervals[1], (4000,))
        coverage_in_drf_acic.append(np.mean((ci_low < cate) & (cate < ci_high)))
        missed_in_drf_acic.append(np.mean((ci_low < 0) & (0 < ci_high)))


    # DR Sparse
    model = dr.SparseLinearDRLearner(  random_state=123,
                                       featurizer=PolynomialFeatures(degree = 1, include_bias=False))

    pehe_out_drsparse_acic = []
    pehe_in_drsparse_acic = []
    coverage_out_drsparse_acic = []
    missed_out_drsparse_acic = []
    coverage_in_drsparse_acic = []
    missed_in_drsparse_acic = []
    ate_drsparse_acic = []

    for i in tqdm(datas):
        df = i
        df_train = df[0:4000]
        df_test = df[4000:]
        cate = df_test['mu1'] - df_test['mu0']
        y = df_train['y']
        x = X[0:4000]
        model.fit(Y = y, T = df_train['z'], X = x)
        subtraction = np.subtract(cate, np.reshape(model.effect(X = X[4000:]), (802,)))
        subtraction = subtraction**2
        pehe_out_drsparse_acic.append(np.sqrt(sum(subtraction)/802))

        intervals = model.effect_interval(X = X[4000:])
        ci_low = np.reshape(intervals[0], (802,))
        ci_high = np.reshape(intervals[1], (802,))
        coverage_out_drsparse_acic.append(np.mean((ci_low < cate) & (cate < ci_high)))
        missed_out_drsparse_acic.append(np.mean((ci_low < 0) & (0 < ci_high)))

        cate = df_train['mu1'] - df_train['mu0']

        # In sample ATE
        ate_drsparse_acic.append(model.ate(X = x))

        subtraction = np.subtract(cate, np.reshape(model.effect(X = x), (4000,)))
        subtraction = subtraction**2
        pehe_in_drsparse_acic.append(np.sqrt(sum(subtraction)/4000))

        intervals = model.effect_interval(X = x)
        ci_low = np.reshape(intervals[0], (4000,))
        ci_high = np.reshape(intervals[1], (4000,))
        coverage_in_drsparse_acic.append(np.mean((ci_low < cate) & (cate < ci_high)))
        missed_in_drsparse_acic.append(np.mean((ci_low < 0) & (0 < ci_high)))

    # Linear DR
    model = dr.LinearDRLearner(  random_state=123,
                                       featurizer=PolynomialFeatures(degree = 2, include_bias=False))

    pehe_out_drlin_acic = []
    pehe_in_drlin_acic = []
    coverage_out_drlin_acic = []
    missed_out_drlin_acic = []
    coverage_in_drlin_acic = []
    missed_in_drlin_acic = []
    ate_drlin_acic = []

    for i in tqdm(datas):
        df = i
        df_train = df[0:4000]
        df_test = df[4000:]
        cate = df_test['mu1'] - df_test['mu0']
        y = df_train['y']
        x = X[0:4000]
        model.fit(Y = y, T = df_train['z'], X = x)
        subtraction = np.subtract(cate, np.reshape(model.effect(X = X[4000:]), (802,)))
        subtraction = subtraction**2
        pehe_out_drlin_acic.append(np.sqrt(sum(subtraction)/802))

        intervals = model.effect_interval(X = X[4000:])
        ci_low = np.reshape(intervals[0], (802,))
        ci_high = np.reshape(intervals[1], (802,))
        coverage_out_drlin_acic.append(np.mean((ci_low < cate) & (cate < ci_high)))
        missed_out_drlin_acic.append(np.mean((ci_low < 0) & (0 < ci_high)))

        cate = df_train['mu1'] - df_train['mu0']

        # In sample ATE
        ate_drlin_acic.append(model.ate(X = x))

        subtraction = np.subtract(cate, np.reshape(model.effect(X = x), (4000,)))
        subtraction = subtraction**2
        pehe_in_drlin_acic.append(np.sqrt(sum(subtraction)/4000))

        intervals = model.effect_interval(X = x)
        ci_low = np.reshape(intervals[0], (4000,))
        ci_high = np.reshape(intervals[1], (4000,))
        coverage_in_drlin_acic.append(np.mean((ci_low < cate) & (cate < ci_high)))
        missed_in_drlin_acic.append(np.mean((ci_low < 0) & (0 < ci_high)))


    # Finally we save the data
    final_data_drf = pd.DataFrame(
        np.array([pehe_out_drf_acic, pehe_in_drf_acic, coverage_out_drf_acic, missed_out_drf_acic,
                  coverage_in_drf_acic, missed_in_drf_acic, ate_drf_acic]).T,
        columns =["pehe_out_drf_acic", "pehe_in_drf_acic", "coverage_out_drf_acic", "missed_out_drf_acic",
                  "coverage_in_drf_acic", "missed_in_drf_acic", "ate_drf_acic" ]
    )

    final_data_drsparse = pd.DataFrame(
        np.array([pehe_out_drsparse_acic, pehe_in_drsparse_acic, coverage_out_drsparse_acic, missed_out_drsparse_acic,
                  coverage_in_drsparse_acic, missed_in_drsparse_acic, ate_drsparse_acic]).T,
        columns =["pehe_out_drsparse_acic", "pehe_in_drsparse_acic", "coverage_out_drsparse_acic",
                  "missed_out_drsparse_acic", "coverage_in_drsparse_acic", "missed_in_drsparse_acic",
                  "ate_drsparse_acic" ]
    )

    final_data_drlin = pd.DataFrame(
        np.array([pehe_out_drlin_acic, pehe_in_drlin_acic, coverage_out_drlin_acic, missed_out_drlin_acic,
                  coverage_in_drlin_acic, missed_in_drlin_acic, ate_drlin_acic]).T,
        columns =["pehe_out_drlin_acic", "pehe_in_drlin_acic", "coverage_out_drlin_acic", "missed_out_drlin_acic",
                  "coverage_in_drlin_acic", "missed_in_drlin_acic", "ate_drlin_acic" ]
    )

    final_data_ganite = pd.DataFrame(
        np.array([pehe_in_gan_acic, pehe_out_gan_acic, ate_gan_acic ]).T,
        columns =["pehe_in_gan_acic", "pehe_out_gan_acic", "ate_gan_acic" ]
    )

    final_data_sparse = pd.DataFrame(
        np.array([pehe_out_sparse_acic, pehe_in_sparse_acic, coverage_out_sparse_acic, missed_out_sparse_acic,
                  coverage_in_sparse_acic, missed_in_sparse_acic, ate_sparse_acic]).T,
        columns =["pehe_out_sparse_acic", "pehe_in_sparse_acic", "coverage_out_sparse_acic", "missed_out_sparse_acic",
                  "coverage_in_sparse_acic", "missed_in_sparse_acic", "ate_sparse_acic" ]
    )

    final_data_linear = pd.DataFrame(
        np.array([pehe_out_linear_acic, pehe_in_linear_acic, coverage_out_linear_acic, missed_out_linear_acic,
                  coverage_in_linear_acic, missed_in_linear_acic, ate_linear_acic]).T,
        columns =["pehe_out_linear_acic", "pehe_in_linear_acic", "coverage_out_linear_acic", "missed_out_linear_acic",
                  "coverage_in_linear_acic", "missed_in_linear_acic", "ate_linear_acic" ]
    )

    final_data_cf = pd.DataFrame(
        np.array([pehe_out_cf_acic, pehe_in_cf_acic, coverage_out_cf_acic, missed_out_cf_acic, ate_cf_acic,
                  coverage_in_cf_acic, missed_in_cf_acic ]).T,
        columns =["pehe_out_cf_acic", "pehe_in_cf_acic", "coverage_out_cf_acic", "missed_out_cf_acic", "ate_cf_acic",
                  "coverage_in_cf_acic", "missed_in_cf_acic" ]
    )

    final_data_cfdml = pd.DataFrame(
        np.array([pehe_out_cfdml_acic, pehe_in_cfdml_acic, coverage_out_cfdml_acic, missed_out_cfdml_acic,
                  coverage_in_cfdml_acic, missed_in_cfdml_acic, ate_cfdml_acic ]).T,
        columns =["pehe_out_cfdml_acic", "pehe_in_cfdml_acic", "coverage_out_cfdml_acic", "missed_out_cfdml_acic",
                  "coverage_in_cfdml_acic", "missed_in_cfdml_acic", "ate_cfdml_acic" ]
    )

    final = pd.concat(
        [final_data_drf, final_data_drlin, final_data_drsparse, final_data_sparse, final_data_linear,
         final_data_cf, final_data_cfdml, final_data_ganite, pd.DataFrame(ates, columns = ["ates"]) ],
        axis = 1
    )

    final.to_csv(dir + "/Results/" + str(k) + "_acic_models.csv", index = False)


    path2script = dir + "/R files/ACIC_BART.R"
    try:
        subprocess.call(['Rscript', path2script, dir, str(k)])
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))



