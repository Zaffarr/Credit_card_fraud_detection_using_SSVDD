import numpy as np
import pandas as pd
from Src_SVDD import BaseSVDD
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold
from pandas import read_csv
import random
from itertools import product


def ssvdd_train(x_train, y_train, iter, C, d, eta, kappa, beta, psi, npt):
    """ Train the SSVDD model for the given data and hyper-parameters

    :param x_train: matrix of floats, training data
    :param y_train: array of -1 or 1, labels for training data; 1 for target (fraudulent) class
    :param iter: int, number of iterations for robust training
    :param C: float, determining the proportion of instances forced to be outliers (hyper-parameter)
    :param d: int, d-dimensions for subspace learning (hyper-paramter)
    :param eta: float, determining the step of gradient (hyper-paramter)
    :param kappa: float, specifying the width of kernel function (hyper-paramter)
    :param beta: float, regularizing the psi parameter (hyper-paramter)
    :param psi: int, specifying the type of regularization term (in constraint function) for training
    :param npt: int; 0 for linear version, 1 for non-linear version of model
    :return: ssvdd_npt - list of values used for npt-based testing for non-linear model
             ssvdd_models - list of models for each iteration
             ssvdd_Q - projection matrix tuned in each iteration
    """

    if npt==1:
        print("NPT-based SSVDD running...")
        z=x_train.T
        N = np.shape(x_train)[0]
        dtrain = np.sum(x_train**2,1).reshape(N,1) @ np.ones(shape=(1,N)) + \
                 (np.sum(x_train**2,1).reshape(N,1)@np.ones(shape=(1,N))).T - \
                 (2*(x_train@z))
        sigma = kappa * np.mean(np.mean(dtrain))
        A = 2.0*sigma
        ktrain_exp = np.exp(-dtrain/A)
        N = np.shape(ktrain_exp)[0]
        ktrain = (np.identity(N)-np.ones(shape=(N,N))/N) @ ktrain_exp @ (np.identity(N)-np.ones(shape=(N,N))/N)

        eig_val, eig_vec = np.linalg.eig(ktrain)
        eig_val, eig_vec = eigen_process(eig_val, eig_vec)
        eigval_acc = np.cumsum(eig_val) / np.sum(eig_val)
        eig_diag = np.diag(eig_val)
        II = np.argwhere(eigval_acc >= 0.99)
        LL = II[0][0]
        eig_diag = eig_diag[0:LL, 0:LL]
        pmat = np.linalg.pinv(np.sqrt(eig_diag) @ eig_vec[:, 0:LL].T)
        phi = ktrain @ pmat
        ssvdd_npt = [1,A,ktrain_exp,phi,x_train]
        x_train = phi
    else:
        print("Linear SSVDD running...")
        ssvdd_npt = [0]  # flag is 0


    Q = np.random.rand(np.shape(x_train)[1], d)
    Q, _ = np.linalg.qr(Q)
    reduced_data = x_train@Q
    model = BaseSVDD(C=C, kernel='rbf', display='off')
    svdd_model = model.fit(reduced_data, y_train)
    ssvdd_Q = [Q]
    ssvdd_models = [model]

    for _ in range(iter):
        alpha_vector  = np.array(svdd_model.get_alpha())
        const = constraints_ssvdd(psi, C, Q, x_train, alpha_vector)
        Sa = x_train.T @ (np.diag(alpha_vector) - (alpha_vector @ alpha_vector.T)) @ x_train
        grad = 2 * (Sa @ Q) + (beta*const)
        Q = Q - eta*grad
        Q, _ = np.linalg.qr(Q)
        reduced_data = x_train @ Q

        model = BaseSVDD(C=C, kernel='rbf', display='off')
        svdd_model = model.fit(reduced_data, y_train)

        ssvdd_Q.append(Q)
        ssvdd_models.append(model)

    return ssvdd_npt, ssvdd_models, ssvdd_Q


def ssvdd_test(test_data, test_labels, ssvdd_model, ssvdd_Q, ssvdd_npt):
    """Testing the model trained or being trained

    :param test_data: matrix of floats, testing data
    :param test_labels:  array of -1 or 1, labels for testing data; 1 for target (fraudulent) class
    :param ssvdd_model: model trained or being trained
    :param ssvdd_Q: projection matrix
    :param ssvdd_npt: list of values used for npt-based testing for non-linear model
    :return: pred_labels, labels predicted for testing data
    """

    if ssvdd_npt[0] == 1:
        print("NPT-based SSVDD testing...")
        kernel_exp_test = rbf_kernel(X=ssvdd_npt[4], Y=test_data, gamma=ssvdd_npt[1])
        phi = ssvdd_npt[3]
        k_train = ssvdd_npt[2]
        N = np.shape(k_train)[1]
        M = np.shape(kernel_exp_test)[1]
        kernel_func_test = (np.identity(N) - np.ones(shape=(N,N))/N) @ \
                           (kernel_exp_test - (k_train @ np.ones(shape=(N,1))/N) @ np.ones(shape=(1,M)))
        test_data = (np.linalg.pinv(phi)@kernel_func_test).T


    else:
        print("Linear SSVDD testing...")

    testdata_red = test_data @ ssvdd_Q
    pred_labels = ssvdd_model.predict(testdata_red)
    return pred_labels


def constraints_ssvdd(psi, C_val, Q, train_data, alpha_vector):
    """computing the regularization term for SSVDD based on psi parameter

    :param psi: int, determining the type of regularization term
    :param C_val: float, specifying the proportion of outliers
    :param Q: projection matrix
    :param train_data: training data
    :param alpha_vector: vector of values for each training instance defining datapoints' location in feature space
    :return: const; regularization term
    """

    if psi == 1:
        const = 0

    elif psi == 2:
        const = 2*(train_data.T @ train_data) @ Q

    elif psi == 3:
        const = 2*train_data.T @ (alpha_vector @ alpha_vector.T) @ (train_data) @ Q

    elif psi == 4:
        temp_alpha_vec = alpha_vector[:]
        temp_alpha_vec[temp_alpha_vec==C_val] = 0
        const = 2*np.transpose(train_data) @ (temp_alpha_vec @ temp_alpha_vec.T) @ (train_data) @ Q

    else:
        print("Only psi 1,2,3 or 4 is possible")
        const = None

    return const


def evaluate(actual, prediction):
    """calculating the performance metrics; accuracy, precision, tpr, tnr, f1-measure, and g-mean

    :param actual: ground truths
    :param prediction: predictions for the data
    :return: eval_metrics, list of performance metrics mentioned above
    """

    positive = 1
    negative = -1
    actual = np.ravel(actual).astype(int)
    prediction = np.array(prediction).astype(int)

    tp = np.sum(np.logical_and(prediction == positive, actual == positive))
    tn = np.sum(np.logical_and(prediction == negative, actual == negative))
    fp = np.sum(np.logical_and(prediction == positive, actual == negative))
    fn = np.sum(np.logical_and(prediction == negative, actual == positive))

    p = tp+fn
    n = fp+tn
    N = p+n
    print(f"tn: {tn}")
    print(f"fn: {fn}")
    print(f"tp: {tp}")
    print(f"fp: {fp}")
    print(p, n, N)

    tp_rate = tp/p if p != 0 else 0
    tn_rate = tn/n if n != 0 else 0

    try:
        accuracy = accuracy_score(actual, prediction)
    except ZeroDivisionError:
        accuracy = 0
    try:
        precision = precision_score(actual, prediction, average='binary')
    except ZeroDivisionError:
        precision = 0
    try:
        recall = recall_score(actual, prediction, average='binary')
    except ZeroDivisionError:
        recall = 0
    try:
        f_measure = f1_score(actual, prediction, average='binary')
    except ZeroDivisionError:
        f_measure = 0
    try:
        g_mean = geometric_mean_score(actual, prediction)
    except ZeroDivisionError:
        g_mean = 0

    eval_metrics = [accuracy, tp_rate, tn_rate, precision, f_measure, g_mean]
    eval_metrics = [round(i,3) for i in eval_metrics]

    return eval_metrics


def eigen_process(eig_val, eig_vec):
    """Extracting the positive eigen values and eigen vectors, sorting them and eliminating the non-valid
    (infinite and NaN) values from them

    :param eig_val: Eigen values
    :param eig_vec: Eigen vectors
    :return: eig_val, eig_vec; processed eigen values and vectors
    """

    if np.any(np.iscomplex(eig_val)):
        eig_val = np.abs(eig_val)
        eig_vec = np.abs(eig_vec)

    eig_val[~np.isfinite(eig_val)] = 0.0
    eig_vec[~np.isfinite(eig_vec)] = 0.0

    eig_val[eig_val < 1e-6] = 0.0

    positive_eigval_indices = eig_val > 0
    eig_val = eig_val[positive_eigval_indices]
    eig_vec = eig_vec[:, positive_eigval_indices]

    sorted_indices = np.argsort(-eig_val)
    eig_val = eig_val[sorted_indices]
    eig_vec = eig_vec[:, sorted_indices]

    if eig_val.size == 0:
        eig_val = np.array([0.0])
        eig_vec = np.zeros(len(eig_val), 1)

    return eig_val, eig_vec


def cross_validate(x_train, y_train, iter, C_values, d_values, eta_values, kappa_values, beta_values, psi, npt):
    """Cross Validation

    :param x_train: training data
    :param y_train: labels for training data
    :param iter: no. of iterations for robust training
    :param C_values: list of C values for hyper-parameter tuning
    :param d_values: list of d values for hyper-parameter tuning
    :param eta_values: list of eta values for hyper-parameter tuning
    :param kappa_values: list of kappa values for hyper-parameter tuning
    :param beta_values: list of beta values for hyper-parameter tuning
    :param psi: int, specifying the regularization term
    :param npt: 0 for linear, 1 for non-linear model
    :return: params, Dataframe of best performing hyper-parameters based on g-mean
    """

    skf = StratifiedKFold(n_splits=5)

    df = pd.DataFrame(columns=['g-mean', 'C', 'd', 'eta', 'kappa', 'beta'])
    count = 0

    param_combinations = product(C_values, d_values, eta_values, kappa_values, beta_values)

    for i,j,k,l,m in param_combinations:
        gscore = []
        for _, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
            x_tr, y_tr = x_train[train_index], y_train[train_index]
            x_te, y_te = x_train[test_index], y_train[test_index]

            ssvdd_npt, ssvdd_models, ssvdd_Q = ssvdd_train(x_tr, y_tr, iter, i, j, k, l, m, psi, npt)
            pred_labels = ssvdd_test(x_te, y_te, ssvdd_models[-1], ssvdd_Q[-1], ssvdd_npt)
            eval = evaluate(np.array(y_te), np.array(pred_labels))
            gscore.append(eval[5])

        mean_gscore = np.mean(gscore)
        df.loc[count] = [mean_gscore, i, j, k, l, m]
        count += 1

    df['g-mean'] = pd.to_numeric(df['g-mean'])
    params = df.iloc[df['g-mean'].idxmax()]

    return params

def main():

    target_class = 1.0
    nontarget_class = -1.0

    iter = 1
    subsets = 2

    C_values = [0.2]
    d_values = [5, 10]
    eta_values = [0.1]
    kappa_values = [0.1]
    psi_values = [1, 2, 3]
    npt_values = [0, 1]
    beta_values = [0.1]

    df_eval = pd.DataFrame(columns=['psi', 'npt', 'accuracy', 'tp-rate', 'tn-rate', 'precision', 'f-measure', 'g-mean'])

    xtrain_dataset = []
    ytrain_dataset = []
    xtest_dataset = []
    ytest_dataset = []

    for i in range(1, subsets+1):
        xtrain_dataset.append(read_csv(f'xtrain{i}.csv', header=None).to_numpy())
        ytrain_dataset.append(read_csv(f'ytrain{i}.csv', header=None).to_numpy())
        xtest_dataset.append(read_csv(f'xtest{i}.csv', header=None).to_numpy())
        ytest_dataset.append(read_csv(f'ytest{i}.csv', header=None).to_numpy())

    for psi in psi_values:
        for npt in npt_values:
            pred_labels = []
            evaluations = []
            for i in range(subsets):
                target_index = np.argwhere(ytrain_dataset[i] == target_class)
                target_index = target_index[:, 0]
                ytr = ytrain_dataset[i][target_index, :]
                xtr = xtrain_dataset[i][target_index, :]

                cv_param = cross_validate(xtr, ytr, iter, C_values, d_values, eta_values,
                                          kappa_values, beta_values, psi, npt)
                ssvdd_npt, ssvdd_models, ssvdd_Q = ssvdd_train(xtr, ytr, iter, cv_param[1], int(cv_param[2]),
                                                               cv_param[3], cv_param[4], cv_param[5], psi, npt)
                pred_labels.append(ssvdd_test(xtest_dataset[i], ytest_dataset[i], ssvdd_models[-1],
                                              ssvdd_Q[-1], ssvdd_npt))
                evaluations.append(evaluate(ytest_dataset[i], pred_labels[i]))
            mean_evaluations = list(np.array(evaluations).mean(axis=0))
            results = [psi, npt] + mean_evaluations
            df_eval.loc[len(df_eval)] = results

    print(df_eval)
    df_eval.to_excel('SSVDD_results.xlsx', index=False)

if __name__ == "__main__":
    main()

