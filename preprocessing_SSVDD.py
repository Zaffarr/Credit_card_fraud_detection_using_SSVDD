import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split

TARGET_CLASS = 1.0  # non-fraudulent class
NON_TARGET_CLASS = -1.0  # fraudulent class
SUBSETS = 2

def preprocess_data(x, y):
    """Resampling to extract a smaller portion of training data and normalizing the resampled training data and
    test data. The mean and standard deviation is calculated from the original (before resampling) train data
    to normalize both resampled-training data and testing data

    :param x: data
    :param y: labels; 1 for target class and -1 for nontarget class
    :return:
    """
    xtrain_datasets, ytrain_datasets, xtest_datasets, ytest_datasets = [], [], [], []

    for i in range(SUBSETS):
        # splitting data into test and train data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

        # Data reduction
        reduced_train_data = reduce_data(x_train, y_train)

        # Normalization
        norm_traindata, norm_testdata = normalize_data(reduced_train_data, x_test, x_train, y_train)

        save_to_csv(norm_traindata, norm_testdata, y_train, y_test, i)

        # Append datasets
        xtrain_datasets.append(norm_traindata)
        xtest_datasets.append(norm_testdata)
        ytrain_datasets.append(y_train)
        ytest_datasets.append(y_test)

    return xtrain_datasets, ytrain_datasets, xtest_datasets, ytest_datasets

def reduce_data(x_train, y_train):
    """Resampling the training data

    :param x_train: training data
    :param y_train: labels for training data
    :return: reduced_data, resampled training data
    """
    target_ind = np.where(y_train == TARGET_CLASS)[0]
    reduced_target_ind = np.random.choice(target_ind, 1500, replace=False)  # 1500 non-fraudulent instances

    nontarget_ind = np.where(y_train == NON_TARGET_CLASS)[0]
    reduced_nontarget_ind = np.random.choice(nontarget_ind, 500, replace=False) # 500 fraudulent instances

    reduced_data = np.concatenate((x_train[reduced_target_ind], x_train[reduced_nontarget_ind]), axis=0)
    reduced_data = pd.DataFrame(reduced_data).sample(frac=1).to_numpy()

    return reduced_data

def normalize_data(train_data, test_data, x_train, y_train):
    """Normalizing the data. The mean and standard deviation is calculated from the original (before resampling)
    train data to normalize both resampled-training data and testing data

    :param train_data: resampled train data
    :param test_data: testing data
    :param x_train: training data before resampling, used in calculation of mean and standard deviation
    :param y_train: labels for resampled training data, used in calculation of mean and standard deviation
    :return: norm_traindata, norm_testdata; normalized training and testing data
    """
    target_ind = np.where(y_train == TARGET_CLASS)[0]
    target_traindata = x_train[target_ind]
    mean = np.mean(target_traindata, axis=0)
    st_dev = np.std(target_traindata, axis=0)
    st_dev[st_dev == 0] = 1

    norm_traindata = (train_data - mean) / st_dev
    norm_testdata = (test_data - mean) / st_dev

    return norm_traindata, norm_testdata

def save_to_csv(x_train, x_test, y_train, y_test, i):
    """Making csv files of the datasets

    :param x_train: training data
    :param x_test: testing data
    :param y_train: labels for training data
    :param y_test: labels for testing data
    :param i: iterator
    :return: None
    """
    pd.DataFrame(x_train).to_csv(f"xtrain{i+1}.csv", index=False, header=False)
    pd.DataFrame(x_test).to_csv(f"xtest{i+1}.csv", index=False, header=False)
    pd.DataFrame(y_train).to_csv(f"ytrain{i+1}.csv", index=False, header=False)
    pd.DataFrame(y_test).to_csv(f"ytest{i+1}.csv", index=False, header=False)

if __name__ == "__main__":
    df = read_csv('creditcard_pseudodata.csv', header=None)
    data = df.to_numpy()
    y = data[0:2999, -1].reshape(2999, 1) # working on only 3000 instances
    x = data[0:2999, 1:-1]

    xtrain_datasets, ytrain_datasets, xtest_datasets, ytest_datasets = preprocess_data(x, y)