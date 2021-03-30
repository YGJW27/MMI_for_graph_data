# main.py for MI learning
import os
import glob
import time
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.ensemble import RandomForestClassifier
from pyopls import OPLS

from ggmfit import *
from mutual_information import *
from PSO import *
from MI_learning import *
from paper_network import *


def data_list(sample_path):
    sub_dirs = [x[0] for x in os.walk(sample_path)]
    sub_dirs.pop(0)

    data_list = []

    for sub_dir in sub_dirs:
        file_list = []
        dir_name = os.path.basename(sub_dir)
        file_glob = os.path.join(sample_path, dir_name, '*')
        file_list.extend(glob.glob(file_glob))

        for file_name in file_list:
            data_list.append([file_name, dir_name])

    return np.array(data_list)


class MRI_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, idx):
        filepath, target = self.data_list[idx][0], int(self.data_list[idx][1])
        dataframe = pd.read_csv(filepath, sep="\s+", header=None)
        pic = dataframe.to_numpy()

        return pic, target, idx

    def __len__(self):
        return len(self.data_list)


def main():
    parser = argparse.ArgumentParser(description="AD")
    parser.add_argument('-M', '--learnmethod', type=str, default="opls")
    args = parser.parse_args()

    DATA_PATH = "D:/Project/ADNI_data/dataset/ADNI3_ADvsCN_FN/"
    filelist = data_list(DATA_PATH)
    dataset = MRI_Dataset(filelist)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    for data, target, idx in dataloader:
        x = data.numpy()
        y = target.numpy()
        idx = idx.numpy()

    node_idx = nodes_selection_ADNI()
    x = x[:, node_idx, :][:, :, node_idx]

    x = noise_filter(x, 0.05)

    x = np.tanh(x / 10)

    seed = 123456
    np.random.seed(seed)
    starttime = time.time()

    fe_num = 30

    # 10-fold validation
    acc_sum = 0
    TN_sum = 0
    TP_sum = 0
    cv = 10
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    for idx, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        P_num = np.sum(y_test)
        N_num = y_test.shape[0] - P_num

     
        if args.learnmethod == "kpca":
            # matrix to array
            f1_train = x_train.reshape(x_train.shape[0], -1)
            f1_test = x_test.reshape(x_test.shape[0], -1)

            # Norm
            scaler = StandardScaler()
            scaler.fit(f1_train)
            f1scale_train = scaler.transform(f1_train)
            f1scale_test = scaler.transform(f1_test)

            # PCA
            model = KernelPCA(n_components=fe_num, kernel='rbf')
            model.fit(f1scale_train)
            fscale_train = model.transform(f1scale_train)
            fscale_test = model.transform(f1scale_test)

        
        elif args.learnmethod == "ica":
            # matrix to array
            f1_train = x_train.reshape(x_train.shape[0], -1)
            f1_test = x_test.reshape(x_test.shape[0], -1)

            # Norm
            scaler = StandardScaler()
            scaler.fit(f1_train)
            f1scale_train = scaler.transform(f1_train)
            f1scale_test = scaler.transform(f1_test)

            # ICA
            model = FastICA(n_components=fe_num)
            model.fit(f1scale_train, y_train)
            fscale_train = model.transform(f1scale_train)
            fscale_test = model.transform(f1scale_test)


        elif args.learnmethod == "opls":
            # matrix to array
            f1_train = x_train.reshape(x_train.shape[0], -1)
            f1_test = x_test.reshape(x_test.shape[0], -1)

            # Norm
            scaler = StandardScaler()
            scaler.fit(f1_train)
            f1scale_train = scaler.transform(f1_train)
            f1scale_test = scaler.transform(f1_test)

            # OPLS
            model = OPLS(n_components=fe_num, scale=False)
            model.fit(f1scale_train, y_train)
            fscale_train = model.transform(f1scale_train)
            fscale_test = model.transform(f1scale_test)

        else:
            print("Method Input Error!")

        
        # Ramdom Forest
        rf = RandomForestClassifier(max_depth=5, random_state=0)
        model = rf.fit(fscale_train, y_train)


        # SVC
        # svc = SVC(kernel='rbf', random_state=1, gamma=0.0001, C=1000)
        # model = svc.fit(fscale_train, y_train)

        predict_train = model.predict(fscale_train)
        correct_train = np.sum(predict_train == y_train)
        accuracy_train = correct_train / train_idx.size

        predict = model.predict(fscale_test)
        correct = np.sum(predict == y_test)

        TN = np.sum(predict[y_test == 0] == y_test[y_test == 0])
        TP = np.sum(predict[y_test == 1] == y_test[y_test == 1])
        TN_sum = TN_sum + TN
        TP_sum = TP_sum + TP
        # print()
        # print(classification_report(y_test, predict))
        accuracy = correct / test_idx.size
        print("cv: {}/{}, acc.: {:.1f}/{:.1f}\n".format(idx, cv, accuracy_train*100, accuracy*100))
        acc_sum += accuracy
        print("total acc.: {:.1f}\n".format(acc_sum / cv * 100))
        print()

    Acc = (TN_sum + TP_sum) / y.shape[0]
    Sen = TP_sum / np.sum(y)
    Spe = TN_sum / (y.shape[0] - np.sum(y))

    print("ACC: {:.2f}%, SEN: {:.2f}%, SPE: {:.2f}%".format(Acc * 100, Sen * 100, Spe * 100))


    endtime = time.time()
    runtime = endtime - starttime
    print(runtime)


if __name__ == "__main__":
    main()