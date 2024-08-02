import os

import numpy as np
import pandas as pd
import random

# use it only manually to check the file alone:
my_seed=311
random.seed(my_seed)

# desired splitting ratios:
train_ratio = 0.5
valid_ratio = 0.2

def data_frame_subset(csv_views, csv_boxes, seed=42):

    df_all = pd.read_csv(
        csv_views, dtype={
            "Normal": np.int, "Actionable": np.int, "Benign": np.int, "Cancer": np.int
        }
    )

    df_box = pd.read_csv(csv_boxes)
    df_box = df_box[df_box["PatientID"].isin(df_all["PatientID"])]
    df_box["Diag"] = np.sqrt((df_box["Width"] ** 2 + df_box["Height"] ** 2))

    # set of all unique patients:
    pat_set = set()
    for pat in range(df_all.shape[0]):
        pat_set.add(str(df_all.loc[pat]["PatientID"]))

    # convert the set to a list and sample from it, to keep the same order of patients:
    # this helps for reproducibility when using the same seed.
    pat_list = list(pat_set)
    pat_list.sort()

    # compute the number of patients in three subsets based on the corresponding ratios:
    num_train_pat = int(np.floor(train_ratio * len(pat_list)))
    num_valid_pat = int(np.ceil(valid_ratio * len(pat_list)))

    # declare lists to hold the three subsets:
    train_list = []
    valid_list = []
    test_list = []

    # sample 151 patients (for the training) from the above list:
    valid_train_ids = random.sample(pat_list, num_train_pat)
    pat_train = set(valid_train_ids)
    train_list.extend(valid_train_ids)

    # delete the 151 sampled patients:
    pat_set_no_train = pat_set.difference(pat_train)

    # convert the set to a list :
    pat_set_no_train_list = list(pat_set_no_train)
    pat_set_no_train_list.sort()

    # sample 60 patients (for the validation) from the above list:
    valid_valid_ids = random.sample(pat_set_no_train_list, num_valid_pat)
    pat_valid = set(valid_valid_ids)
    valid_list.extend(valid_valid_ids)

    # delete the 60 sampled patients :
    pat_set_no_train_no_val = pat_set_no_train.difference(pat_valid)

    # keep the remaining patients in test set:
    test_list.extend(pat_set_no_train_no_val)

    df_train = df_all[df_all["PatientID"].isin(train_list)]
    df_valid = df_all[df_all["PatientID"].isin(valid_list)]
    df_test = df_all[df_all["PatientID"].isin(test_list)]


    saving_dir = "/mnt/seagate/DBT/manifest-1617905855234/subsets_DFs_seed_" + str(my_seed) +"/"
    os.makedirs(saving_dir, exist_ok=True)

    df_train.to_csv(saving_dir + "df_train.csv", index = False)
    df_valid.to_csv(saving_dir + "df_valid.csv", index = False)
    df_test.to_csv(saving_dir + "df_test.csv", index = False)

    return df_train, df_valid, df_test, df_all


if __name__ == "__main__":
    # for subset in ["all", "train", "validation", "test"]:

    dftrain, dfvalid, dftest,dfall = data_frame_subset(
        "/mnt/seagate/DBT/manifest-1617905855234/BCS-DBT labels-new-v0.csv",
        "/mnt/seagate/DBT/manifest-1617905855234/BCS-DBT boxes-train-v2.csv",
        # subset,
        seed=42
    )

    dfs = [dfall, dftrain, dfvalid, dftest]

    for index in range(len(dfs)):

        df = dfs[index]

        if index == 0:
            subset_pr = "All"
        elif index == 1:
            subset_pr = "Train"
        elif index == 2:
            subset_pr = "Validation"
        else:
            subset_pr = "Test"

        print(subset_pr, ":")
        # keep the biopsied cases:
        df_boxes = pd.read_csv("/mnt/seagate/DBT/manifest-1617905855234/BCS-DBT boxes-train-v2.csv")

        df_biopsied = df[(df["Benign"] == 1) | (df["Cancer"] == 1)]
        df_benign = df_biopsied[df_biopsied["Benign"] == 1]
        df_cancer = df_biopsied[df_biopsied["Cancer"] == 1]

        df_normal = df[df["Normal"] == 1]

        print("Volumes: {}".format(len(df)), end="       ")
        print("Biopsied Volumes: {}".format(len(df_biopsied)), end="       ")
        print("Benign Volumes: {}".format(len(df_benign)), end="       ")
        print("Cancer Volumes: {}".format(len(df_cancer)), end="       ")
        print("Normal Volumes: {}".format(len(df_normal)))

        print("Studies: {}".format(len(set(df["StudyUID"]))), end="       ")
        print("Biopsied Studies: {}".format(len(set(df_biopsied["StudyUID"]))), end="        ")
        print("Benign Studies: {}".format(len(set(df_benign["StudyUID"]))), end="       ")
        print("Cancer Studies: {}".format(len(set(df_cancer["StudyUID"]))), end="       ")
        print("Normal Studies: {}".format(len(set(df_normal["StudyUID"]))))

        print("Patients: {}".format(len(set(df["PatientID"]))), end="      ")
        print("Biopsied Patients: {}".format(len(set(df_biopsied["PatientID"]))), end="      ")
        print("'Benign' Patients: {}".format(len(set(df_benign["PatientID"]))), end="     ")
        print("'Cancer' Patients: {}".format(len(set(df_cancer["PatientID"]))), end="   ")
        print("'Normal' Patients: {}".format(len(set(df_normal["PatientID"]))))
        print()