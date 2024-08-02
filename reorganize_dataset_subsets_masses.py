
import argparse
import pandas as pd
import numpy as np
import os
from skimage.io import imsave
from skimage.io import imread


def main(args):
    # read all bounding boxes coordinates to a dataframe:
    df_boxes = pd.read_csv(args.data_boxes)

    # keep only masses i df_boxes:
    df_boxes = df_boxes[df_boxes["AD"] == 0]

    # let me see how many masses and how many ADs we have in biopsied volumes:
    masses_sum = df_boxes[df_boxes["AD"] == 0].shape[0]
    ads_sum = df_boxes[df_boxes["AD"] == 1].shape[0]

    dfs_path = args.dataframes

    df_train = pd.read_csv(dfs_path + "df_train.csv")
    df_valid = pd.read_csv(dfs_path + "df_valid.csv")
    df_test = pd.read_csv(dfs_path + "df_test.csv")


    dataframes = [df_train, df_valid, df_test]

    for subset_df in dataframes:

        # keep biopsied volumes of this subset to a dataframe:
        biopsied_sub_df = subset_df.copy()
        biopsied_sub_df = biopsied_sub_df[(biopsied_sub_df["Benign"] == 1) | (biopsied_sub_df["Cancer"] == 1)]

        # df_biopsied = df[df["StudyUID"].isin(df_boxes["StudyUID"])]

        # keep the df_boxes rows of the studies that are biopsied:
        # (in order to extract the GT box of those)
        biopsied_sub_df_boxes = df_boxes[df_boxes["StudyUID"].isin(biopsied_sub_df["StudyUID"])]

        # masses and ADs of this subset:
        masses_subset_sum = biopsied_sub_df_boxes[biopsied_sub_df_boxes["AD"] == 0].shape[0]
        ads_subset_sum = biopsied_sub_df_boxes[biopsied_sub_df_boxes["AD"] == 1].shape[0]
        print()

        # call function to find unique quadruplets of patient/study/view/central_slice,
        # this is the number of the central slices with GT bounding boxes of the corresponding subset:
        unique_gt_slices = find_unique_quadruplets(df=biopsied_sub_df_boxes)

        subsets_dir = "train/" if subset_df.shape[0] == df_train.shape[0] else "valid/" if subset_df.shape[0] == df_valid.shape[0] else "test/"

        # iterate through all quadruplets:
        for unique_gt_sl in unique_gt_slices:
            # extract the informations:
            pid = unique_gt_sl.split("_", 3)[0]
            sid = unique_gt_sl.split("_", 3)[1]
            view = unique_gt_sl.split("_", 3)[2]
            cslice = int(unique_gt_sl.split("_", 3)[3])

            print("pid:", pid)
            print("sid:", sid)
            print("view:", view)
            print("central slice:", cslice)

            # save the image in the destination folder + /images:

            images_dir = args.dest_dir + "images/" + subsets_dir
            os.makedirs(images_dir, exist_ok=True)

            image_name = "{}TomosynthesisReconstruction_{}_.png".format(view.upper(), cslice)
            image_path = os.path.join(args.images, pid, sid, image_name)

            image = imread(image_path)

            # new filename:
            img_filename = str(unique_gt_sl) + ".png"
            imsave(os.path.join(images_dir, img_filename), image)

            # create a txt file in the destination folder + /labels:
            labels_dir = args.dest_dir + "labels/" + subsets_dir
            os.makedirs(labels_dir, exist_ok=True)

            txt_filename = str(unique_gt_sl) + ".txt"
            f = open(os.path.join(labels_dir, txt_filename), 'w')

            # keep to a dataframe the bounding boxes of this gt slice of the corresponding DICOM:
            df_gt_slice_boxes = df_boxes.loc[(df_boxes['PatientID'] == pid)
                                             & (df_boxes['StudyUID'] == sid)
                                             & (df_boxes['View'] == view)
                                             & (df_boxes['Slice'] == cslice)]

            df_gt_slice_boxes = df_gt_slice_boxes.reset_index(drop=True)

            if df_gt_slice_boxes.shape[0] > 1:
                print("HERE")

            # get the number of gt bounding boxes of this gt central slice:
            bboxes_num = df_gt_slice_boxes.shape[0]

            # iterate for every bounding bx:
            for i in range(bboxes_num):
                # extract class and dimensions x, y, width, height:
                class_id = 0
                x_dim = int(df_gt_slice_boxes["X"][i])
                y_dim = int(df_gt_slice_boxes["Y"][i])
                w_dim = int(df_gt_slice_boxes["Width"][i])
                h_dim = int(df_gt_slice_boxes["Height"][i])

                # apply downscaling in image boxes(because the images are preprocessed wth a downscale factor of 2):
                x_dim = x_dim // 2
                y_dim = y_dim // 2
                w_dim = w_dim // 2
                h_dim = h_dim // 2

                # convert upper left box point to center box point coordinates:
                x_dim = x_dim + (w_dim // 2)
                y_dim = y_dim + (h_dim // 2)

                # finally keep bounding box dimensions in (0.1):
                # divide x and w with image's width
                # divide y and h with image's height
                img_w = image.shape[1]
                img_h = image.shape[0]
                x_dim = x_dim / img_w
                w_dim = w_dim / img_w

                y_dim = y_dim / img_h
                h_dim = h_dim / img_h

                # save the coordinates as a row at the txt file:
                # YOLOV5 TXT PyTorch Anotation format:
                # 1 0.617 0.3594420600858369 0.114 0.17381974248927037

                f.write(str(class_id) + " " + str(x_dim) + " " + str(y_dim) + " " + str(w_dim) + " " + str(h_dim) + "\n")
            f.close()

        print("subset end")

    print("end")


def find_unique_triplets(df):
    all_str_psv = []

    for index, row in df.iterrows():
        all_str_psv.append(
            str(df["PatientID"][index])  + "_" + str(df["StudyUID"][index]) +  "_" + str(df["View"][index]))

    unique_psv = []

    for unique in list(sorted(set(all_str_psv))):
        unique_psv.append(unique)

    return unique_psv


def find_unique_quadruplets(df):
    all_str_psvc = []

    for index, row in df.iterrows():
        all_str_psvc.append(
            str(df["PatientID"][index]) + "_" + str(df["StudyUID"][index]) + "_" + str(df["View"][index]) + "_" + str(df["Slice"][index]))

    unique_psvc = []

    for unique in list(sorted(set(all_str_psvc))):
        unique_psvc.append(unique)

    return unique_psvc



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save the image dataset into a YOLOV5 annotation format."
    )
    parser.add_argument(
        "--data-boxes",
        type=str,
        default="/mnt/seagate/DBT/manifest-1617905855234/BCS-DBT boxes-train-v2.csv",
        help="csv file defining ground truth bounding boxes",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="/mnt/seagate/DBT/TomoImagesPP_WholeDataset_NSR",
        help="root folder with preprocessed images",
    )

    parser.add_argument(
        "--dataframes",
        type=str,
        default="/mnt/seagate/DBT/manifest-1617905855234/subsets_DFs_seed_311/",
        help="root folder with dataframes of train/validation/test subsets",
    )
    # only_biopsied argument doesn't have an impact for now.
    # We can take biopsied only from boxes_csv file:
    parser.add_argument(
        "--only-biopsied",
        default=True,
        action="store_true",
        help="flag to use only biopsied cases",
    )
    parser.add_argument(
        "--dest-dir",
        default="/home/lazaros/PycharmProjects/yolo_new_clone/datasets/dbt_dataset_ONLY-BIOPSIED_masses_NO-CLASSES/",
        help="destination directory to save images and labels directories",
    )

    args = parser.parse_args()
    main(args)