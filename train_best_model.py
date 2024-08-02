import argparse
import os
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from matplotlib import pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg
from mlflow import log_metric, log_param, get_artifact_uri
from skimage.io import imsave
from sklearn.model_selection import ParameterGrid
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

import random

from dataset import TomoDetectionDataset as Dataset
from dense_yolo import DenseYOLO
from loss import objectness_module, LocalizationLoss
from sampler import TomoBatchSampler
from transform import transforms

from subsets_split import data_frame_subset

def main(args):

    run_example = "Run_90"

    # choosing the parameters for the corresponding run:
    if run_example == "Run_90":
        validation_interval_ex = 1
        schedule_patience_ex = 30
        factor_ex = 0.4
    elif run_example == "Run_58":
        validation_interval_ex = 1
        schedule_patience_ex = 30
        factor_ex = 0.9
    elif run_example == "Run_102":
        validation_interval_ex = 5
        schedule_patience_ex = 10
        factor_ex = 0.3
    elif run_example == "Run_6":
        validation_interval_ex = 1
        schedule_patience_ex = 50
        factor_ex = 0.1
    elif run_example == "Run_22":
        validation_interval_ex = 1
        schedule_patience_ex = 5
        factor_ex = 0.5
    elif run_example == "Run_34":
        validation_interval_ex = 5
        schedule_patience_ex = 10
        factor_ex = 0.1
    elif run_example == "Run_38":
        validation_interval_ex = 1
        schedule_patience_ex = 50
        factor_ex = 0.3
    elif run_example == "Run_42":
        validation_interval_ex = 1
        schedule_patience_ex = 50
        factor_ex = 0.5
    elif run_example == "debugging":
        validation_interval_ex = 1
        schedule_patience_ex = 5
        factor_ex = 0.3

    elif run_example == "Run_90_factor_09":
        validation_interval_ex = 1
        schedule_patience_ex = 30
        factor_ex = 0.9
    # this is for no specific run examples. Not so recommended to use:
    else:
        validation_interval_ex = 5
        schedule_patience_ex = 10
        factor_ex = 0.1

    # choose the destination folder based on the only_biopsied parameter:
    if args.only_biopsied == True:
        mlruns_path_chosen = "data/mlruns/best_config2"
    else:
        mlruns_path_chosen = "data/mlruns/best_config2_whole"

    # seed_list = [42, 536, 311, 1234]
    seed_list = [311]

    running_experiment_name = int(args.experiment_name)

    for running_seed in seed_list:

        torch.backends.cudnn.benchmark = True

        # seed the numpy random generator for using it to reproduce training batch indexes:
        rng_ob = np.random.RandomState(seed=running_seed)
        random.seed(running_seed)
        np.random.seed(running_seed)
        torch.manual_seed(running_seed)

        device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

        loader_train, loader_valid, loader_test = data_loaders(args, rng_ob, running_seed)
        loaders = {"train": loader_train, "valid": loader_valid}


        hparams_dict = {
            "block_config": [(1, 3, 2, 6, 4)],
            "num_init_features": [8],
            "growth_rate": [8],
            "bn_size": [4],
        }
        hparams = list(ParameterGrid(hparams_dict))  # 1 config


        loss_params_dict = [
            # {"loss": ["CE", "weighted-CE"], "alpha": [0.25, 0.5, 1.0]},  # 6 configs
            {"loss": ["CE"], "alpha": [0.5]},  # 1 config
        ]

        loss_params = list(ParameterGrid(loss_params_dict))

        # loss_params = loss_params * 2  # 12 configs

        run_params_dict = {
            "batch_size" : args.batch_size,
            "epochs": args.epochs,
            "patience" : args.patience,
            "init_lr": args.lr,
            "slice_offset": args.slice_offset,
            "images" : args.images,
            "only_biopsied" : args.only_biopsied,
            "workers" : args.workers,
            "seed": running_seed,
            "validation_interval": validation_interval_ex,
            "schedule_patience": schedule_patience_ex,
            "factor:": factor_ex,
            "Run_Example": run_example
        }

        try:
            mlflow.set_tracking_uri(mlruns_path_chosen)
            experiment_id = (
                args.experiment_id
                if args.experiment_id
                else mlflow.create_experiment(name= str(running_experiment_name))
            )
        except Exception as _:
            print("experiment-id must be unique")
            return

        for i, loss_param in tqdm(enumerate(loss_params)):

            for j, hparam in enumerate(hparams):

                with mlflow.start_run(experiment_id=experiment_id):
                    mlflow_log_params(loss_param, hparam, run_params_dict)

                    # try:
                    yolo = DenseYOLO(img_channels=1, out_channels=Dataset.out_channels, **hparam)
                    yolo.to(device)

                    objectness_loss = objectness_module(
                        name=loss_param["loss"], args=argparse.Namespace(**loss_param)
                    )
                    localization_loss = LocalizationLoss(weight=args.loc_weight)

                    optimizer = optim.Adam(yolo.parameters(), lr=args.lr)

                    # use scheduler for learning rate reduction strategy:
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", min_lr=0.000001, factor=factor_ex, patience=schedule_patience_ex, verbose=True)

                    early_stop = args.patience
                    run_tpr2 = 0.0
                    run_tpr1 = 0.0
                    run_auc = 0.0

                    run_tpr2_train = 0.0
                    run_tpr1_train = 0.0
                    run_auc_train = 0.0

                    # declare lists to save loss and tpr per epoch:
                    train_losses = []
                    valid_losses = []
                    tpr2_train = []
                    tpr2_valid = []

                    lr_values_per_epoch = []

                    color_train = "darkorange"
                    color_valid = "blue"
                    color_test = "black"

                    # epochs loop:
                    for epoch_iter in range(args.epochs):
                        print("epoch:", epoch_iter+1)

                        running_train_loss = 0.0
                        running_valid_loss = 0.0
                        epoch_scores_train = []
                        epoch_scores_valid = []

                        if early_stop == 0:
                            break

                        # append the learning rate value of this epoch to the list:
                        lr_values_per_epoch.append(optimizer.param_groups[0]['lr'])

                        # new early stopping method based on lr value:
                        if optimizer.param_groups[0]['lr'] <= 1e-6:
                            print("Learning rate is:", optimizer.param_groups[0]['lr'], "so training stops!")
                            break

                        # phases loop:
                        for phase in ["train", "valid"]:
                            if phase == "train":
                                yolo.train()            # training mode for training phase
                                early_stop -= 1
                            else:
                                yolo.eval()             # evaluation mode for validation phase

                            df_training_pred = pd.DataFrame()
                            train_target_nb = 0

                            df_validation_pred = pd.DataFrame()
                            valid_target_nb = 0

                            # batches loop for training and validation:
                            for batch_index, data in enumerate(loaders[phase]):
                                x, y_true = data
                                x, y_true = x.to(device), y_true.to(device)

                                x_max = torch.amax(x)
                                x_min = torch.amin(x)

                                optimizer.zero_grad()

                                with torch.set_grad_enabled(phase == "train"):
                                    y_pred = yolo(x)
      
                                    obj = objectness_loss(y_pred, y_true)

                                    loc = localization_loss(y_pred, y_true)
                                    total_loss = obj + loc

                                    # training loss computation and batch evaluation:
                                    if phase == "train":
                                        # optimizer.zero_grad()
                                        total_loss.backward()
                                        clip_grad_norm_(yolo.parameters(), 0.5)
                                        optimizer.step()

                                        y_true_np = y_true.detach().cpu().numpy()
                                        train_target_nb += np.sum(y_true_np[:, 0])
                                        df_train_batch_pred = evaluate_batch(y_pred=y_pred, y_true=y_true)

                                        df_training_pred = df_training_pred.append(
                                            df_train_batch_pred, ignore_index=True, sort=False
                                        )

                                        # update the training loss :
                                        running_train_loss += total_loss.item()
                                        if (batch_index + 1) == len(loaders[phase]):
                                            epoch_avg_train_loss = running_train_loss / len(loaders['train'])

                                    # validation loss computation and batch evaluation:
                                    else:

                                        if (epoch_iter+1) % validation_interval_ex == 0:
                                            y_true_np = y_true.detach().cpu().numpy()
                                            valid_target_nb += np.sum(y_true_np[:, 0])
                                            df_valid_batch_pred = evaluate_batch(y_pred=y_pred, y_true=y_true)

                                            df_validation_pred = df_validation_pred.append(
                                                df_valid_batch_pred, ignore_index=True, sort=False
                                            )

                                            # update the validation loss :
                                            running_valid_loss += total_loss.item()
                                            if (batch_index + 1) == len(loaders[phase]):
                                                epoch_avg_valid_loss = running_valid_loss / len(loaders['valid'])

                            # print this only once (e.g in training phase):
                            if phase == "train":
                                # print the learning rate from ptimizer:
                                print("epoch:", epoch_iter + 1, "-> lr:", optimizer.param_groups[0]['lr'])

                            # evaluation of training set with froc curve:
                            if phase == "train":
                                if (epoch_iter + 1) % validation_interval_ex == 0:
                                    print("training +++")
                                    tpr_train, fps_train, svd_thresh_train = froc(df_training_pred, train_target_nb)
                                    epoch_tpr2_train = np.interp(2.0, fps_train, tpr_train)
                                    epoch_tpr1_train = np.interp(1.0, fps_train, tpr_train)
                                    if epoch_tpr2_train > run_tpr2_train:
                                        run_tpr2_train = epoch_tpr2_train
                                        run_tpr1_train = epoch_tpr1_train
                                        run_auc_train = np.trapz(tpr_train, fps_train)

                                        imsave(
                                            os.path.join(get_artifact_uri(), "froc_train.png"),
                                            plot_froc(fps=fps_train,
                                                      tpr=tpr_train,
                                                      subset=str(phase),
                                                      color=color_train,
                                                      )
                                        )

                            # evaluation validation set with froc curve:
                            if phase == "valid":
                                if (epoch_iter + 1) % validation_interval_ex == 0:
                                    print("validation +++")
                                    tpr_valid, fps_valid, svd_thresh_valid = froc(df_validation_pred, valid_target_nb)

                                    # take the threshold for number of FPs = 2.0:
                                    extracted_threshold_value = 0.0
                                    extracted_tpr_value = 0.0
                                    extracted_fps_value = 0.0
                                    for i, elem in reversed(list(enumerate(fps_valid))):
                                        if elem <= 2.0:
                                            extracted_threshold_value = svd_thresh_valid[i]
                                            extracted_tpr_value = tpr_valid[i]
                                            extracted_fps_value = elem
                                            break

                                    epoch_tpr2 = np.interp(2.0, fps_valid, tpr_valid)
                                    epoch_tpr1 = np.interp(1.0, fps_valid, tpr_valid)
                                    if epoch_tpr2 > run_tpr2:
                                        early_stop = args.patience
                                        run_tpr2 = epoch_tpr2
                                        run_tpr1 = epoch_tpr1
                                        run_auc = np.trapz(tpr_valid, fps_valid)
                                        torch.save(
                                            yolo.state_dict(),
                                            os.path.join(get_artifact_uri(), "yolo.pt"),
                                        )
                                        imsave(
                                            os.path.join(get_artifact_uri(), "froc_valid.png"),
                                            plot_froc(fps=fps_valid,
                                                      tpr=tpr_valid,
                                                      subset=str(phase),
                                                      color=color_valid,
                                                      )
                                        )

                        # keep the average loss from all training batches for every epoch:
                        train_losses.append(epoch_avg_train_loss)

                        # keep the average loss from all validation batches for the epochs that the validation is being applied (validation_interval):
                        # also keep the TPR2 value for training and validation subsets for those epochs:
                        if (epoch_iter + 1) % validation_interval_ex == 0:
                            valid_losses.append(epoch_avg_valid_loss)
                            tpr2_train.append(epoch_tpr2_train)
                            tpr2_valid.append(epoch_tpr2)

                        # append the average train loss from the current epoch to the scheduler:
                        if (epoch_iter + 1) % validation_interval_ex == 0:
                            scheduler.step(epoch_tpr2)

                    log_metric("TPR2_valid", run_tpr2)
                    log_metric("TPR1_valid", run_tpr1)
                    log_metric("AUC_valid", run_auc)
                    log_metric("TPR2_train", run_tpr2_train)
                    log_metric("TPR1_train", run_tpr1_train)
                    log_metric("AUC_train", run_auc_train)
                    log_metric("value_threshold", extracted_threshold_value)
                    log_metric("value_TPR", extracted_tpr_value)
                    log_metric("value_FPs", extracted_fps_value)

                    destination_path = mlflow.get_artifact_uri()

                    with open(os.path.join(destination_path, "validation_TPR2_thresh_values.txt"), 'w') as f:
                        f.write(str(svd_thresh_valid))
                        f.close()

                    with open(os.path.join(destination_path, "validation_TPR_values.txt"), 'w') as f:
                        f.write(str(tpr_valid))
                        f.close()

                    with open(os.path.join(destination_path, "validation_FPs_values.txt"), 'w') as f:
                        f.write(str(fps_valid))
                        f.close()

                    imsave(
                        os.path.join(get_artifact_uri(), "loss_train.png"),
                        plot_loss_train(train_losses=train_losses,
                                        color_train=color_train)
                    )
                    imsave(
                        os.path.join(get_artifact_uri(), "loss_valid.png"),
                        plot_loss_valid(train_losses=train_losses,
                                        valid_losses=valid_losses,
                                        val_interv=validation_interval_ex,
                                        color_valid=color_valid)
                    )
                    imsave(
                        os.path.join(get_artifact_uri(), "losses.png"),
                        plot_losses(train_losses=train_losses,
                                    valid_losses=valid_losses,
                                    val_interv=validation_interval_ex,
                                    color_train=color_train,
                                    color_valid=color_valid)
                    )
                    imsave(
                        os.path.join(get_artifact_uri(), "tpr2_valid.png"),
                        plot_tpr2(vals=tpr2_valid,
                                  color=color_valid)
                    )

                    imsave(
                        os.path.join(get_artifact_uri(), "tpr2_train.png"),
                        plot_tpr2(vals=tpr2_train,
                                  color=color_train)
                    )

                    imsave(
                        os.path.join(get_artifact_uri(), "tpr2_train_&_valid.png"),
                        plot_tpr2_together(vals_train=tpr2_train,
                                           vals_valid=tpr2_valid,
                                           val_interv=validation_interval_ex,
                                           color_train=color_train,
                                           color_valid=color_valid)
                    )

                    imsave(
                        os.path.join(get_artifact_uri(), "lr_per_epoch.png"),
                        plot_loss_learning_rate(values=lr_values_per_epoch,
                                                color_val="red")
                    )

                    # In each combination of hyper-parameters, a best model is produced,
                    # based on evaluation of validation set from all the running epochs:

                    # Retrieve this best model of this configuration for testing:
                    model_path = mlflow.get_artifact_uri(artifact_path="yolo.pt")

                    with torch.set_grad_enabled(False):
                        yolo = DenseYOLO(img_channels=1, out_channels=Dataset.out_channels, **hparam)
                        yolo.to(device)

                        # load best model's state of weights:
                        state_dict = torch.load(model_path)
                        yolo.load_state_dict(state_dict)
                        yolo.eval()             # evaluation mode for testing

                        # initialize losses for testing:
                        objectness_loss = objectness_module(
                            name=loss_param["loss"], args=argparse.Namespace(**loss_param)
                        )
                        localization_loss = LocalizationLoss(weight=args.loc_weight)

                        df_test_pred = pd.DataFrame()
                        test_target_nb = 0

                        loss_sum_running_batch = 0.0
                        loss_batches_avg = 0.0

                        epoch_scores_test = []

                        # batch loop for testing:
                        for batch_index, data in enumerate(loader_test):
                            x, y_true = data
                            x, y_true = x.to(device), y_true.to(device)

                            optimizer.zero_grad()

                            y_pred = yolo(x)

                            obj = objectness_loss(y_pred, y_true)
                            loc = localization_loss(y_pred, y_true)
                            loss_batch = obj + loc

                            # evaluation of testing batch:
                            print("evaluating testing batch...")
                            y_true_np = y_true.detach().cpu().numpy()
                            test_target_nb += np.sum(y_true_np[:, 0])

                            df_test_batch_pred = evaluate_batch(y_pred=y_pred, y_true=y_true)

                            df_test_pred = df_test_pred.append(
                                df_test_batch_pred, ignore_index=True, sort=False
                            )

                            # print loss of every batch:
                            print("batch:", batch_index + 1, ", testing loss:", loss_batch)

                            # keep the sum of batch losses till now:
                            loss_sum_running_batch += loss_batch.item()

                            # print the average loss of all batches:
                            if (batch_index + 1) == len(loader_test):
                                # print("batch:", batch_index+1, "train averaging")
                                loss_batches_avg = loss_sum_running_batch / len(loader_test)
                                print("batch:", batch_index + 1, ", avg testing loss:", loss_batches_avg)

                        # evaluation of testing set with froc curve:
                        print("frocing +++")
                        tpr_test, fps_test, svd_thresh_test = froc(df_test_pred, test_target_nb)
                        tpr2_test = np.interp(2.0, fps_test, tpr_test)
                        tpr1_test = np.interp(1.0, fps_test, tpr_test)
                        auc_test = np.trapz(tpr_test, fps_test)

                        print()
                        print("Validation Results:")
                        print("validation TPR2:", run_tpr2)
                        print("validation TPR1:", run_tpr1)
                        print("validation AUC:", run_auc)

                        print()
                        print("Test Results:")
                        print("test TPR2:", tpr2_test)
                        print("test TPR1:", tpr1_test)
                        print("test AUC:", auc_test)

                        imsave(
                            os.path.join(get_artifact_uri(), "froc_test.png"),
                            plot_froc(fps=fps_test,
                                      tpr=tpr_test,
                                      subset="test",
                                      color=color_test
                                      )
                        )

                        log_metric("test_avg_batch_loss", loss_batches_avg)
                        log_metric("TPR2_test", tpr2_test)
                        log_metric("TPR1_test", tpr1_test)
                        log_metric("AUC_test", auc_test)

        running_experiment_name += 1


def mlflow_log_params(loss_param, hparam, run_params_dict):
    for key in loss_param:
        log_param(key, loss_param[key])
    log_param("loss_fun", str(loss_param))
    for key in hparam:
        log_param(key, hparam[key])
    log_param("network", str(hparam))
    log_param("run_params", str(run_params_dict))


def data_loaders(args, rng_ob, running_seed):
    dataset_train, dataset_valid, dataset_test = datasets(args, running_seed)
    sampler_train = TomoBatchSampler(
        batch_size=args.batch_size, data_frame=dataset_train.data_frame,
        # seed for the sampler:
        rng_ob=rng_ob
    )

    def worker_init(worker_id):
        np.random.seed(running_seed)

    loader_train = DataLoader(
        dataset_train,
        batch_sampler=sampler_train,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid, loader_test


def datasets(args, running_seed):

    df_train, df_valid, df_test, df_all = splitting_dataset(args, running_seed)

    train = Dataset(
        csv_views=args.data_views,
        csv_bboxes=args.data_boxes,
        root_dir=args.images,
        subset="train",
        random=True,
        only_biopsied=args.only_biopsied,
        transform=transforms(train=True),
        skip_preprocessing=True,
        downscale=args.downscale,
        max_slice_offset=args.slice_offset,
        seed=running_seed,
        dataframe_subset = df_train
    )
    print("Number of volumes in train dataset:", len(train))

    valid = Dataset(
        csv_views=args.data_views,
        csv_bboxes=args.data_boxes,
        root_dir=args.images,
        subset="validation",
        random=False,
        only_biopsied=args.only_biopsied,
        transform=transforms(train=False),
        skip_preprocessing=True,
        downscale=args.downscale,
        max_slice_offset=args.slice_offset,
        seed=running_seed,
        dataframe_subset=df_valid
    )
    print("Number of volumes in validation dataset:", len(valid))

    test = Dataset(
        csv_views=args.data_views,
        csv_bboxes=args.data_boxes,
        root_dir=args.images,
        subset="test",
        random=False,
        only_biopsied=args.only_biopsied,
        transform=transforms(train=False),
        skip_preprocessing=True,
        downscale=args.downscale,
        max_slice_offset=args.slice_offset,
        seed=running_seed,
        dataframe_subset=df_test
    )
    print("Number of volumes in test dataset:", len(test))

    return train, valid, test


def splitting_dataset(args, running_seed):
    df_train, df_valid, df_test, df_all = data_frame_subset(
        csv_views=args.data_views, csv_boxes=args.data_boxes,
        seed=running_seed
    )

    return df_train, df_valid, df_test, df_all



def froc(df, targets_nb):
    total_slices = len(df.drop_duplicates(subset=["PID"]))
    total_tps = targets_nb
    tpr = [0.0]
    fps = [0.0]
    saved_thresholds = [0.0]
    max_fps = 4.0
    thresholds = sorted(df[df["TP"] == 1]["Score"], reverse=True)

    for th in thresholds:
        df_th = df[df["Score"] >= th]
        df_th_unique_tp = df_th.drop_duplicates(subset=["PID", "TP", "GTID"])
        num_tps_th = float(sum(df_th_unique_tp["TP"]))
        tpr_th = num_tps_th / total_tps
        num_fps_th = float(len(df_th[df_th["TP"] == 0]))
        fps_th = num_fps_th / total_slices
        # before appending tpr and FPs values check:
        # if number of FPs bigger than 4 then
        # limit the number of FPs to 4 and append the corresponding tpr to the last value of tpr list
        # and stop the loop. We have our values:
        if fps_th > max_fps:
            tpr.append(tpr[-1])
            fps.append(max_fps)
            # keep threshold value:
            saved_thresholds.append(saved_thresholds[-1])
            break
        tpr.append(tpr_th)      # append tpr value
        fps.append(fps_th)      # append FPs value
        # keep threshold value:
        saved_thresholds.append(th)

    # if we don't have tpr for FPs=4, then set for fps=4 the last value of tpr list:
    if np.max(fps) < max_fps:
        tpr.append(tpr[-1])
        fps.append(max_fps)
        saved_thresholds.append(saved_thresholds[-1])
    return tpr, fps, saved_thresholds

def plot_loss_train(train_losses, color_train="darkorange", linestyle="-"):

    y_limit = 15

    # compute lengths of lists:
    train_vals_len = len(train_losses)

    x_limit = train_vals_len

    fig = plt.figure(figsize=(10, 8))
    canvas = FigureCanvasAgg(fig)

    train_x1 = np.arange(1, train_vals_len + 1, 1)
    train_y1 = train_losses

    plt.plot(train_x1, train_y1, label="training", color=color_train, linestyle=linestyle, lw=2)

    plt.xlim([0.0, x_limit])
    plt.ylim([0.0, y_limit])

    plt.xlabel("Epochs", fontsize=24)
    plt.ylabel("Training Loss", fontsize=24)
    plt.grid(color="silver", alpha=0.3, linestyle="--", linewidth=1)
    plt.tight_layout()
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.fromstring(s, np.uint8).reshape((height, width, 4))

def plot_loss_valid(train_losses, valid_losses, val_interv, color_valid, linestyle="-"):

    y_limit = 15

    # compute lengths of lists:
    train_vals_len = len(train_losses)
    valid_vals_len = len(valid_losses)

    x_limit = train_vals_len

    fig = plt.figure(figsize=(10, 8))
    canvas = FigureCanvasAgg(fig)

    if train_vals_len != valid_vals_len:
        valid_x1 = np.arange(val_interv, train_vals_len + val_interv, int(train_vals_len / valid_vals_len))
    else:
        valid_x1 = np.arange(1, valid_vals_len + 1, 1)

    valid_y1 = valid_losses

    plt.plot(valid_x1, valid_y1, label="validation", color=color_valid, linestyle=linestyle, lw=2)

    # plt.ylim([0.0, y_thresh])
    plt.xlim([0.0, x_limit])
    plt.ylim([0.0, y_limit])

    plt.xlabel("Epochs", fontsize=24)
    plt.ylabel("Validation Loss", fontsize=24)
    plt.grid(color="silver", alpha=0.3, linestyle="--", linewidth=1)
    plt.tight_layout()
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.fromstring(s, np.uint8).reshape((height, width, 4))

def plot_losses(train_losses, valid_losses, val_interv, color_train, color_valid, linestyle="-"):

    y_limit = 15

    # compute lengths of lists:
    train_vals_len = len(train_losses)
    valid_vals_len = len(valid_losses)

    x_limit = max(train_vals_len, valid_vals_len)

    fig = plt.figure(figsize=(10, 8))
    canvas = FigureCanvasAgg(fig)

    train_x1 = np.arange(1, train_vals_len+1, 1)
    valid_x1 = np.arange(val_interv, train_vals_len + val_interv, int(train_vals_len / valid_vals_len))

    train_y1 = train_losses
    valid_y1 = valid_losses

    plt.plot(train_x1, train_y1,label="training", color=color_train, linestyle=linestyle, lw=2)
    plt.plot(valid_x1, valid_y1,label="validation", color=color_valid, linestyle=linestyle, lw=2)

    # plt.ylim([0.0, y_thresh])
    plt.xlim([0.0, x_limit])
    plt.ylim([0.0, y_limit])

    plt.xlabel("Epochs", fontsize=24)
    plt.ylabel("Loss", fontsize=24)
    plt.legend()
    plt.grid(color="silver", alpha=0.3, linestyle="--", linewidth=1)
    plt.tight_layout()
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.fromstring(s, np.uint8).reshape((height, width, 4))


def plot_tpr2(vals, color="darkorange", linestyle="-"):

    values_len = len(vals)
    x_limit = values_len

    fig = plt.figure(figsize=(10, 8))
    canvas = FigureCanvasAgg(fig)

    values_x1 = np.arange(1, values_len + 1, 1)
    values_y1 = vals

    plt.plot(values_x1, values_y1, label="training", color=color, linestyle=linestyle, lw=2)

    plt.xlim([0.0, x_limit])
    plt.ylim([0.0, 1.0])

    plt.xlabel("Epochs", fontsize=24)
    plt.ylabel("Sensitivity at 2 FPs per slice", fontsize=24)
    plt.grid(color="silver", alpha=0.3, linestyle="--", linewidth=1)
    plt.tight_layout()
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.fromstring(s, np.uint8).reshape((height, width, 4))

def plot_tpr2_together(vals_train, vals_valid, val_interv, color_train, color_valid, linestyle="-"):

    # compute lengths of lists:
    train_vals_len = len(vals_train)
    valid_vals_len = len(vals_valid)

    x_limit = max(train_vals_len, valid_vals_len)

    fig = plt.figure(figsize=(10, 8))
    canvas = FigureCanvasAgg(fig)

    values_train_x1 = np.arange(1, train_vals_len + 1, 1)
    values_train_y1 = vals_train

    values_valid_x1 = np.arange(1, valid_vals_len + 1, 1)
    values_valid_y1 = vals_valid

    plt.plot(values_train_x1, values_train_y1,label="training", color=color_train, linestyle=linestyle, lw=2)
    plt.plot(values_valid_x1, values_valid_y1,label="validation", color=color_valid, linestyle=linestyle, lw=2)

    plt.xlim([0.0, x_limit])
    plt.ylim([0.0, 1])

    plt.xlabel("Epochs", fontsize=24)
    plt.ylabel("Sensitivities at 2 FPs per slice", fontsize=24)
    plt.legend()
    plt.grid(color="silver", alpha=0.3, linestyle="--", linewidth=1)
    plt.tight_layout()
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.fromstring(s, np.uint8).reshape((height, width, 4))

def plot_loss_learning_rate(values, color_val="darkorange", linestyle="-"):

    y_limit = max(values)

    # compute lengths of lists:
    x_vals_len = len(values)

    x_limit = x_vals_len

    fig = plt.figure(figsize=(10, 8))
    canvas = FigureCanvasAgg(fig)

    x_vals = np.arange(1, x_vals_len + 1, 1)
    y_vals = values

    plt.plot(x_vals, y_vals, color=color_val, linestyle=linestyle, lw=2)

    plt.xlim([0.0, x_limit])
    plt.ylim([0.0, y_limit])

    plt.xlabel("Epochs", fontsize=24)
    plt.ylabel("Learning rate", fontsize=24)
    plt.grid(color="silver", alpha=0.3, linestyle="--", linewidth=1)
    plt.tight_layout()
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.fromstring(s, np.uint8).reshape((height, width, 4))


def plot_hist(values, color_hist, y_limit, subset, epoch, linestyle="-"):

    fig = plt.figure(figsize=(10, 8))
    canvas = FigureCanvasAgg(fig)

    plt.hist(values, bins=100,color=color_hist)

    plt.ylabel('frequency', fontsize=24)
    plt.xlabel('confidence scores', fontsize=24)

    if subset in ["training", "validation"]:
        plt.title("Confidence Scores for " + str(subset) + " predicted boxes without thresholding(epoch: " + str(
            epoch) + "):")
    else:
        plt.title("Confidence Scores for " + str(subset) + " predicted boxes without thresholding:")

    plt.ylim([0, y_limit])

    plt.grid(color="silver", alpha=0.3, linestyle="--", linewidth=1)
    plt.tight_layout()
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.fromstring(s, np.uint8).reshape((height, width, 4))

def plot_froc(fps, tpr, subset, color="darkorange", linestyle="-"):
    fig = plt.figure(figsize=(10, 8))
    canvas = FigureCanvasAgg(fig)
    plt.plot(fps, tpr, color=color, linestyle=linestyle, lw=2)
    plt.xlim([0.0, 4.0])
    plt.xticks(np.arange(0.0, 4.5, 0.5))
    plt.ylim([0.0, 1.0])
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.tick_params(axis="both", which="major", labelsize=16)

    plt.title("FROC curve for " + str(subset) + " set:")

    plt.xlabel("Mean FPs per slice", fontsize=24)
    plt.ylabel("Sensitivity", fontsize=24)
    plt.grid(color="silver", alpha=0.3, linestyle="--", linewidth=1)
    plt.tight_layout()
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.fromstring(s, np.uint8).reshape((height, width, 4))


def is_tp(pred_box, true_box, min_dist=50):
    # box: center point + dimensions
    pred_y, pred_x = pred_box["Y"], pred_box["X"]
    gt_y, gt_x = true_box["Y"], true_box["X"]
    # distance between GT and predicted center points
    dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
    # TP radius based on GT box size
    dist_threshold = np.sqrt(true_box["Width"] ** 2 + true_box["Height"] ** 2) / 2.
    dist_threshold = max(dist_threshold, min_dist)
    # TP if predicted center within GT radius
    return dist <= dist_threshold

def is_tp_using_IoU(pred_box, true_box, iou_threshold):
    # input "X" and "Y" coordinates are the coordinates of the center point of the corresponding box.
    # We want to extract the coordinates of the top lef and bottom right points.

    # pred_y1, pred_x1: top left point of predicted box:
    # pred_y2, pred_x2: bottom right point of predicted box:
    pred_y1, pred_x1 = pred_box["Y"] - pred_box["Height"] // 2, pred_box["X"] - pred_box["Width"] // 2
    pred_y2, pred_x2 = pred_box["Y"] + pred_box["Height"] // 2, pred_box["X"] + pred_box["Width"] // 2

    # gt_y1, gt_x1: top left point of GT box:
    # gt_y2, gt_x2: bottom right point of GT box:
    gt_y1, gt_x1 = true_box["Y"] - true_box["Height"] // 2, true_box["X"] - true_box["Width"] // 2
    gt_y2, gt_x2 = true_box["Y"] + true_box["Height"] // 2, true_box["X"] + true_box["Width"] // 2

    # y1, x1: top left point of intersection box:
    # y2, x2: bottom right point of intersection box:
    y1, x1 = max(gt_y1, pred_y1), max(gt_x1, pred_x1)
    y2, x2 = min(gt_y2, pred_y2), min(gt_x2, pred_x2)

    # w: weight of intersection box:
    # h: height of intersection box:
    w = x2 - x1
    h = y2 - y1

    # intersection area:
    inter_area = w * h

    # gt_w: width of GT box:
    # gt_h: height of GT box:
    gt_w = true_box["Width"]
    gt_h = true_box["Height"]

    # pred_w: width of predicted box:
    # pred_h: height of predicted box:
    pred_w = pred_box["Width"]
    pred_h = pred_box["Height"]

    # union area:
    union_area = gt_w * gt_h + pred_w * pred_h - inter_area

    iou = inter_area / union_area

    return iou >= iou_threshold

def evaluate_batch(y_pred, y_true, froc_th=None):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    df_eval = pd.DataFrame()

    # froc_th: None for train/validation/test sets
    if froc_th is None:
        pass

    # iterate through batch (batch size = 16 times):
    for i in range(y_pred.shape[0]):
        # df_gt_boxes, scores_label_gt = pred2boxes(y_true[i], threshold=1.0)
        df_gt_boxes = pred2boxes(y_true[i], threshold=1.0)
        df_gt_boxes["GTID"] = np.random.randint(10e10) * (1 + df_gt_boxes["X"])

        df_pred_boxes = pred2boxes(y_pred[i])
        df_pred_boxes["PID"] = np.random.randint(10e12)

        df_pred_boxes["TP"] = 0

        if df_gt_boxes.shape[0] > 0:
            df_pred_boxes["GTID"] = np.random.choice(
                list(set(df_gt_boxes["GTID"])), df_pred_boxes.shape[0]
            )
        for index, pred_box in df_pred_boxes.iterrows():
            tp_list = [
                (j, is_tp(pred_box, x_box)) for j, x_box in df_gt_boxes.iterrows()
            ]
            if any([tp[1] for tp in tp_list]):
                tp_index = [tp[0] for tp in tp_list if tp[1]][0]
                df_pred_boxes.at[index, "TP"] = 1
                df_pred_boxes.at[index, "GTID"] = df_gt_boxes.at[tp_index, "GTID"]

        df_eval = df_eval.append(df_pred_boxes, ignore_index=True, sort=False)

    # return df_eval, scores_batch_pred
    return df_eval

def pred2boxes(pred, threshold=None, froc_th=None):
    # box: center point + dimensions
    anchor = Dataset.anchor
    cell_size = Dataset.cell_size
    np.nan_to_num(pred, copy=False)
    obj_th = pred[0]

    # # so keep all the predicted scores for analytical purposes to a list:
    # scores_label = list(obj_th[obj_th > 0])

    # * threshold:1.0, froc_th:None -> means we evaluate GT
    # * threshold:None, froc_th:None -> means we evaluate predictions on train/validation/test sets
    if threshold is None:
        if froc_th is None:
            threshold = 0
        else:
            threshold = froc_th
    obj_th[obj_th < threshold] = 0
    yy, xx = np.nonzero(obj_th)
    scores = []
    xs = []
    ys = []
    ws = []
    hs = []
    for i in range(len(yy)):
        scores.append(pred[0, yy[i], xx[i]])
        h = int(anchor[0] * pred[3, yy[i], xx[i]] ** 2)
        hs.append(h)
        w = int(anchor[1] * pred[4, yy[i], xx[i]] ** 2)
        ws.append(w)
        y_offset = pred[1, yy[i], xx[i]]
        y_mid = yy[i] * cell_size + (cell_size / 2) + (cell_size / 2) * y_offset
        ys.append(int(y_mid))
        x_offset = pred[2, yy[i], xx[i]]
        x_mid = xx[i] * cell_size + (cell_size / 2) + (cell_size / 2) * x_offset
        xs.append(int(x_mid))

    df_dict = {"Score": scores, "X": xs, "Y": ys, "Width": ws, "Height": hs}
    df_boxes = pd.DataFrame(df_dict)
    df_boxes.sort_values(by="Score", ascending=False, inplace=True)

    return df_boxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyper-parameters grid search for YOLO model for cancer detection in Duke DBT volumes"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20000,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=500,
        help="early stopping: number of epochs to wait for improvement (default: 25)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="initial learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--loc-weight",
        type=float,
        default=0.5,
        help="weight of localization loss (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--data-views",
        type=str,
        default="/mnt/seagate/DBT/manifest-1617905855234/BCS-DBT labels-new-v0.csv",
        help="csv file listing training views together with category label",
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
        "--seed",
        type=int,
        default=42,
        help="random seed for validation split (default: 42)",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=2,
        help="input image downscale factor (default 2)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=8,
        help="experiment name for new mlflow (default: 0)",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="experiment id to restore in-progress mlflow experiment (default: None)",
    )
    parser.add_argument(
        "--slice-offset",
        type=int,
        default=0,
        help="maximum offset from central slice to consider as GT bounding box (default: 0)",
    )
    parser.add_argument(
        "--only-biopsied",
        default=True,  # set to true by default for convenience
        action="store_true",
        help="flag to use only biopsied cases",
    )
    args = parser.parse_args()
    main(args)