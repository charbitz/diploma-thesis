# Mass detection in Digital Breast Tomosynthesis Images

This is the code repository of my Diploma Thesis. The was not the original repository. In this specific repository the final files were uploaded for demonstration purposes. In this README file you can see a summary about the **image databases**, **model architectures** and **learning techniques** that were selected for the final implemetation of the problem of _mass detection_. 

## Image Databases:

* Digital Breast Tomosynthesis dataset published by Duke University, Durham, NC. You can find the dataset [here](https://www.cancerimagingarchive.net/collection/breast-cancer-screening-dbt/). This dataset consists of digital breast tomosynthesis images. Only 200 Biopsied case views and 234 Normal case views from this database were used. Biopsied cases have masses (benign or cancer) which were confirmed by radiologists by biopsy as dangerous. Normal cases have masses which do not correspond to dangerous findings, also confirmed by radiologists.  

* The famous INBreast dataset which can be found [here](https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset). The INBreast database was used because of lack of DBT data. This dataset consists of digital mammography images. Only 107 biopsied masses cases from this database were used. The groundtruth boxes were in Regions Of Interest information so the conversion to bounding boxes was implemented with the help of this [repository](https://github.com/charbitz/breast_mass_detection), which i forked and then commit changes for my implemetation. Also this repository was used for the conversion of the images from the DICOM format to PNG.

## Deep Learning Implementations:

### * Apply a YOLO network in DenseNet architecture on DBT data.

Parameters of dataset splitting into subsets:
- training: 50%, validation: 20%, test: 10%
- splitting based on patients, in order not to have two different views of the same patient's mass in two different sets

Hyper-parameter tuning:
- epochs
- patience
- initial learning rate
- validation interval
- schedule patience
- factor of learning rate

Preprocessing in order to prepare the dataset:
- image downscaling
- image segmentation
- getting largest connected component

Data augmentations while training the YOLO model in DenseNet architecture:
- random scaling in [0.9 1.3]
- random area cropping with a way that contains though the ground truth bounding box, final fixed image dimensions: (1056, 672)
- random horrizontal flipping with a probability of 50%

### Apply a YOLOv5 network on DBT and INBreast data.

The YOLOv5 network is implemented by [ultralytics](https://www.ultralytics.com/). The documentation of YOLOv5 and the under development code can be found [here](https://github.com/ultralytics/yolov5). 

Apply preprocessing on both image databases in order to get better model performance:

- gaussian filtering
- image segmentation using OTSU thresholding
- getting largest connected component 
- truncation normalization
- Contrast Limited Adaptive Histogram Equalization
- image synthesizing into RGB format

Data augmentations while training the YOLOv5 model:
- random image translation of 20%
- random horrizontal flipping with a probability of 50%
- random vertical flipping with a probability of 50%

### Apply transfer learning techniques on YOLOv5 model:

Technique 1: 
Pretraining YOLOv5 on INBreast data and then use YOLOv5 with the best weights on DBT data.

Technique 2:
Using all INBreast and part of DBT data on training set and then using the rest of DBT data on validation and test sets.

## File Description:

Files for training YOLO network in DenseNet architecture on DBT data:
- [dense_yolo.py](https://github.com/charbitz/diploma-thesis/blob/main/dense_yolo.py)
- [dataset.py](https://github.com/charbitz/diploma-thesis/blob/main/dataset.py)
- [loss.py](https://github.com/charbitz/diploma-thesis/blob/main/loss.py)
- [preprocess.py](https://github.com/charbitz/diploma-thesis/blob/main/preprocess.py)
- [transform.py](https://github.com/charbitz/diploma-thesis/blob/main/transform.py)
- [sampler.py ](https://github.com/charbitz/diploma-thesis/blob/main/sampler.py)
- [subsets_split.py](https://github.com/charbitz/diploma-thesis/blob/main/subsets_split.py)
- [train_best_model.py](https://github.com/charbitz/diploma-thesis/blob/main/train_best_model.py)

### Files for creating DBT and INBreast datasets in YOLOv5 annotation format:

For cases with only biopsied masses:
- [reorganize_dataset_subsets_masses.py](https://github.com/charbitz/diploma-thesis/blob/main/reorganize_dataset_subsets_masses.py)
- [dbt_dataset_ONLY-BIOPSIED_masses_NO-CLASSES.yaml](https://github.com/charbitz/diploma-thesis/blob/main/dbt_dataset_ONLY-BIOPSIED_masses_NO-CLASSES.yaml)

For cases with both biopsied and normal masses:
- [reorganize_dataset_subsets_masses_multiple_slices_normal_10_perc.py](https://github.com/charbitz/diploma-thesis/blob/main/reorganize_dataset_subsets_masses_multiple_slices_normal_10_perc.py)
- [dbt_dataset_multiple_slices_masses_NO-CLASSES_WHOLE-NORMAL-10.yaml](https://github.com/charbitz/diploma-thesis/blob/main/dbt_dataset_multiple_slices_masses_NO-CLASSES_WHOLE-NORMAL-10.yaml)


## Attribution:

So finally this project includes code from the following repositories:
- [duke-dbt-data](https://github.com/mazurowski-lab/duke-dbt-data) by [mateuszbuda](https://github.com/mateuszbuda)
- [duke-dbt-detection](https://github.com/mateuszbuda/duke-dbt-detection) by [mateuszbuda](https://github.com/mateuszbuda)
- [yolov5](https://github.com/ultralytics/yolov5) by [ultralytics](https://github.com/ultralytics)
- [breast_mass_detection](https://github.com/jordanvaneetveldt/breast_mass_detection) by [jordanvaneetveldt](https://github.com/jordanvaneetveldt)

<!-- try table -->

| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Row 1    | Data 1   | Data 2   |
| Row 2    | Data 3   | Data 4   |
