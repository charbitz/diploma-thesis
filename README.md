# Mass detection in digital breast tomosynthesis

This is the code repository of my diploma thesis. The was not the original repository. In this specific repository the final files were uploaded for demonstration purposes.

## Image Databases:

* Digital Breast Tomosynthesis dataset published by Duke University, Durham, NC. You can find the dataset [here](https://www.cancerimagingarchive.net/collection/breast-cancer-screening-dbt/). This dataset consists of digital breast tomosynthesis images. Only 200 Biopsied case views and 234 Normal case views from this database were used.

* The famous INBreast dataset which can be found [here](https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset). This dataset consists of digital mammography images. Only 107 biopsied masses cases from this database were used. The groundtruth boxes were in Regions Of Interest information so the conversion to bounding boxes was implemented with the help of this [repository](github.com/charbitz/breast_mass_detection). Also this repository was used for the conversion of the images from the DICOM format to PNG.

## Deep Learning Implementations:

### * Apply a YOLO network in DenseNet architecture on DBT data.

Parameters of dataset splitting:
- training: 50%, validation: 20%, test: 10%
- splitting based on patients, in order not to have two different views of the same patient's mass in two different sets

Parameters of training:
-here

Preprocessing in order to prepare the dataset:
- image downscaling
- image segmentation
- getting largest connected component

Data augmentations while training:
- random scaling in [0.9 1.3]
- random area cropping with a way that contains though the ground truth bounding box, final fixed image dimensions: (1056, 672)
- random horrizontal flipping with a probability of 50%


### * Apply a YOLOv5 network on DBT and INBreast data.

The YOLOv5 network is implemented by [ultralytics](https://www.ultralytics.com/). The documentation of YOLOv5 and the under development code can be found [here](https://github.com/ultralytics/yolov5). 

Here ...

### * Apply preprocessing on both image databases in order to get better model performance:

- gaussian filtering
- image segmentation using OTSU thresholding
- getting largest connected component 
- truncation normalization
- Contrast Limited Adaptive Histogram Equalization
- image synthesizing into RGB format

### * Apply transfer learning techniques on YOLOv5 model:

Pretraining YOLOv5 on INBreast data and then use YOLOv5 with the best weights on DBT data 

Using all INBreast and part of DBT data on training set and then using the rest of DBT data on validation and test sets