# Data preparation

Download specific class images and annotations from COCO dataset and convert to YOLO format

## Usage:
- #### Prerequisite
    * Prepare COCO dataset. Please visit http://cocodataset.org for more information on COCO dataset
    * Install COCO API according to the instructions here: https://github.com/cocodataset/cocoapi
    * For Python, run `make` under `PythonAPI`
    
- #### Download specific class images and annotations from COCO dataset

    * We can download any specific class from COCO dataset, e.g. `toaster`, `bus` ...
    * Download specific class images and annotation csv file by running the code [pycoco.ipynb](https://github.com/chenghanc/preprocess/blob/main/pycoco.ipynb)

- #### Convert csv file to YOLO format by running the code

    * `python converting-coco-yolo.py`

