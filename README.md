# Data preparation

Download specific class images and annotations from COCO dataset and convert to YOLO format. Perform training/fine-tuning on MS COCO dataset using Darknet


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
---

## Get MS COCO training dataset and perform training
- Use bash-script to get MS COCO training dataset
    * `./get_coco_dataset.sh`
    * Please visit https://github.com/AlexeyAB/darknet/wiki/Train-Detector-on-MS-COCO-(trainvalno5k-2014)-dataset for more information
- Download testing dataset for validation purpose (mAP calculations)
    * `wget http://images.cocodataset.org/zips/test2017.zip`
    * `wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/scripts/testdev2017.txt`
    * Please visit https://yanwei-liu.medium.com/training-coco-object-detection-with-yolo-v4-f11bece3feb6 for more information
- Modify `coco.data`
```
classes= 80
train  = /home/nechk/NECHK-Results/coco2021/coco/trainvalno5k.txt
valid  = /home/nechk/NECHK-Results/coco2021/coco/testdev2017.txt
names  = data/coco.names
backup = backup
eval=coco
```
- Training data structure
    * If we need to fine-tune the model on COCO dataset, add custom images in images/val2014 and labels in labels/val2014
```
Images:

images/
├── train2014
└── val2014

Annotations:

labels/
├── train2014
└── val2014
```
- Train and fine-tune the model
```
$ ./darknet detector train coco.data yolov4-ft.cfg yolov4.weights -clear -map -dont_show -mjpeg_port 8090 |tee -a trainRecord.txt
```

---

# Appendix

We can perform the downloading by running a **Jupyter notebook** from a remote server. Please visit https://ljvmiranda921.github.io/notebook/2018/01/31/running-a-jupyter-notebook for more information

## Usage:

- #### Step 1: Run Jupyter Notebook from remote machine

    * Log-in to your remote machine via `ssh` command. Type the following
    *

  ```ini
  $ jupyter notebook --no-browser --port=8888
  ```

- #### Step 2: Forward port XXXX to YYYY and listen to it

    * In your remote machine, the notebook is running at the port XXXX=8888 which you specified
    * Forward port XXXX=8888 to port YYYY=8889 of your local machine so that you can listen and run it from your browser
    *
   
  ```
  $ ssh -N -f -L localhost:8889:localhost:8888 nechk@192.168.1.117
  ```

- #### Step 3: Open Jupyter Notebook

    * To open the Jupyter notebook from your remote machine
    * Type http://localhost:8889/

