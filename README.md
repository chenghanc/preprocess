# Project VTS

Download specific class images and annotations from COCO dataset and convert to YOLO format. Perform training/fine-tuning on MS COCO dataset using Darknet


## Download specific classes from COCO:

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
- Download testing dataset for validation purpose (mAP calculations on the MS COCO evaluation server)
    * `wget http://images.cocodataset.org/zips/test2017.zip`
    * `wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/scripts/testdev2017.txt`
    * To evaluate accuracy of Yolov4, run validation `./darknet detector valid coco.data yolov4.cfg yolov4.weights`
    * To evaluate accuracy of custom model, run validation `./darknet detector valid coco.data yolov4-ft.cfg yolov4-ft.weights`
    * Submit file to the MS COCO evaluation server for the test-dev2019 (bbox) https://competitions.codalab.org/competitions/20794#participate
    * Please visit https://github.com/AlexeyAB/darknet/wiki/How-to-evaluate-accuracy-and-speed-of-YOLOv4 for more information
    
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
    * We are going to reuse COCO classes (nc=80)
    * We are transfer learning in the sense of using pretrained weights from COCO as a starting point (yolov4.weights)
    * **Use the original learning rate (= 0.000013) of last few thousand iterations seems to give better overall performance**

```
$ ./darknet detector train coco.data yolov4-ft.cfg yolov4.weights -clear -map -dont_show -mjpeg_port 8090 |tee -a trainRecord.txt
```

<details><summary><b>CLICK ME</b> - Official Yolov4 results on COCO dataset</summary>

- By running `./darknet detector valid coco.data yolov4.cfg yolov4.weights` and submit file to the MS COCO evaluation server as described earlier, we will get results (AP=0.435 and AP50=0.657) in the end of file View scoring output log

```
overall performance
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.435
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.657
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.473
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.267
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.467
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.580
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.403
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.617
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.713
Done (t=568.18s)
```

</details>

<details><summary><b>CLICK ME</b> - Results on COCO + custom dataset (truck, handbag and backpack)</summary>

- **with stopbackward** [yolov4-ft.cfg](https://github.com/chenghanc/preprocess/blob/main/yolov4-ft.cfg)
    * `Batch size: 64 (batch=64)`
    * `Total training data: 120,000`
    * `Iterations: 10,000 (max_batches = 10,000)`
    * `1 epoch = 120000 / 64 = 1875 iterations`
    * `10000 * 64 / 120000 = 5.3333 epochs`
    * **learning rate = 0.0001**
    * AP=0.407 and AP50=0.626
```
SCORE=0.686
overall performance
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.407
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.626
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.441
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.255
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.440
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.491
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.329
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.388
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.601
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.686
Done (t=540.49s)
```

- **without stopbackward** [yolov4-ft-wosb.cfg](https://github.com/chenghanc/preprocess/blob/main/yolov4-ft-wosb.cfg)
    * `Iterations: 10,000 (max_batches = 10,000)`
    * `10000 * 64 / 120000 = 5.3333 epochs`
    * **learning rate = 0.00013**
    * AP=0.407 and AP50=0.626
```
SCORE=0.685
overall performance
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.407
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.626
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.441
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.253
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.438
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.494
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.329
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.559
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.387
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.599
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.685
Done (t=366.62s)
```

- **without stopbackward** [yolov4-ft2.cfg](https://github.com/chenghanc/preprocess/blob/main/yolov4-ft2.cfg) [yolov4-ft-wosb2.cfg](https://github.com/chenghanc/preprocess/blob/main/yolov4-ft-wosb2.cfg)
    * `Iterations: 10,000 (max_batches = 10,000)`
    * `10000 * 64 / 120000 = 5.3333 epochs`
    * **learning rate = 0.000013**
    * **AP=0.418 and AP50=0.639**
```
yolov4-ft2.cfg
SCORE=0.696
overall performance (iterations 7000)
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.418
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.639
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.455
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.261
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.450
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.503
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.334
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.539
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.570
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.395
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.609
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.696
Done (t=382.29s)
```

```
yolov4-ft-wosb2.cfg
SCORE=
overall performance (iterations 10000)
```

- References
    * https://github.com/AlexeyAB/darknet/issues/2147
    * https://github.com/AlexeyAB/darknet/issues/6652
    * https://github.com/AlexeyAB/darknet/issues/5529
    * https://github.com/AlexeyAB/darknet/issues/5934
    * https://github.com/AlexeyAB/darknet/issues/6623
    * https://github.com/AlexeyAB/darknet/issues/4386
    * https://github.com/pjreddie/darknet/issues/224
    * https://yanwei-liu.medium.com/training-coco-object-detection-with-yolo-v4-f11bece3feb6

</details>

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

