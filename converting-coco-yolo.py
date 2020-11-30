import pandas as pd
import os
import cv2

full_path_to_ts_dataset = '/home/nechk/NECHK-Results/helmet2/coco/downloaded_images'

p = ['bus']

ann = pd.read_csv(full_path_to_ts_dataset + '/' + 'annotations_download_bus.csv',
                  names=['ImageID',
                         'XMin',
                         'YMin',
                         'XMax',
                         'YMax',
                         'ClassID'],
                  sep=',')

print(ann.head())

ann['CategoryID'] = ''
ann['center x'] = ''
ann['center y'] = ''
ann['width'] = ''
ann['height'] = ''

ann.loc[ann['ClassID'].isin(p), 'CategoryID'] = 0

print(ann.head())

ann['center x'] = (ann['XMax'] + ann['XMin']) / 2
ann['center y'] = (ann['YMax'] + ann['YMin']) / 2

ann['width'] = ann['XMax'] - ann['XMin']
ann['height'] = ann['YMax'] - ann['YMin']


print(ann.shape)
print(ann.head(5))
ann['ImageID'] = ann['ImageID'].str.split('/').str[1]
print(ann.head(5))

r = ann.loc[:, ['ImageID',
                'CategoryID',
                'center x',
                'center y',
                'width',
                'height']].copy()

print(r.head())

print(os.getcwd())

os.chdir(full_path_to_ts_dataset)

print(os.getcwd())

for current_dir, dirs, files in os.walk('.'):
    # Going through all files
    for f in files:
        # Checking if filename ends with '.ppm'
        if f.endswith('.jpg'):
            # Reading image and getting its real width and height
            image_ppm = cv2.imread(f)

            # Slicing from tuple only first two elements
            h, w = image_ppm.shape[:2]

            # Slicing only name of the file without extension
            image_name = f[:-4]

            # Getting Pandas dataFrame that has only needed rows
            # By using 'loc' method we locate needed rows
            # that satisfies condition 'classes['ImageID'] == f'
            # that is 'find from the 1st column element that is equal to f'
            # By using copy() we create separate dataFrame
            # not just a reference to the previous one
            # and, in this way, initial dataFrame will not be changed
            sub_r = r.loc[r['ImageID'] == f].copy()

            # Normalizing calculated bounding boxes' coordinates
            # according to the real image width and height
            sub_r['center x'] = sub_r['center x'] / w
            sub_r['center y'] = sub_r['center y'] / h
            sub_r['width'] = sub_r['width'] / w
            sub_r['height'] = sub_r['height'] / h

            # Getting resulted Pandas dataFrame that has only needed columns
            # By using 'loc' method we locate here all rows
            # but only specified columns
            # By using copy() we create separate dataFrame
            # not just a reference to the previous one
            # and, in this way, initial dataFrame will not be changed
            resulted_frame = sub_r.loc[:, ['CategoryID',
                                           'center x',
                                           'center y',
                                           'width',
                                           'height']].copy()

            # Checking if there is no any annotations for current image
            if resulted_frame.isnull().values.all():
                # Skipping this image
                continue

            # Preparing path where to save txt file
            # Pay attention! If you're using Windows, it might need to change
            # this: + '/' +
            # to this: + '\' +
            # or to this: + '\\' +
            path_to_save = full_path_to_ts_dataset + '/' + image_name + '.txt'

            # Saving resulted Pandas dataFrame into txt file
            resulted_frame.to_csv(path_to_save, header=False, index=False, sep=' ')

            # Preparing path where to save jpg image
            # Pay attention! If you're using Windows, it might need to change
            # this: + '/' +
            # to this: + '\' +
            # or to this: + '\\' +
            path_to_save = full_path_to_ts_dataset + '/' + image_name + '.jpg'

            # Saving image in jpg format by OpenCV function
            # that uses extension to choose format to save with
            cv2.imwrite(path_to_save, image_ppm)
