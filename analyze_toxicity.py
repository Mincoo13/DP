import numpy as np
import pandas as pd
import seaborn as sns
from numpy import genfromtxt
from PIL import Image, ImageDraw

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

from keras.utils import to_categorical
from keras.models import model_from_json
from keras.models import Model

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from mrcnn import visualize

from mrcnn.config import Config
from skimage import io
import sys
import matplotlib
import argparse
import datetime
import os
import shutil
import skimage.draw
from glob import glob
import config as cfg
from os import listdir
from sklearn import metrics
import json
from utils import annotation_to_mask, generate_segment_graph, load_model
from utils import mask_to_rgb, color_splash, write_result

os.chdir(f'{os.path.realpath(__file__)}\\..\\')

class TrainConfig(Config):
    NAME = "68_20_resnet50_all_layers"

    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 1

    STEPS_PER_EPOCH = 68

    BACKBONE = 'resnet50'

    WEIGHT_DECAY = 0.01

    DETECTION_MIN_CONFIDENCE = 0.9

class InferenceConfig(TrainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def categorize_class(predict):
    if predict == 0:
        predict = "healthy"
    elif predict == 1:
        predict = "healthy_sick"
    elif predict == 2:
        predict = "sick"
    return predict

def load_image(im_path):
    image = io.imread(im_path, plugin='matplotlib')
    image = cv2.resize(image, (1100, 1100),
                       interpolation=cv2.INTER_NEAREST)
    if len(image.shape) > 2 and image.shape[2] > 3:
        image = image[:, :, :3]
    elif len(image.shape) == 2:
        image = np.stack((image, image, image), axis=2)

    return image

def load_data(folder):

    folder = f'dataset\\analyze_toxicity\\{folder}'

    seg_data = []
    clf_data = []
    ids = []

    for filename in os.listdir(folder):
        print(filename)
        image_seg = load_image(f'{folder}\\{filename}')

        image_clf = cv2.imread(f'{folder}\\{filename}',
                           cv2.IMREAD_UNCHANGED)
        image_clf = cv2.cvtColor(image_clf, cv2.COLOR_BGR2RGB)
        image_clf = cv2.resize(image_clf, (128, 128),
                               interpolation=cv2.INTER_NEAREST)
        
        seg_data.append(image_seg)
        clf_data.append(np.array(image_clf))
        ids.append(filename.split('.')[0])
    
    return clf_data, seg_data, ids

def get_model(intent, path, cfg=None):
    if intent is 'seg':
        model = modellib.MaskRCNN(mode="inference", config=cfg,
                                  model_dir=path)
        model.load_weights(path, by_name=True)
    else:
        pass
    return model


def segment(data, model, after=False):
    df = pd.DataFrame(columns=['leaf_count', 'area'])
    for image in data:
        results = model.detect([image], verbose=True)
        masks = results[0]['masks']
        predicted_area = np.count_nonzero(masks)
        df = df.append({'leaf_count': masks.shape[2], 'area': predicted_area},
                        ignore_index=True)
    return df

def group(data, toxicity):
    df = pd.DataFrame(columns='')
    return df


before_clf, before_seg, before_ids = load_data('before')
after_clf, after_seg, after_ids = load_data('after')

clf_args = {'pretrained_model_path': 'models/health_classification/shuffle_7_3_soft_cat_adam_40_plateau_early'}
clf_model, clf_model_name, clf_model_history = load_model(clf_args)
x_data = np.array(after_clf)
x_data = x_data.astype('float32')/255

predicted_health = clf_model.predict_classes(x_data)
health_predicted = pd.DataFrame(columns=['id', 'health'])
health_predicted['id'] = before_ids
health_predicted['health'] = predicted_health

sorted_health_predicted = pd.DataFrame(columns=['id', 'health'])
for tox in ['M', '5', '10', '20', '40', '80', '160']:
    for index, row in health_predicted.iterrows():
        
        if str(tox) in str(row['id']):
            print(row.to_dict())
            sorted_health_predicted = sorted_health_predicted.append(row.to_dict(), ignore_index=True)
print(sorted_health_predicted)
sorted_health_predicted.to_csv('data/health_predicted.csv', index=False)


path = '''models/segmentation/model_segment_overlap_68_20_resnet50_head_layers_reg0_01/mask_rcnn_68_20_resnet50_head_layers_reg0_01_0020.h5'''
cfg = InferenceConfig()
seg_model = get_model('seg', path, cfg=cfg)

before_seg_predicted = segment(before_seg, seg_model)
after_seg_predicted = segment(after_seg, seg_model, after=True)
seg_differences = after_seg_predicted - before_seg_predicted
seg_differences['ids'] = before_ids

seg_predicted = pd.DataFrame(columns=['id', 'leaf_count_before', 'leaf_count_after', 'area_before', 'area_after', 'leaf_difference', 'area_difference'])
seg_predicted['id'] = before_ids
seg_predicted['leaf_count_before'] = before_seg_predicted['leaf_count']
seg_predicted['leaf_count_after'] = after_seg_predicted['leaf_count']
seg_predicted['area_before'] = before_seg_predicted['area']
seg_predicted['area_after'] = after_seg_predicted['area']
seg_predicted['leaf_difference'] = after_seg_predicted['leaf_count'] - before_seg_predicted['leaf_count']
seg_predicted['area_difference'] = after_seg_predicted['area'] - before_seg_predicted['area']

sorted_seg_predicted = pd.DataFrame(columns=['id', 'leaf_count_before', 'leaf_count_after', 'area_before', 'area_after', 'leaf_difference', 'area_difference'])
for tox in ['M', '5', '10', '20', '40', '80', '160']:
    for index, row in seg_predicted.iterrows():
        
        if str(tox) in str(row['id']):
            print(row.to_dict())
            sorted_seg_predicted = sorted_seg_predicted.append(row.to_dict(), ignore_index=True)
print(sorted_seg_predicted)
sorted_seg_predicted.to_csv('data/segmentation_predicted.csv', index=False)


value1=np.random.uniform(size=24)
value2=value1+np.random.uniform(size=24)/4
df_leaf = pd.DataFrame({'group':sorted_seg_predicted['id'], 'value1':sorted_seg_predicted['leaf_count_before'] , 'value2':sorted_seg_predicted['leaf_count_after'] , 'diff': sorted_seg_predicted['leaf_difference']})
df_area = pd.DataFrame({'group':sorted_seg_predicted['id'], 'value1':sorted_seg_predicted['area_before'], 'value2':sorted_seg_predicted['area_after'] , 'diff': sorted_seg_predicted['area_difference']})

real_df = pd.read_csv('data/analyze_toxicity.csv', sep=';')


my_range = list(range(1, 3 * len(df_leaf.index) + 1, 3))
my_range_real=[x + 1 for x in my_range]
 
plt.figure(figsize=[10, 6])
plt.vlines(x=my_range, ymin=df_leaf['value1'], ymax=df_leaf['value2'], color='grey', alpha=0.4)
plt.vlines(x=my_range_real, ymin=real_df['leaf_before'], ymax=real_df['leaf_after'], color='grey', alpha=0.4)
plt.scatter(my_range, df_leaf['value1'], color='blue', marker='+', alpha=0.6, label='predikovaný začiatok experimentu')
plt.scatter(my_range, df_leaf['value2'], color='red', marker='+', alpha=0.6, label='predikovaný koniec experimentu')
plt.scatter(my_range_real, real_df['leaf_before'], color='green', marker='+', alpha=0.6, label='reálny začiatok experimentu')
plt.scatter(my_range_real, real_df['leaf_after'], color='orange', marker='+', alpha=0.6, label='reálny koniec experimentu')
plt.legend()
 
plt.xticks(my_range, df_leaf['group'], rotation=45)
plt.title("Rozdiel v počte lístkov")
plt.xlabel('Vzorky')
plt.ylabel('Počet lístkov')

plt.savefig(f'graphs/analyze_toxicity/leaf_count_compare.png')
plt.close()

plt.figure(figsize=[10, 6])
plt.vlines(x=my_range, ymin=df_area['value1'], ymax=df_area['value2'], color='grey', alpha=0.4)
plt.scatter(my_range, df_area['value1'], color='blue', marker='+', alpha=0.6, label='začiatok experimentu')
plt.scatter(my_range, df_area['value2'], color='red', marker='+', alpha=0.6 , label='koniec experimentu')
plt.legend()

plt.xticks(my_range, df_area['group'], rotation=45)
plt.title("Rozdiel v ploche")
plt.xlabel('Vzorky')
plt.ylabel('Plocha v pixeloch')

plt.savefig(f'graphs/analyze_toxicity/area.png')
plt.close()