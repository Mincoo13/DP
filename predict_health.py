import sys
import os
import pandas as pd
import numpy as np
import pickle
import collections
import config as cfg
from train_health_model import set_model

import cv2
import matplotlib.pyplot as plt
from keras.utils import plot_model

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import models, layers
from keras import backend as K
from keras.models import model_from_json
from keras.models import Model

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import matplotlib.image as mpimg

args = cfg.PRED_HEALTH_ARGS

if args['current_folder']:
    os.chdir(f'{os.path.realpath(__file__)}\\..\\')

def categorize_class(predict):
    if predict == 0:
        predict = "healthy"
    elif predict == 1:
        predict = "healthy_sick"
    elif predict == 2:
        predict = "sick"
    return predict

def cm_analysis(y_true, y_pred, labels, path, ymap=None, figsize=(10, 10), title=""):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap='viridis')
    plt.title(title)
    plt.savefig(path)
    plt.close()

image_path = 'dataset/lemna_minor/'
df = pd.read_csv(args['csv_path'], sep=args['csv_sep'])
health_data = df['health'].to_numpy()
labels = collections.Counter(health_data).keys()
nums = collections.Counter(health_data).values()
# num = []
# for label in labels:
#     k = 0

# print(nums)
# plt.figure(figsize=(8, 8))
# plt.bar(labels, nums)
# plt.title('Zdravie')
# plt.xlabel('kategórie')
# plt.ylabel('počet')
# plt.show()

# FOR TESTING ONLY
# df = df.sample(frac=1)
count = 0
# ---------

img_dim = (128, 128)
x_data = []
y_data = []
ids = []

print('Nacitanie obrazkov...')
for index, row in df.iterrows():

    # FOR TESTING ONLY
    count += 1
    if count == 50:
        break
    # ----------------

    print(f'{args["imgs_path"]}\\{row["id"]}.jpg')
    image = cv2.imread(f'{args["imgs_path"]}\\{row["id"]}.jpg', cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, img_dim, interpolation=cv2.INTER_NEAREST)
    x_data.append(np.array(image_resized))
    y_data.append(row['health'])
    ids.append(row['id'])
print('Hotovo')

print('Spracovanie dat...')
x_data = np.array(x_data)
y_data = np.array(y_data)
ids = np.array(ids)

plt.figure(figsize=(6, 6))
plt.imshow(x_data[0])
plt.title(y_data[0])
plt.close()

x_data = x_data.astype('float32')/255

y_encoded = LabelEncoder().fit_transform(y_data)
y_categorical = to_categorical(y_encoded)

r = np.arange(x_data.shape[0])
np.random.shuffle(r)
X = x_data[r]
Y = y_categorical[r]
ids = ids[r]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5)
print('Hotovo')
################################################################################

input_shape = X_train.shape[1:]
if args['pretrained_model']:
    print('Nacitanie predtrenovaneho modelu...')
    model_name = os.path.basename(os.path.normpath(args['pretrained_model_path']))
    pretrained_path = f"{args['pretrained_model_path']}\\{model_name}"
    json_file = open(f'{pretrained_path}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(f'{pretrained_path}.h5')
    with open(f'{pretrained_path}.pickle', 'rb') as pickle_file:
        history = pickle.load(pickle_file)
    print('Hotovo')
else:
    print('Trenovanie...')
    model_name = os.path.basename(os.path.normpath(args['model_path']))
    model_path = f"{args['model_path']}\\{model_name}"

    model, history = set_model(input_shape, X_train, Y_train, X_val, Y_val, 12)

    if not os.path.exists(args['model_path']):
        os.makedirs(args['model_path'])

    model_json = model.to_json()
    with open(f'{model_path}.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(f'{model_path}.h5')
    with open(f'{model_path}.pickle', 'wb') as file_pi:
        pickle.dump(history, file_pi)
    print('Hotovo')

################################################################################
layer_names = ['conv2d_1', 'max_pooling2d_1', 'conv2d_2', 'conv2d_3',
               'max_pooling2d_2']
outputs = []
img_filter = X_test[0]
img_filter = img_filter.reshape([-1, 128, 128, 3])

for layer_name in layer_names:
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(img_filter)
    outputs.append(intermediate_output)

fig,ax = plt.subplots(nrows=5, ncols=5)

for i in range(5):
    for z in range(5):
        ax[i][z].imshow(outputs[i][0, :, :, z])
        ax[i][z].set_title(layer_names[i])
        ax[i][z].set_xticks([])
        ax[i][z].set_yticks([])
plt.savefig('filters.jpg')
# plt.show()
plt.close()

################################################################################
preds = model.predict(X_test)
argmax = np.argmax(preds[0])
output = model.output[:, argmax]

last_conv_layer = model.get_layer('max_pooling2d_2')
hif = .8

suffix = {'heatmap': '_heatmap.jpg',
          'heatmap_graph': '_heatmap_graph.png',
          'acc': 'accuracy_graph.png',
          'loss': 'loss_graph.png',
          'predict': 'predict.png',
          'cm': 'confusion_matrix.png'}

if args['heatmaps']:
    print('Generovanie map pozornosti...')
    if not os.path.exists(f'{args["heatmaps_path"]}\\{model_name}'):
        os.makedirs(f'{args["heatmaps_path"]}\\{model_name}')
    for idx, img in enumerate(X_test):
        grads = K.gradients(output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input],
                             [pooled_grads, last_conv_layer.output[idx]])
        pooled_grads_value, conv_layer_output_value = iterate([X_test])
        for i in range(64):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        heatmap_base = np.mean(conv_layer_output_value, axis=-1)
        heatmap_base = np.maximum(heatmap_base, 0)
        heatmap_base /= np.max(heatmap_base)
        heatmap = cv2.resize(heatmap_base, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * hif + img
        heatmap_name = f'{args["heatmaps_path"]}\\{model_name}\\{ids[idx]}'
        cv2.imwrite(f'{heatmap_name}{suffix["heatmap"]}', superimposed_img)
        img_heatmap = mpimg.imread(f'{heatmap_name}{suffix["heatmap"]}')
        plt.subplot(1, 2, 1)
        plt.imshow(img_heatmap)
        plt.title('Mapa pozornosti')
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.title('Originálny obrázok')
        plt.savefig(f'{heatmap_name}{suffix["heatmap_graph"]}')
        plt.close()
    print('Hotovo')

print('Generovanie grafov...')
if not os.path.exists(f'{args["graphs_path"]}\\{model_name}'):
    os.makedirs(f'{args["graphs_path"]}\\{model_name}')

graphs_path = f'{args["graphs_path"]}\\{model_name}\\'
plt.figure(figsize=(8, 8))
plt.plot(history.history['accuracy'], label='Presnosť trénovania')
plt.plot(history.history['val_accuracy'], label='Presnosť validácie')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.title('Presnosť')
plt.xlabel('Epochy')
plt.ylabel('Presnosť')
plt.legend()
plt.savefig(f'{graphs_path}{suffix["acc"]}')
plt.close()

plt.figure(figsize=(8, 8))
plt.plot(history.history['loss'], label='Chybovosť trénovania')
plt.plot(history.history['val_loss'], label='Chybovosť validácie')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.title('Chybovosť')
plt.xlabel('Epochy')
plt.ylabel('Chybovosť')
plt.legend()
plt.savefig(f'{graphs_path}{suffix["loss"]}')
plt.close()

plot_model(model)

y_pred = model.predict_classes(X_test)

Y_test = np.argmax(Y_test, axis=1)

y_pred = list(y_pred)
Y_test = list(Y_test)
Y_classes = [0, 1, 2]

plt.subplot(2, 2, 1)
plt.imshow(X_test[0])
plt.title("Pred: "  + categorize_class(y_pred[0]) +
          ", Real: " + categorize_class(Y_test[0]))
plt.subplot(2, 2, 2)
plt.imshow(X_test[1])
plt.title("Pred: " + categorize_class(y_pred[1]) +
          ", Real: " + categorize_class(Y_test[1]))
plt.subplot(2, 2, 3)
plt.imshow(X_test[2])
plt.title("Pred: " + categorize_class(y_pred[2]) +
          ", Real: " + categorize_class(Y_test[2]))
plt.subplot(2, 2, 4)
plt.imshow(X_test[3])
plt.title("Pred: " + categorize_class(y_pred[3]) + 
          ", Real: " + categorize_class(Y_test[3]))
plt.tight_layout(w_pad=1)
plt.savefig(f'{graphs_path}{suffix["predict"]}')
plt.close()

cm_path = f'{graphs_path}{suffix["cm"]}'
cm_analysis(Y_test, y_pred, Y_classes, cm_path,ymap=None, figsize=(8, 8),
            title="Konfúzna matica")
print('Hotovo')

print('Presnosť:', accuracy_score(y_pred, Y_test))
