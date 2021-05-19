import os
import csv
import json
import pickle
import itertools
import collections
import skimage.draw
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

def cm_analysis(y_true, y_pred, labels, path, ymap=None, figsize=(10, 10), title=""):
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
    cm.index.name = 'Reálne'
    cm.columns.name = 'Predikované'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap='viridis')
    plt.title(title)
    plt.savefig(path)
    plt.close()

def load_data(args, df, orig=False):
    # FOR TESTING ONLY
    # df = df.sample(frac=1)
    count = 0
    # ---------

    x_data = []
    y_data = []
    ids = []

    print('Nacitanie obrazkov...')
    for index, row in df.iterrows():

        # # FOR TESTING ONLY
        # count += 1
        # if count == 200:
        #     break
        # # ----------------

        print(f'{args["imgs_path"]}\\{row["id"]}.jpg')
        if orig:
            image = cv2.imread(f'{args["imgs_path"]}\\{row["id"]}.jpg',
                               cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(f'{args["imgs_path"]}\\{row["id"]}.jpg',
                               cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, args['img_dim'],
                                   interpolation=cv2.INTER_NEAREST)
        x_data.append(np.array(image_resized))
        y_data.append(row['health'])
        ids.append(row['id'])
    print('Hotovo')
    return x_data, y_data, ids

def load_model(args):
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
    return model, model_name, history

def save_extracted_model(clf, path):
    pickle.dump(clf, open(path, 'wb'))

def save_np(path, data):
    with open(path, 'wb') as f:
        np.save(f, data)

def load_np(path):
    with open(path, 'rb') as f:
        data = np.load(f)
        return data

def prepare_data(args, model=False, orig_data=False):
    df = pd.read_csv(args['csv_path'], sep=args['csv_sep'])
    health_data = df['health'].to_numpy()
    labels = collections.Counter(health_data).keys()
    nums = collections.Counter(health_data).values()
    img_dim = args['img_dim']

    x_data, y_data, ids = load_data(args, df, orig=orig_data)

    print('Spracovanie dat...')
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    ids = np.array(ids)

    x_data = x_data.astype('float32')/255

    y_encoded = LabelEncoder().fit_transform(y_data)
    y_categorical = to_categorical(y_encoded)

    r = np.arange(x_data.shape[0])
    np.random.shuffle(r)
    X = x_data[r]
    Y = y_categorical[r]
    Y_ax = np.argmax(Y, axis=1)
    ids = ids[r]
    print('Hotovo')
    if model:
        model_path = args['pretrained_model_path']
        model_name = os.path.basename(os.path.normpath(model_path))
        minimized_name = f"{args['minimized_path']}\\{model_name}_minimized"
        if not os.path.exists(minimized_name):
            os.makedirs(minimized_name)
        if orig_data and args['minimized']:
            print('Zuzenie originalneho datasetu...')
            X_original = X.reshape((len(x_data), img_dim[0]*img_dim[1]))
            X_original_2 = TSNE(n_components=2).fit_transform(X_original)
            X_original_3 = TSNE(n_components=3).fit_transform(X_original)

            np.savetxt(f"{minimized_name}\\2d_orig.csv",
                       X_original_2, delimiter=",")
            np.savetxt(f"{minimized_name}\\3d_orig.csv",
                       X_original_3, delimiter=",")
            np.savetxt(f"{minimized_name}\\labels_orig.csv", Y_ax, delimiter=",")
            print('Hotovo')
            return
        print('Extrakcia priznakov...')
        model, model_name, _ = load_model(args)
        model.summary()
        model.pop()
        model.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        model.summary()

        features = model.predict(X)
        features_normalized = preprocessing.MinMaxScaler(feature_range=(0, 1))
        X = features_normalized.fit_transform(features)
        print('Hotovo')

        if args['minimized']:
            print('Zuzenie dat extrahovanych priznakov...')
            X_2 = TSNE(n_components=2).fit_transform(features)
            np.savetxt(f"{minimized_name}\\2d_features.csv", X_2, delimiter=",")

            X_3 = TSNE(n_components=3).fit_transform(features)
            np.savetxt(f"{minimized_name}\\3d_features.csv", X_3, delimiter=",")
            np.savetxt(f"{minimized_name}\\labels_features.csv", Y_ax, delimiter=",")
            print('Hotovo')
        return


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5)

    return X_train, X_test, X_val, Y_train, Y_test, Y_val, ids

def extract_features(data, model):
    features = model.predict(data)
    features_normalized = preprocessing.MinMaxScaler(feature_range=(0, 1))
    data = features_normalized.fit_transform(features)
    return data

def prepare_extracted_data(args):
    
    img_dim = args['img_dim']
    model_path = args['pretrained_model_path']
    model_name = os.path.basename(os.path.normpath(model_path))
    minimized_name = f"{args['minimized_path']}\\{model_name}_minimized"

    X_train = load_np(f'{model_path}/X_train.npy')
    X_test = load_np(f'{model_path}/X_test.npy')
    X_val = load_np(f'{model_path}/X_val.npy')
    Y_train = load_np(f'{model_path}/Y_train.npy')
    Y_test = load_np(f'{model_path}/Y_test.npy')
    Y_val = load_np(f'{model_path}/Y_val.npy')
    ids = load_np(f'{model_path}/ids.npy')

    print('Extrakcia priznakov...')
    model, model_name, _ = load_model(args)
    model.summary()
    model.pop()
    model.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    model.summary()
    X_train = extract_features(X_train, model)
    X_test = extract_features(X_test, model)
    X_val = extract_features(X_val, model)
    print('Hotovo')

    return X_train, X_test, X_val, Y_train, Y_test, Y_val, ids

def generate_line_graph(graphs_path, suffix, history=None, intent=None, clf=None):
    plt.figure(figsize=(8, 8))
    if clf:
        axes = plt.axes()
        axes.set_ylim([0, 1])
        x_label = 'Nastavenia'
        title = 'Priemerná presnosť '
        hstr = clf.cv_results_
        plt.plot(hstr['mean_test_score'], label=f'{title} testovacej vzorky')
        plt.plot(hstr['mean_train_score'], label=f'{title} trénovacej vzorky')
    else:
        if intent == 'accuracy':
            hstr = [history.history['accuracy'], history.history['val_accuracy']]
            title = 'Presnosť'
            suffix = suffix["acc"]
        else:
            hstr = [history.history['loss'], history.history['val_loss']]
            title = 'Chybovosť'
            suffix = suffix["loss"]
        x_label = 'Epochy'
        plt.plot(hstr[0], label=f'{title} trénovania')
        plt.plot(hstr[1], label=f'{title} validácie')
    ax = plt.gca()
    if clf:
        plt.xlim(-1, len(hstr['mean_fit_time']))
    else:
        plt.xlim(-1, len(hstr[0]))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#666666', linestyle='-', alpha=0.2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(title)
    plt.legend()
    plt.savefig(f'{graphs_path}{suffix}')
    plt.close()

def generate_segment_graph(graphs_path, y_pred, y_test, intent='leaf_count'):
    if intent == 'leaf_count':
        blue_patch_label = 'Predikované počty lístkov'
        mag_patch_label = 'Reálne počty lístkov'
        ylabel = 'Počet lístkov'
        title = 'Počty lístkov po segmentácii'
        save_name = 'leaf_count.png'
    elif intent == 'area':
        blue_patch_label = 'Predikovaná plocha v pixeloch'
        mag_patch_label = 'Reálna plocha v pixeloch'
        ylabel = 'Plocha v pixeloch'
        title = 'Plocha v pixeloch'
        save_name = 'area.png'
    else:
        return
    blue_patch = mpatches.Patch(color='blue', label=blue_patch_label)
    magenta_patch = mpatches.Patch(color='darkviolet', label=mag_patch_label)
    plt.figure(figsize=[10, 4.8])
    plt.legend(handles=[magenta_patch, blue_patch])
    plt.plot(y_pred, 'b+', y_test, 'mx', alpha=0.5)
    plt.ylabel(ylabel)
    plt.xlabel('Inštancie')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#666666', linestyle='-', alpha=0.2)
    plt.title(title)
    plt.savefig(f'{graphs_path}/{save_name}')
    plt.close()

def generate_reduced_graph(x_path, y_path, delimeter, graph_path, orig, dim='2D'):
    X = genfromtxt(x_path, delimiter=delimeter)
    Y = genfromtxt(y_path, delimiter=delimeter)
    if orig:
        title = f"Originálny dataset (Zobrazenie {dim} priestoru)"
        save_path = f'{graph_path}original_{dim}.png'
    else:
        title = f"Extrahované príznaky  dataset (Zobrazenie {dim} priestoru)"
        save_path = f'{graph_path}extracted_{dim}.png'
    if dim is '2D':
        ax = plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='Set1', s=1)
    else:
        ax = plt.axes(projection="3d").scatter3D(X[:, 0], X[:, 1], X[:, 2],
                                                 c=Y, cmap='Set1', s=1)
    plt.legend(*ax.legend_elements(), loc="lower left", title="kategórie")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def write_result(path, fieldnames, result):
    with open(path, 'a', newline='') as f, open(path, 'r') as r:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        data = [row for row in list(csv.reader(r))]
        if len(data) is 0:
            writer.writeheader()
        writer.writerow(result)

def annotation_to_mask(args):
    with open(args['annot_path']) as f:
        data = json.load(f)
    data = {k.split(".")[0]:v for k,v in data.items()}
    actual_masks = {}
    for im_id in data:
        try:
            im = Image.open(f"{args['imgs_path']}/{im_id}.jpg")
            one_mask = []
            for i, _ in enumerate(data[im_id]["regions"]):
                x_points = data[im_id]["regions"][i]["shape_attributes"]["all_points_x"]
                y_points = data[im_id]["regions"][i]["shape_attributes"]["all_points_y"]
                leaf_points = []
                for j, x_point in enumerate(x_points):
                    leaf_points.append((x_point, y_points[j]))
                img = Image.new("L", im.size, 0)
                ImageDraw.Draw(img).polygon(list(map(tuple, leaf_points)),
                                            outline=1, fill=1)
                leaf_mask = np.array(img)
                one_mask.append(leaf_mask)
            actual_masks[im_id] = one_mask
        except:
            continue
    return actual_masks

def mask_to_rgb(mask, alpha=0.1, predicted=True, color='rainbow'):
    if predicted:
        leaf_count = mask.shape[2]
    else:
        leaf_count = mask.shape[0]

    if color == 'rainbow':
        color = visualize.random_colors(leaf_count)
    else:
        color = list(itertools.repeat(color, leaf_count))

    print(mask.shape)
    if predicted:
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
        for i in range(mask.shape[2]):
            for c in range(3):
                rgb_mask[:, :, c] = np.where(mask[:, :, i] != 0,
                                             rgb_mask[:, :, c] *
                                             (1 - alpha) + alpha *
                                             color[i][c] * 255,
                                             rgb_mask[:, :, c])
    else:
        rgb_mask = np.zeros((mask.shape[1], mask.shape[2], 3))
        for i in range(mask.shape[0]):
            for c in range(3):
                rgb_mask[:, :, c] = np.where(mask[i, :, :] != 0,
                                             rgb_mask[:, :, c] *
                                             (1 - alpha) + alpha *
                                             color[i][c] * 255,
                                             rgb_mask[:, :, c])
    return rgb_mask

def color_splash(image, mask, color='black'):

    if color == 'black':
        bckgrnd = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.uint8)
        bckgrnd.fill(0)
    elif color == 'gray':
        bckgrnd = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    elif color == 'overlap':
        bckgrnd = image.copy()
        hsv = cv2.cvtColor(bckgrnd, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        shift_h = (h - 220)
        shift_hsv = cv2.merge([shift_h, s, v])
        bckgrnd = cv2.cvtColor(shift_hsv, cv2.COLOR_HSV2BGR)
    else:
        print('Nie je mozne aplikovat tuto farbu. Vyberte "black" alebo "gray".')
        return
    print(mask.shape)
    if mask.shape[-1] > 0:
        if color is 'overlap':
            mask = mask_to_rgb(mask, alpha=0.5, color=(0, 168, 0))
        else:
            mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, mask, bckgrnd).astype(np.uint8)
    else:
        splash = bckgrnd.astype(np.uint8)
    return splash