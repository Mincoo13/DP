import os
import csv
import pickle
import collections
import pandas as pd
import seaborn as sns
import numpy as np
from numpy import genfromtxt
    
import cv2
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.models import model_from_json
from keras.models import Model

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

def cm_analysis(y_true, y_pred, labels, path, ymap=None, figsize=(10, 10), title=""):
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

def prepare_data(args, model=False, orig_data=False):
    df = pd.read_csv(args['csv_path'], sep=args['csv_sep'])
    health_data = df['health'].to_numpy()
    labels = collections.Counter(health_data).keys()
    nums = collections.Counter(health_data).values()
    img_dim = args['img_dim']

    # print(nums)
    # plt.figure(figsize=(8, 8))
    # plt.bar(labels, nums)
    # plt.title('Zdravie')
    # plt.xlabel('kategórie')
    # plt.ylabel('počet')
    # plt.show()
    x_data, y_data, ids = load_data(args, df, orig=orig_data)


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

            # print(X_original.shape)
            # print(Y.shape)

            np.savetxt(f"{minimized_name}\\2d_orig.csv",
                       X_original_2, delimiter=",")
            np.savetxt(f"{minimized_name}\\3d_orig.csv",
                       X_original_3, delimiter=",")
            np.savetxt(f"{minimized_name}\\labels_orig.csv", Y_ax, delimiter=",")
            print('Hotovo')
            return
        print('Extrakcia priznakov...')
        model, model_name, _ = load_model(args)
        # pred_classes = model.predict(X)
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


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5)

    return X_train, X_test, X_val, Y_train, Y_test, Y_val, ids

def generate_graph(graphs_path, suffix, history=None, intent=None, clf=None):
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
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#666666', linestyle='-', alpha=0.2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(title)
    plt.legend()
    plt.savefig(f'{graphs_path}{suffix}')
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

def write_result(args, fieldnames, result):
    with open(args["results_path"], 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(result)
