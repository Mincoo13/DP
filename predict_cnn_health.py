import os
import pickle
import numpy as np
import config as cfg
from utils import cm_analysis, prepare_data, generate_line_graph, load_model
from utils import write_result, save_np
from train_health_model import set_model

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import plot_model

from keras import backend as K
from keras.models import Model

from sklearn.metrics import accuracy_score

args = cfg.PRED_CNN_HEALTH_ARGS

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

X_train, X_test, X_val, Y_train, Y_test, Y_val, ids = prepare_data(args)
################################################################################

input_shape = X_train.shape[1:]
if args['pretrained_model']:
    model, model_name, history = load_model(args)
else:
    print('Trenovanie...')
    model_name = os.path.basename(os.path.normpath(args['model_path']))
    model_path = f"{args['model_path']}\\{model_name}"

    model, history = set_model(input_shape, X_train, Y_train, X_val, Y_val)

    if not os.path.exists(args['model_path']):
        os.makedirs(args['model_path'])

    model_json = model.to_json()
    with open(f'{model_path}.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(f'{model_path}.h5')
    with open(f'{model_path}.pickle', 'wb') as file_pi:
        pickle.dump(history, file_pi)
    print('Hotovo')

save_np(f"{args['model_path']}/X_train.npy", X_train)
save_np(f"{args['model_path']}/X_test.npy", X_test)
save_np(f"{args['model_path']}/X_val.npy", X_val)
save_np(f"{args['model_path']}/Y_train.npy", Y_train)
save_np(f"{args['model_path']}/Y_test.npy", Y_test)
save_np(f"{args['model_path']}/Y_val.npy", Y_val)
save_np(f"{args['model_path']}/ids.npy", ids)

################################################################################
suffix = {'heatmap': '_heatmap.jpg',
          'heatmap_graph': '_heatmap_graph.png',
          'acc': 'accuracy_graph.png',
          'loss': 'loss_graph.png',
          'predict': 'predict.png',
          'cm': 'confusion_matrix.png',
          'model': f'{model_name}.png'}

if args['heatmaps']:
    print('Generovanie map pozornosti...')
    preds = model.predict(X_test)
    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]
    try:
        last_conv_layer = model.get_layer('heatmap_layer')
    except:
        last_conv_layer = model.get_layer('max_pooling2d_2')
    hif = .8
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

################################################################################
if not os.path.exists(f'{args["graphs_path"]}\\{model_name}'):
    os.makedirs(f'{args["graphs_path"]}\\{model_name}')

graphs_path = f'{args["graphs_path"]}\\{model_name}\\'

print('Generovanie filtrov...')
layer_names = []
for layer in model.layers:
    if layer.name == 'flatten_1':
        break
    layer_names.append(layer.name)

for idx in range(5):
    outputs = []
    img_filter = X_test[idx]
    img_filter = img_filter.reshape([-1, 128, 128, 3])

    for layer_name in layer_names:
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(img_filter)
        outputs.append(intermediate_output)

    dpi = mpl.rcParams['figure.dpi']
    figsize = 5 * (256 / float(dpi)), len(outputs) * (256 / float(dpi))

    fig, ax = plt.subplots(nrows=len(outputs), ncols=5,
                           figsize=figsize)

    for i in range(len(outputs)):
        for z in range(5):
            figure = outputs[i][0, :, :, z]
            ax[i][z].imshow(figure)
            ax[i][z].set_title(layer_names[i])
            ax[i][z].set_xticks([])
            ax[i][z].set_yticks([])
    plt.tight_layout(w_pad=1)
    plt.savefig(f'{graphs_path}filters_{idx}.png')
    plt.close()
print('Hotovo')
################################################################################
print('Generovanie grafov...')

generate_line_graph(graphs_path, suffix, history, 'accuracy')
generate_line_graph(graphs_path, suffix, history, 'loss')

plot_model(model, f'{graphs_path}{suffix["model"]}')

y_pred = model.predict_classes(X_test)

Y_test = np.argmax(Y_test, axis=1)

y_pred = list(y_pred)
Y_test = list(Y_test)
Y_classes = [0, 1, 2]

for i in range(0, 4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(X_test[i])
    plt.title("Pred: "  + categorize_class(y_pred[i]) +
              ", Real: " + categorize_class(Y_test[i]))
plt.tight_layout(w_pad=1)
plt.savefig(f'{graphs_path}{suffix["predict"]}')
plt.close()

accuracy = accuracy_score(y_pred, Y_test)
cm_path = f'{graphs_path}{suffix["cm"]}'
cm_analysis(Y_test, y_pred, Y_classes, cm_path, ymap=None, figsize=(8, 8),
            title=f"Konfúzna matica [{accuracy*100} %]")
print('Hotovo')

print('Presnosť:', accuracy)

result_row = {'model': model_name,
              'accuracy': accuracy,
              'test_size': len(y_pred),
              'settings': model.get_config()}
fieldnames = ['model', 'accuracy', 'test_size', 'settings']

write_result(args['results_path'], fieldnames, result_row)

print(accuracy_score(y_pred, Y_test))
print(len(y_pred))
print(model_name)
