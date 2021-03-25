import os
import os.path
import numpy as np
import config as cfg
from utils import cm_analysis, prepare_data, write_result
from utils import generate_graph, generate_reduced_graph

import matplotlib
matplotlib.use('Agg')

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

args = cfg.PRED_EXTRACT_HEALTH_ARGS

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

# def loader(photo):
#     photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
#     dim = (128, 128)
#     photo = cv2.resize(photo, dim, interpolation=cv2.INTER_NEAREST)
#     photo.reshape([-1, 128, 128, 1])
#     return photo

X_train, X_test, X_val, Y_train, Y_test, Y_val, ids = prepare_data(args, model=True)


print(X_train.shape)
print(Y_train.shape)
################################################################################
model_path = args['pretrained_model_path']
model_name = os.path.basename(os.path.normpath(model_path))
if not os.path.exists(f'{args["graphs_path"]}\\{model_name}_extracted'):
    os.makedirs(f'{args["graphs_path"]}\\{model_name}_extracted')

graphs_path = f'{args["graphs_path"]}\\{model_name}_extracted\\'
model_name = os.path.basename(os.path.normpath(args['pretrained_model_path']))

if args['minimized']:
    prepare_data(args, model=True, orig_data=True)
    print('Generovanie grafov zuzenych dat...')
    minimized_name = f"{args['minimized_path']}\\{model_name}_minimized"
    orig_labels_path = f"{minimized_name}\\labels_orig.csv"
    generate_reduced_graph(f"{minimized_name}\\2d_orig.csv", orig_labels_path, ',',
                           graphs_path, True, dim='2D')
    generate_reduced_graph(f"{minimized_name}\\3d_orig.csv", orig_labels_path, ',',
                           graphs_path, True, dim='3D')

    extracted_labels_path = f"{minimized_name}\\labels_features.csv"
    generate_reduced_graph(f"{minimized_name}\\2d_features.csv",
                           extracted_labels_path, ',', graphs_path,
                           False, dim='2D')
    generate_reduced_graph(f"{minimized_name}\\3d_features.csv",
                           extracted_labels_path, ',', graphs_path,
                           False, dim='3D')
    print('Hotovo')

################################################################################
print('Hladanie parametrov cez Grid Search...')
mlp = MLPClassifier(max_iter=1000)

grid_params = args['parameter_space']
clf = GridSearchCV(mlp, grid_params, n_jobs=-1, cv=3, return_train_score=True)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, Y_test)

Y_test_cm = list(np.argmax(Y_test, axis=1))
y_pred = list(np.argmax(y_pred, axis=1))

Y_classes = [0, 1, 2]

accuracy = round(score*100, 2)
cm_path = f'{graphs_path}confusion_matrix_neural_network.png'
cm_analysis(Y_test_cm, y_pred, Y_classes, cm_path, ymap=None,
            figsize=(5, 5), title=f"Heatmap pre NS [{accuracy} %]")
print('Najlepsie vysledky dosiahla NS s nastavenim:\n', clf.best_params_)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    # plt.plot(clf.loss_curve_)
    # plt.show()
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

generate_graph(graphs_path, 'mean_acc_graph.png', clf=clf)

result_row = {'model': f'{model_name}_grid_search',
              'accuracy': accuracy,
              'test_size': len(y_pred),
              'settings': clf.best_params_}
fieldnames = ['model', 'accuracy', 'test_size', 'settings']
write_result(args, fieldnames, result_row)

print('Hotovo')
################################################################################
print('Random Forest...')
clf = RandomForestClassifier(max_depth=12, n_estimators=340, random_state=0)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, Y_test)

Y_test_cm = list(np.argmax(Y_test, axis=1))
y_pred = list(np.argmax(y_pred, axis=1))

Y_classes = [0, 1, 2]

accuracy = round(score*100, 2)
cm_path = f'{graphs_path}confusion_matrix_random_forest.png'
cm_analysis(Y_test_cm, y_pred, Y_classes, cm_path, ymap=None,
            figsize=(5, 5), title=f"Heatmap pre Random Forest [{accuracy} %]")

result_row = {'model': f'{model_name}_random_forest',
              'accuracy': accuracy,
              'test_size': len(y_pred),
              'settings': clf.get_params()}
fieldnames = ['model', 'accuracy', 'test_size', 'settings']
write_result(args, fieldnames, result_row)

print("Presnosť Random Forest klasifikátora: ", accuracy)
print('Hotovo')
################################################################################
print('Decision Tree...')
clf = DecisionTreeClassifier(max_depth=11, min_samples_split=150)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, Y_test)

Y_test_cm = list(np.argmax(Y_test, axis=1))
y_pred = list(np.argmax(y_pred, axis=1))

accuracy = round(score*100, 2)
cm_path = f'{graphs_path}confusion_matrix_decision_tree.png'
cm_analysis(Y_test_cm, y_pred, Y_classes, cm_path, ymap=None,
            figsize=(5, 5), title=f"Heatmap pre Decision Tree [{accuracy} %]")

result_row = {'model': f'{model_name}_decision_tree',
              'accuracy': accuracy,
              'test_size': len(y_pred),
              'settings': clf.get_params()}
write_result(args, fieldnames, result_row)


print("Presnosť Decision Tree klasifikátora: ", accuracy)
print('Hotovo')