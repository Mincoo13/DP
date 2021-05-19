import os
import os.path
import numpy as np
import config as cfg
from utils import cm_analysis, prepare_data, prepare_extracted_data, write_result
from utils import generate_line_graph, generate_reduced_graph, save_extracted_model

import matplotlib
matplotlib.use('Agg')

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

X_train, X_test, X_val, Y_train, Y_test, Y_val, ids = prepare_extracted_data(args)


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
    prepare_data(args, model=True)
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
print('MLP...')
mlp = MLPClassifier(max_iter=1000)

mlp_grid_params = args['mlp_parameters']
clf = GridSearchCV(mlp, mlp_grid_params, n_jobs=-1, cv=3, 
                   return_train_score=True)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, Y_test)

Y_test_cm = list(np.argmax(Y_test, axis=1))
y_pred = list(np.argmax(y_pred, axis=1))

Y_classes = [0, 1, 2]

accuracy = round(score*100, 2)
cm_path = f'{graphs_path}confusion_matrix_neural_network.png'
cm_analysis(Y_test_cm, y_pred, Y_classes, cm_path, ymap=None,
            figsize=(5, 5), title=f"Konfúzna matica pre MLP [{accuracy} %]")
print('Najlepsie vysledky dosiahla NS s nastavenim:\n', clf.best_params_)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


save_extracted_model(clf, f'{model_path}/mlp.sav')

generate_line_graph(graphs_path, 'mlp_mean_acc_graph.png', clf=clf)

result_row = {'model': f'{model_name}_mlp',
              'accuracy': accuracy,
              'test_size': len(y_pred),
              'settings': clf.best_params_}
fieldnames = ['model', 'accuracy', 'test_size', 'settings']
write_result(args['results_path'], fieldnames, result_row)

print("Presnosť MLP: ", accuracy)
print('Hotovo')
################################################################################
print('Support Vector Machine...')
a = np.array(Y_train)
Y_train_svm = np.where(a == 1)[1]

svm = SVC()
svm_grid_params = args['svm_parameters']
clf = GridSearchCV(svm, svm_grid_params, n_jobs=-1, cv=3, 
                   return_train_score=True)
clf.fit(X_train, Y_train_svm)
y_pred = clf.predict(X_test)
a = np.array(Y_test)
Y_test_svm = np.where(a==1)[1]
score = clf.score(X_test, Y_test_svm)

Y_test_cm = list(np.argmax(Y_test, axis=1))

Y_classes = [0, 1, 2]

accuracy = round(score*100, 2)
cm_path = f'{graphs_path}confusion_matrix_SVM.png'
cm_analysis(Y_test_cm, y_pred, Y_classes, cm_path, ymap=None,
            figsize=(5, 5), title=f"Konfúzna matica pre SVM [{accuracy} %]")
print('Najlepsie vysledky dosiahlo SVM s nastavenim:\n', clf.best_params_)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

save_extracted_model(clf, f'{model_path}/svm.sav')

generate_line_graph(graphs_path, 'svm_mean_acc_graph.png', clf=clf)

result_row = {'model': f'{model_name}_svm',
              'accuracy': accuracy,
              'test_size': len(y_pred),
              'settings': clf.best_params_}
fieldnames = ['model', 'accuracy', 'test_size', 'settings']
write_result(args['results_path'], fieldnames, result_row)

print("Presnosť Support Vector Machine: ", accuracy)
print('Hotovo')
################################################################################
print('Random Forest...')
forest = RandomForestClassifier()
forest_grid_params = args['forest_parameters']
clf = GridSearchCV(forest, forest_grid_params, n_jobs=-1, cv=3,
                   return_train_score=True)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, Y_test)

Y_test_cm = list(np.argmax(Y_test, axis=1))
y_pred = list(np.argmax(y_pred, axis=1))

Y_classes = [0, 1, 2]

accuracy = round(score*100, 2)
cm_path = f'{graphs_path}confusion_matrix_random_forest.png'
cm_analysis(Y_test_cm, y_pred, Y_classes, cm_path, ymap=None, figsize=(5, 5),
            title=f"Konfúzna matica pre Random Forest [{accuracy} %]")
print('Najlepsie vysledky dosiahol les s nastavenim:\n', clf.best_params_)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

save_extracted_model(clf, f'{model_path}/forest.sav')

generate_line_graph(graphs_path, 'forest_mean_acc_graph.png', clf=clf)

result_row = {'model': f'{model_name}_random_forest',
              'accuracy': accuracy,
              'test_size': len(y_pred),
              'settings': clf.best_params_}
fieldnames = ['model', 'accuracy', 'test_size', 'settings']
write_result(args['results_path'], fieldnames, result_row)

print("Presnosť Random Forest klasifikátora: ", accuracy)
print('Hotovo')
################################################################################
print('Decision Tree...')
tree = DecisionTreeClassifier()
tree_grid_params = args['tree_parameters']
clf = GridSearchCV(tree, tree_grid_params, n_jobs=-1, cv=3,
                   return_train_score=True)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, Y_test)

Y_test_cm = list(np.argmax(Y_test, axis=1))
y_pred = list(np.argmax(y_pred, axis=1))

accuracy = round(score*100, 2)
cm_path = f'{graphs_path}confusion_matrix_decision_tree.png'
cm_analysis(Y_test_cm, y_pred, Y_classes, cm_path, ymap=None, figsize=(5, 5),
            title=f"Konfúzna matica pre Decision Tree [{accuracy} %]")
print('Najlepsie vysledky dosiahol strom s nastavenim:\n', clf.best_params_)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

save_extracted_model(clf, f'{model_path}/tree.sav')

generate_line_graph(graphs_path, 'tree_mean_acc_graph.png', clf=clf)

result_row = {'model': f'{model_name}_decision_tree',
              'accuracy': accuracy,
              'test_size': len(y_pred),
              'settings': clf.best_params_}
write_result(args['results_path'], fieldnames, result_row)


print("Presnosť Decision Tree klasifikátora: ", accuracy)
print('Hotovo')
