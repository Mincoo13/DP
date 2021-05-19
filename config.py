"""
-----dataset.py-----
Argumenty pre skript na rozsirenie datasetu.
current_folder: Nastavenie, ci cesta k suborom pocina aktualnym umiestnenim suboru
                Boolean hodnota - ak je True, aktualna cesta pracovneho
                prostredia bude nastavena na umiestnenie daneho suboru
old_csv_path: Nastavuje cestu povodneho csv suboru
new_csv_path: Nastavuje cestu novovytvoreneho rozsireneho csv suboru
csv_sep: Oddelovac pouzity v csv subore
imgs_path : Cesta k povodnemu datasetu, ktory sa bude rozsirovat
results_path : Cesta vysledneho rozsireneho datasetu
"""
DATASET_ARGS = {'current_folder': True,
                'old_csv_path': r'data\dataset_1.csv',
                'new_csv_path': r'data\new_dataset_1.csv',
                'csv_sep': ';',
                'imgs_path': r'..\dp_dataset\lemna_all_cropped_reflection',
                'results_path': r'new_dataset_1'
                }


"""
-----predict_cnn_health.py-----
Argumenty pre skript na predikciu zdravia rastliny lemna minor pomocou CNN.
current_folder: Nastavenie, ci cesta k suborom pocina aktualnym umiestnenim suboru
                Boolean hodnota - ak je True, aktualna cesta pracovneho
                prostredia bude nastavena na umiestnenie daneho suboru
csv_path: Nastavuje cestu k csv suboru
csv_sep: Oddelovac pouzity v csv subore
results_path: Cesta k csv suboru, do ktoreho sa zapise vysledok z trenovania
imgs_path: Cesta k datasetu pouzitom na trenovanie/testovanie/validaciu
img_dim: Udava rozmer, na ktory sa maju fotografie zmensit. Tuple
pretrained_model: Nastavuje, ci sa ma pouzit predtrenovany model s nastavenymi
                  vahami. Boolean hodnota - ak je True, pouzije sa predtrenovany
                  model, ak je False, spusti sa nove trenovanie.
pretrained_model_path: Nastavuje cestu k predtrenovanemu modelu. Cielom ma byt
                       priecinok, v ktorom, sa bude nachadzat subor s nastaveniami
                       modelu (.json), subor s vahami (.h5) a tiez subor s historiou
                       trenovania (.pickle). Subory musia mat rovnaky nazov ako
                       priecinok, v ktorom sa nachadzaju.
                       Premenna sa pouzije iba v pripade, ak je pretrained_model
                       nastavene na True
model_path: Nastavuje cestu, do ktorej sa ulozi model po trenovani. Cielom je
            priecinok, do ktoreho sa ulozia tri subory pod rovnakym nazvom.
            Premenna sa pouzije iba v pripade, ak je pretrained_model nastavene
            na False. Nazov priecinka, do ktoreho sa model ulozi bude taktiez
            nazvom priecinka pre grafy
heatmaps: Nastavuje, ci sa maju vygenerovat mapy pozornosti pre kazdu polozku
          testovacich dat. Boolean hodnota - ak je nastavena na True, mapy sa
          vygeneruju.
heatmaps_path: Nastavuje ciel, do ktoreho sa ulozia vygenerovane mapy pozornosti.
               Premenna sa pouzije iba v pripade, ak je heatmaps nastavene na True
graphs_path: Nastavuje ciel, do ktoreho sa ulozia vsetky vygenerovane grafy.
"""
PRED_CNN_HEALTH_ARGS = {'current_folder': True,
                        'csv_path': r'data\new_dataset_1.csv',
                        'csv_sep': ',',
                        'results_path': r'data\health_accuracy_results.csv',
                        'imgs_path': r'segmented_results\segmented_7_splash',
                        'img_dim': (128, 128),
                        'pretrained_model': False,
                        'pretrained_model_path': r'models\model_2',
                        'model_path': r'models\health_classification\segmented_5_2_softplus_cat_adadelta_40_early',
                        'heatmaps': False,
                        'heatmaps_path': r'heatmaps',
                        'graphs_path': r'graphs\health_cnn'
                        }


"""
-----predict_extracted_health.py-----
Argumenty pre skript na predikciu zdravia rastliny lemna minor pomocou
extrahovanych priznakov a nasledneho pouzitia klasickej neuronovej siete,
rozhodovacieho stromu, nahodneho lesa a SVM.
current_folder: Nastavenie, ci cesta k suborom pocina aktualnym umiestnenim suboru
                Boolean hodnota - ak je True, aktualna cesta pracovneho
                prostredia bude nastavena na umiestnenie daneho suboru
csv_path: Nastavuje cestu k csv suboru
csv_sep: Oddelovac pouzity v csv subore
results_path: Cesta k csv suboru, do ktoreho sa zapise vysledok z trenovania
imgs_path: Cesta k datasetu pouzitom na trenovanie/testovanie/validaciu
img_dim: Udava rozmer, na ktory sa maju fotografie zmensit. Tuple
pretrained_model_path: Cesta k modelu, pomocou ktoreho sa extrahuju priznaky
                       z datasetu
minimized: Boolean hodnota, ktora nastavuje, ci sa ma vykonat zuzenie extrahovanych
           a originalnych dat na vizualne zobrazenie 2D/3D priestoru v grafoch
minimized_path: Cesta, kam sa vygeneruju a ulozia csv subory so zuzenymi datami.
graphs_path: Nastavuje ciel, do ktoreho sa ulozia vsetky vygenerovane grafy.
mlp_parameters: Nastavenia trenovania neuronovej siete, ktore sa budu cez Grid
                Search prehladavat. Vstupom je dict obsahujuci zoznamy vsetkych
                nastaveni, ktore sa maju vyskusat.
tree_parameters: Nastavenia Grid Search pre rozhodovaci strom.
forest_parameters: Nastavenia Grid Search pre nahodny les.
svm_parameters: Nastavenia Grid Search pre SVM.
"""
PRED_EXTRACT_HEALTH_ARGS = {'current_folder': True,
                            'csv_path': r'data\new_dataset_1.csv',
                            'csv_sep': ',',
                            'results_path': r'data\health_accuracy_results.csv',
                            'imgs_path': r'new_dataset',
                            'img_dim': (128, 128),
                            'pretrained_model_path': r'models\health_classification\5_2_soft_cat_adadelta_100_early',
                            'minimized': False,
                            'minimized_path': r'data\feature_extraction',
                            'graphs_path': r'graphs\feature_extraction',
                            'mlp_parameters': {
                                'hidden_layer_sizes': [(50, 50, 50), (100,), (50, 20)],
                                'activation': ['tanh', 'relu'],
                                'solver': ['sgd', 'adam'],
                                'alpha': [0.0001, 0.05],
                                'learning_rate': ['constant', 'adaptive'],
                            },
                            'tree_parameters': {
                                'criterion': ['gini', 'entropy'],
                                'splitter': ['best', 'random'],
                                'max_depth': [10, 20, 40, 60, 80, 100, None],
                                'min_samples_leaf': [1, 2, 4],
                                'min_samples_split': [2, 5, 10],
                            },
                            'forest_parameters': {
                                'bootstrap': [True, False],
                                'max_depth': [10, 40, 100, None],
                                'max_features': ['auto', 'sqrt'],
                                'min_samples_leaf': [1, 2, 4],
                                'min_samples_split': [2, 5, 10],
                                'n_estimators': [100, 200, 400, 1200]
                            },
                            'svm_parameters': {
                                'C': [0.1, 1, 10, 100, 1000],
                                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                                'kernel': ['rbf', 'poly']
                            }
                            }


'''
-----segment_train.py-----
Argumenty pre trenovanie Mask R-CNN na segmentaciu masiek.
current_folder: Nastavenie, ci cesta k suborom pocina aktualnym umiestnenim suboru
                Boolean hodnota - ak je True, aktualna cesta pracovneho
                prostredia bude nastavena na umiestnenie daneho suboru
model_pretrained: Cesta k modelu, ktory sa pouzije na upravu vah pri trenovani.
logs: Priecinok, do ktoreho sa ulozia vysledne modely.
dataset_path: Cesta k datasetu.
annot_path: Cesta k priecinku s anotovanymi datami. Priecinok musi obsahovat dva
            podpriecinky "train" a "val", pricom kazdy z nich obsahuje prisluchajuce
            fotografie a json subor s anotaciami k nim.
epochs: Pocet epoch trenovania.
graphs_path: Nastavuje ciel, do ktoreho sa ulozia vsetky vygenerovane grafy.
'''
SEGMENT_TRAIN_ARGS = {'current_folder': True,
                      'model_pretrained': r'models\leafSegmenter0005.h5',
                      'logs': r'models\68_20_resnet50_head_layers_reg0_01_test',
                      'dataset_path': r'dataset_segmenter',
                      'annot_path': r'{dataset_path}\{subset}\lemna_minor_annotations_overlap.json',
                      'epochs': 1,
                      'graphs_path': r'graphs\segmentation_losses'
                      }


'''
-----segment_predict.py-----
Argumenty na predikciu segmentacie dat natrenovaneho modelu pomocou M-RCNN.
current_folder: Nastavenie, ci cesta k suborom pocina aktualnym umiestnenim suboru
                Boolean hodnota - ak je True, aktualna cesta pracovneho
                prostredia bude nastavena na umiestnenie daneho suboru
intent: Zamer skriptu, podla ktoreho sa vykona segmentacia.
        Moze byt "mask" - v
        tomto pripade sa vytvori maska pre kazdy vstup, spocitaju sa na nich listky,
        plocha. Tieto predikcie sa porovnaju s realnymi datami a vygeneruju sa
        MAE a MSE.
        Druha moznost "splash" - Z fotografii sa extrahuju segmentovane casti,
        ktore budu mat zanechanu farbu. Vypocet plochy, listkov a errorov sa
        v tomto pripade nevykona.
splash_color: Farba pozadia. Moze byt "black" - v tomto pripade sa farba pozadia
              vyfarbi na cierno. Druha moznost je "gray" - Pozadnie bude v style
              black & white. Tretia moznost "overlap" - Fotografia sa prekonvertuje
              na HSV format, pricom hodnota hue sa posunie tak, aby sa zelene
              farby zmenili na cervene. Segmentovane masky pritom budu mat zelenu
              farbu a tak bude mozne vidiet, ktore casti su segmentovane (vratane
              prekrytia) a ktore nie (vsetko okrem zelenej).
mask_color: Farba segmentovanej masky fotografii. Moze byt "rainbow" - v tomto
            pripade sa vygeneruju segmentovane casti v roznych farbach. Druha
            moznost je zadat priamo farbu v RGB - napr. pre zelenu (0, 168, 0).
csv_path: Nastavuje cestu k csv suboru.
imgs_path: Cesta k obrazkom, na ktorych bude aplikovana segmentacia.
annot_path: Cesta k json suboru anotovanych dat. Predpoklada sa anotacia prostrednictvom
            nastroja VGG Image Annotator.
output_path: Cesta, kam sa ulozia vygenerovane segmentovane obrazky.
results_path: Cesta k csv suboru, do ktoreho sa ulozia vysledne hodnoty errorov.
predictions_path: Cesta, kam sa ulozi csv subor s predikovanymi poctami listkov/plochy
                  a realnych hodnot tychto parametrov.
model_path: Cesta k modelu, ktory sa pouzije na segmentaciu.
graphs_path: Cesta, kam sa vygeneruju grafy predikcii.
'''
SEGMENT_PRED_ARGS = {'current_folder': True,
                     'intent': 'splash',
                     'splash_color': 'overlap',
                     'mask_color': (0, 168, 0),
                     'csv_path': r'data\new_dataset_1.csv',
                     'imgs_path': r'new_dataset_1',
                     'annot_path': r'dataset\lemna_minor_annotations.json',
                     'output_path': r'segmented_results',
                     'results_path': r'data\segmentation_results.csv',
                     'predictions_path': r'data\segmentation_predictions',
                     'model_path': r'models\segmentation\model_segment_overlap_68_20_resnet50_head_layers_reg0_01\mask_rcnn_68_20_resnet50_head_layers_reg0_01_0019.h5',
                     'graphs_path': r'graphs\segmentation_predictions'
                     }

'''
-----visualize_annot_data.py-----
Skript na vizualizaciu manualnej anotacie dat.
current_folder: Nastavenie, ci cesta k suborom pocina aktualnym umiestnenim suboru
                Boolean hodnota - ak je True, aktualna cesta pracovneho
                prostredia bude nastavena na umiestnenie daneho suboru
annot_path: Cesta k json suboru anotovanych dat. Predpoklada sa anotacia prostrednictvom
            nastroja VGG Image Annotator.
imgs_path: Cesta k obrazkom, na ktorych bude aplikovana segmentacia.
output_path: Cesta, kam sa ulozia vygenerovane segmentovane obrazky.
mask_color: Farba segmentovanej masky fotografii. Moze byt "rainbow" - v tomto
            pripade sa vygeneruju segmentovane casti v roznych farbach. Druha
            moznost je zadat priamo farbu v RGB - napr. pre zelenu (0, 168, 0).
'''
VIS_ANNOT_DATA = {'current_folder': True,
                  'annot_path': r'dataset\lemna_minor_annotations.json',
                  'imgs_path': r'new_dataset',
                  'output_path': r'segmented_results\annotated_masks',
                  'mask_color': (0, 168, 0)
                  }
