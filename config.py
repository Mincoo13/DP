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
                'new_csv_path': r'data\new_dataset.csv',
                'csv_sep': ';',
                'imgs_path': r'..\dp_dataset\lemna_all_cropped_reflection',
                'results_path': r'new_dataset'
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
            na False
heatmaps: Nastavuje, ci sa maju vygenerovat mapy pozornosti pre kazdu polozku
          testovacich dat. Boolean hodnota - ak je nastavena na True, mapy sa
          vygeneruju.
heatmaps_path: Nastavuje ciel, do ktoreho sa ulozia vygenerovane mapy pozornosti.
               Premenna sa pouzije iba v pripade, ak je heatmaps nastavene na True
graphs_path: Nastavuje ciel, do ktoreho sa ulozia vsetky vygenerovane grafy.
"""
PRED_CNN_HEALTH_ARGS = {'current_folder': True,
                        'csv_path': r'data\new_dataset.csv',
                        'csv_sep': ',',
                        'results_path': r'data\results.csv',
                        'imgs_path': r'new_dataset',
                        'img_dim': (128, 128),
                        'pretrained_model': False,
                        'pretrained_model_path': r'models\model_2',
                        'model_path': r'models\model_3',
                        'heatmaps': False,
                        'heatmaps_path': r'heatmaps',
                        'graphs_path': r'graphs'
                        }


"""
-----predict_extracted_health.py-----
Argumenty pre skript na predikciu zdravia rastliny lemna minor pomocou
extrahovanych priznakov a nasledneho pouzitia klasickej neuronovej siete,
rozhodovacieho stromu a random forestu.
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
parameter_space: Nastavenia trenovania neuronovej siete, ktore sa budu cez Grid
                 Search prehladavat. Vstupom je dict obsahujuci zoznamy vsetkych
                 nastaveni, ktore sa maju vyskusat.
minimized: Boolean hodnota, ktora nastavuje, ci sa ma vykonat zuzenie extrahovanych
           a originalnych dat na vizualne zobrazenie 2D/3D priestoru v grafoch
minimized_path: Cesta, kam sa vygeneruju a ulozia csv subory so zuzenymi datami
graphs_path: Nastavuje ciel, do ktoreho sa ulozia vsetky vygenerovane grafy.
"""
PRED_EXTRACT_HEALTH_ARGS = {'current_folder': True,
                            'csv_path': r'data\new_dataset.csv',
                            'csv_sep': ',',
                            'results_path': r'data\results.csv',
                            'imgs_path': r'new_dataset',
                            'img_dim': (128, 128),
                            'pretrained_model_path': r'models\model_3',
                            'parameter_space': {
                                'hidden_layer_sizes': [(50, 50, 50), (100,), (50, 20)],
                                # 'hidden_layer_sizes': [(50, 50, 50), (100,)],
                                'activation': ['tanh', 'relu'],
                                'solver': ['sgd', 'adam'],
                                'alpha': [0.0001, 0.05],
                                'learning_rate': ['constant', 'adaptive'],
                                # 'learning_rate_init': [0.00001, 0.000001],
                                # 'tol': [0.000001, 0.00001, 0.0001]
                            },
                            'minimized': False,
                            'minimized_path': r'data',
                            'graphs_path': r'graphs'
                            }


'''
-----segment_train.py-----
'''
SEGMENT_TRAIN_ARGS = {'current_folder': True,
                    #   'model_pretrained': r'models\model_segment_pretrained\leafSegmenter0005.h5',
                      'model_pretrained': r'models\segmentation_second_train\mask_rcnn_lemna_minor_0030.h5',
                      'command': 'splash',
                      'splash_img': r'lemna6.JPG',
                      'logs': r'models\model_segment_1',
                      'dataset_path': r'dataset_segmenter',
                      'annot_path': r'{dataset_path}\{subset}\lemna_minor_annotations.json',
                      'weight': 'a'
                      }

SEGMENT_PRED_ARGS = {'current_folder': True
                     }
