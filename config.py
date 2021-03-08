"""
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
Argumenty pre skript na predikciu zdravia rastliny lemna minor.
current_folder: Nastavenie, ci cesta k suborom pocina aktualnym umiestnenim suboru
                Boolean hodnota - ak je True, aktualna cesta pracovneho
                prostredia bude nastavena na umiestnenie daneho suboru
csv_path: Nastavuje cestu k csv suboru
csv_sep: Oddelovac pouzity v csv subore
imgs_path: Cesta k datasetu pouzitom na trenovanie/testovanie/validaciu
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
PRED_HEALTH_ARGS = {'current_folder': True,
                    'csv_path': r'data\new_dataset.csv',
                    'csv_sep': ',',
                    'imgs_path': r'new_dataset',
                    'pretrained_model': True,
                    'pretrained_model_path': r'models\model_2',
                    'model_path': r'models\model_2',
                    'heatmaps': True,
                    'heatmaps_path': r'heatmaps',
                    'graphs_path': r'graphs'
                    }
