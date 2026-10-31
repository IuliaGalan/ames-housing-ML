# ames-housing-ML

Proiect de echipa pentru cursul de Machine Learning, pe setul de date Ames Housing
(2930 de case vandute in Ames, Iowa, 2006-2010, 82 de coloane, tinta: SalePrice).

- Lab 1: analiza exploratorie, relatii intre features, feature engineering, regresie liniara (bonus)
- Lab 2: clasificare multiclasa a pretului (6 clase) cu XGBoost + optimizare cu Optuna
- Lab 3: clustering (urmeaza)

## Structura proiectului

```
ames-housing-ML/
|-- data/
|   |-- raw/                      AmesHousing.csv - fisierul original (NU e in Git, vezi mai jos)
|   |-- processed/                df.csv, df_fe.csv - generate de notebook-uri (NU sunt in Git)
|   `-- AmesExplanations.txt      descrierea fiecarei coloane din dataset
|-- notebooks/
|   |-- lab1_eda/
|   |   |-- 01_EDA.ipynb                  curatarea datelor, valori lipsa, distributii, outlieri  -> scrie data/processed/df.csv
|   |   |-- 02_feature_engineering.ipynb  heatmap, scatter, violin, groupby, features noi        -> scrie data/processed/df_fe.csv
|   |   `-- 03_linear_regression.ipynb    perechi liniare, spurious correlations, regresie liniara (bonus)
|   `-- lab2_xgboost/
|       |-- 01_data_validation.ipynb      Partea A: validarea common.py + experimentele de preprocesare, split-uri,
|       |                                 k-fold, curbe de invatare, dezechilibru de clase
|       `-- 02_xgboost_model.ipynb        Partea B: XGBoost, arbori, feature importance, metrici, praguri,
|                                         cu/fara ponderi, hiperparametri modificati manual
|-- src/
|   |-- common.py                 cod partajat pentru Lab 2: make_classes, build_features, evaluate
|   `-- metrics_eval.py           functiile Partii B (fit_xgb, arbori, importance, praguri, sweep-uri, full_report)
|-- figures_B/                    figurile Partii B (PNG, salvate de notebook-ul 02_xgboost_model)
|-- requirements.txt              pachetele necesare (pip)
|-- pyproject.toml / poetry.lock  aceleasi pachete, pentru cine foloseste Poetry
`-- .gitignore
```

Fiecare notebook incepe cu o celula de configurare care gaseste singura radacina proiectului
(folderul care contine `pyproject.toml`) si defineste `ROOT`, `DATA_RAW`, `DATA_PROCESSED`.
De aceea notebook-urile pot fi rulate din orice folder si de pe orice calculator, fara sa schimbi cai.
Cand adaugi un notebook nou, copiaza aceeasi celula la inceputul lui.
Notebook-urile din Lab 2 adauga si `src/` la `sys.path`, ca sa poata face `import common` si `import metrics_eval`.

## Ordinea de rulare

Notebook-urile depind unele de altele prin fisierele din `data/processed/`:

1. `notebooks/lab1_eda/01_EDA.ipynb` - citeste `data/raw/AmesHousing.csv`, scrie `data/processed/df.csv`
2. `notebooks/lab1_eda/02_feature_engineering.ipynb` - citeste `df.csv`, scrie `data/processed/df_fe.csv`
3. `notebooks/lab1_eda/03_linear_regression.ipynb` - citeste `df_fe.csv` (bonus Lab 1, nu e necesar pentru Lab 2)
4. `notebooks/lab2_xgboost/01_data_validation.ipynb` (Partea A) - citeste `df_fe.csv` si foloseste `src/common.py`
5. `notebooks/lab2_xgboost/02_xgboost_model.ipynb` (Partea B) - citeste `df_fe.csv` si foloseste `src/common.py` si `src/metrics_eval.py`

Doar primii doi pasi sunt obligatorii, in aceasta ordine: fara `df_fe.csv`, nimic din Lab 2 nu merge.
Notebook-urile 3, 4 si 5 nu depind unul de altul, deci pot fi rulate in orice ordine dupa ce exista `df_fe.csv`.
In interiorul unui notebook, celulele se ruleaza de sus in jos (sau "Restart & Run All").
Fisierele `.py` din `src/` nu se ruleaza direct, doar se importa din notebook-uri.

Fisierele CSV nu sunt in Git (sunt mari si se regenereaza), deci dupa un `git clone` proaspat
rulezi notebook-urile 1 si 2 o singura data.

## Instalare

```
git clone https://github.com/IuliaGalan/ames-housing-ML.git
cd ames-housing-ML
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt     (Windows)
# sau: source .venv/bin/activate && pip install -r requirements.txt   (Mac/Linux)
```

`requirements.txt` contine toate pachetele pentru Lab 1 si Lab 2, inclusiv `xgboost`, `optuna`, `plotly`,
`anywidget` si `graphviz`. Nu mai trebuie instalat nimic separat prin pip.
Se recomanda Python 3.13 (cel din `pyproject.toml`).

In VS Code: deschide folderul proiectului, iar in notebook alege kernel-ul `.venv`
(Select Kernel -> Python Environments -> .venv).

### Graphviz (pentru desenarea arborilor XGBoost)

Pachetul Python `graphviz` doar apeleaza programul Graphviz, care trebuie instalat separat in sistem:

- Windows: `winget install Graphviz.Graphviz` (sau installer-ul de la https://graphviz.org/download/,
  cu optiunea "Add Graphviz to the system PATH"), apoi redeschide terminalul / VS Code
- Mac: `brew install graphviz`
- Linux: `sudo apt install graphviz`

Verificare: `dot -V` trebuie sa afiseze versiunea. Fara Graphviz, `02_xgboost_model.ipynb` nu da eroare,
dar in locul desenului afiseaza arborele ca text.

## Datele

`data/raw/AmesHousing.csv` este fisierul original al lui Dean De Cock (separat prin tab).
Nu este in Git. Il primesti de la un coleg sau il descarci de la
https://jse.amstat.org/v19n3/decock/AmesHousing.txt si il salvezi ca `data/raw/AmesHousing.csv`.

## Lab 2: configuratia comuna (strategia recomandata de Partea A)

Partile B si C folosesc aceeasi configuratie, ca rezultatele sa fie comparabile:

| Componenta            | Decizie                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| Dataset               | `df_fe.csv` (cu features din Lab 1)                                                               |
| Encoding              | one-hot (`build_features(..., encoding="onehot")`)                                                |
| Outlieri              | se pastreaza                                                                                      |
| Split                 | 80/20, stratificat, `random_state=42`                                                             |
| Cross-validation      | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, doar pe train                       |
| Metrica principala    | macro-F1                                                                                          |
| Dezechilibru de clase | baseline fara ponderi; `sample_weight` balanced ramane optiune cu/fara                            |
| Setul de test         | doar pentru evaluarea finala, nu pentru alegerea hiperparametrilor, a ponderilor sau a pragurilor |

Pentru a rula Partea B pe modelul final de la Partea C:

Intr-o celula de notebook (variabilele `X_train`, `y_train`, `X_test`, `y_test`, `preprocessor` si
`best_params` exista deja in notebook, de aici fragmentul nu e Python de sine statator):

```text
import metrics_eval as B
model = B.fit_xgb(X_train, y_train, preprocessor, params=best_params)
B.full_report(model, X_train, y_train, X_test, y_test, save_dir="figures_B")
```

## Cum lucram

- Fiecare lucreaza pe branch-ul sau si deschide un Pull Request spre `main`.
- Inainte de a incepe: `git pull`. Nu edita acelasi notebook ca altcineva in acelasi timp
  (conflictele in notebook-uri sunt greu de rezolvat).
- Nu comite fisiere CSV, baze de date Optuna (`*.db`) sau modele salvate.