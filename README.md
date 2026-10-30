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
|       `-- 01_data_validation.ipynb      clasele de pret, encoding, split stratificat, model XGBoost de baza
|-- src/
|   `-- common.py                 cod partajat pentru Lab 2: make_classes, build_features, evaluate
|-- requirements.txt              pachetele necesare (pip)
|-- pyproject.toml / poetry.lock  aceleasi pachete, pentru cine foloseste Poetry
`-- .gitignore
```

Fiecare notebook incepe cu o celula de configurare care gaseste singura radacina proiectului
(folderul care contine `pyproject.toml`) si defineste `ROOT`, `DATA_RAW`, `DATA_PROCESSED`.
De aceea notebook-urile pot fi rulate din orice folder si de pe orice calculator, fara sa schimbi cai.
Cand adaugi un notebook nou, copiaza aceeasi celula la inceputul lui.

## Ordinea de rulare

Notebook-urile depind unele de altele prin fisierele din `data/processed/`:

1. `notebooks/lab1_eda/01_EDA.ipynb` - citeste `data/raw/AmesHousing.csv`, scrie `data/processed/df.csv`
2. `notebooks/lab1_eda/02_feature_engineering.ipynb` - citeste `df.csv`, scrie `data/processed/df_fe.csv`
3. `notebooks/lab1_eda/03_linear_regression.ipynb` - citeste `df_fe.csv`
4. `notebooks/lab2_xgboost/*.ipynb` - citesc `df_fe.csv` si folosesc `src/common.py`

Fisierele CSV nu sunt in Git (sunt mari si se regenereaza), deci dupa un `git clone` proaspat
rulezi notebook-urile in ordinea de mai sus o singura data.

## Instalare

```
git clone https://github.com/IuliaGalan/ames-housing-ML.git
cd ames-housing-ML
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt     (Windows)
# sau: source .venv/bin/activate && pip install -r requirements.txt   (Mac/Linux)
```

Pentru Lab 2 mai sunt necesare: `pip install xgboost plotly optuna anywidget`.

In VS Code: deschide folderul proiectului, iar in notebook alege kernel-ul `.venv`
(Select Kernel -> Python Environments -> .venv).

## Datele

`data/raw/AmesHousing.csv` este fisierul original al lui Dean De Cock (separat prin tab).
Nu este in Git. Il primesti de la un coleg sau il descarci de la
https://jse.amstat.org/v19n3/decock/AmesHousing.txt si il salvezi ca `data/raw/AmesHousing.csv`.

## Cum lucram

- Fiecare lucreaza pe branch-ul sau si deschide un Pull Request spre `main`.
- Inainte de a incepe: `git pull`. Nu edita acelasi notebook ca altcineva in acelasi timp
  (conflictele in notebook-uri sunt greu de rezolvat).
- Nu comite fisiere CSV, baze de date Optuna (`*.db`) sau modele salvate.
