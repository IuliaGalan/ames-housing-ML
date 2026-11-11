import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from common import build_features


# Aplica log1p pe coloanele numerice continue, nenegative si asimetrice (skew > prag)
class LogSkewed(BaseEstimator, TransformerMixin):
    """Logaritmeaza (log1p) doar coloanele unde are sens; alegerea se face pe datele de fit."""

    def __init__(self, skew_threshold=0.75, min_unique=10):
        self.skew_threshold = skew_threshold
        self.min_unique = min_unique

    # Alegem coloanele: numerice, nenegative, cu multe valori distincte si asimetrie pozitiva mare
    def fit(self, X, y=None):
        num = X.select_dtypes(include="number")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.log_cols_ = [
            c for c in num.columns
            if num[c].min() >= 0
            and num[c].nunique() > self.min_unique
            and num[c].skew() > self.skew_threshold
        ]
        return self

    # Aplicam log1p pe coloanele alese la fit (si pe date noi)
    def transform(self, X):
        X = X.copy()
        for c in self.log_cols_:
            X[c] = np.log1p(X[c])
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


# Preprocesare comuna pentru Lab 3: encoding one-hot, logaritmare optionala, scalare
def preprocess(df, scaling="minmax", log=False):
    """
    Intoarce (X_scaled, y, pipeline).

    scaling: "minmax" (MinMaxScaler) sau "standard" (StandardScaler = normalizare, medie 0 si abatere 1)
    log:     daca e True, coloanele numerice asimetrice sunt logaritmate (log1p) inainte de encoding si scalare

    X_scaled - DataFrame cu datele codificate si scalate (coloanele: numele features)
    y        - clasa de pret (0-5), nu intra in X_scaled
    pipeline - Pipeline-ul deja antrenat; pentru case noi se foloseste pipeline.transform(X_nou),
               unde X_nou trece intai prin build_features (ca sa primeasca aceleasi completari de valori lipsa)
    """
    if scaling not in {"minmax", "standard"}:
        raise ValueError("scaling trebuie sa fie 'minmax' sau 'standard'.")

    X, y, encoder = build_features(df, encoding="onehot")
    scaler = MinMaxScaler() if scaling == "minmax" else StandardScaler()

    steps = []
    if log:
        steps.append(("log", LogSkewed()))
    steps += [("encode", encoder), ("scale", scaler)]

    pipeline = Pipeline(steps)
    values = pipeline.fit_transform(X)
    X_scaled = pd.DataFrame(values, columns=pipeline.get_feature_names_out(), index=X.index)
    return X_scaled, y, pipeline


# Profilul clusterelor: numarul de case si media features-urilor numerice, per cluster
def cluster_profile(df, labels):
    """
    df     - dataset-ul curatat (valori originale, nescalate), ca mediile sa fie interpretabile
    labels - etichetele clusterelor, in aceeasi ordine ca randurile din df
    Intoarce un DataFrame cu un rand per cluster: n_houses + media fiecarei coloane numerice.
    """
    if len(labels) != len(df):
        raise ValueError("labels trebuie sa aiba cate o valoare pentru fiecare rand din df.")

    # MS SubClass si Mo Sold sunt coduri/categorii, deci media lor nu are sens
    data = df.drop(columns=["Order", "PID", "MS SubClass", "Mo Sold"], errors="ignore").select_dtypes(include="number").copy()
    data["cluster"] = np.asarray(labels)

    profile = data.groupby("cluster").mean()
    profile.insert(0, "n_houses", data.groupby("cluster").size())
    return profile
