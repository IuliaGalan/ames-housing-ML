
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, label_binarize


PRICE_BINS = [-np.inf, 100000, 130000, 180000, 250000, 400000, np.inf]
CLASSES = np.arange(6)


def make_classes(sale_price):
    prices = pd.Series(sale_price, copy=True)

    if prices.empty:
        raise ValueError("SalePrice nu poate fi gol.")

    try:
        numeric = pd.to_numeric(prices, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SalePrice trebuie sa contina doar valori numerice."
        ) from exc

    if numeric.isna().any() or not np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all():
        raise ValueError("SalePrice contine valori lipsa sau infinite.")

    if (numeric < 0).any():
        raise ValueError("SalePrice nu poate contine valori negative.")

    return pd.cut(
        numeric,
        bins=PRICE_BINS,
        labels=CLASSES,
        right=False
    ).astype(int)


def build_features(df, encoding="onehot"):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df trebuie sa fie un pandas DataFrame.")

    if df.empty or not df.columns.is_unique:
        raise ValueError(
            "Dataset-ul este gol sau are nume de coloane duplicate."
        )

    if "SalePrice" not in df.columns:
        raise ValueError("Lipseste coloana obligatorie SalePrice.")

    if encoding not in {"onehot", "ordinal"}:
        raise ValueError(
            "Encoding trebuie sa fie 'onehot' sau 'ordinal'."
        )

    y = make_classes(df["SalePrice"])

    X = df.drop(
        columns=[
            "SalePrice",
            "SalePrice_log",
            "PriceClass",
            "Order",
            "PID"
        ],
        errors="ignore"
    ).copy()

    if X.shape[1] == 0:
        raise ValueError("Nu au ramas caracteristici in X.")

    absence_cols = [
        "Alley",
        "Bsmt Qual",
        "Bsmt Cond",
        "Bsmt Exposure",
        "BsmtFin Type 1",
        "BsmtFin Type 2",
        "Fireplace Qu",
        "Garage Type",
        "Garage Finish",
        "Garage Qual",
        "Garage Cond",
        "Pool QC",
        "Fence",
        "Misc Feature"
    ]

    for col in absence_cols:
        if col in X.columns:
            X[col] = X[col].fillna("None")

    if "Mas Vnr Type" in X.columns:
        missing = X["Mas Vnr Type"].isna()

        if "Mas Vnr Area" in X.columns:
            zero_area = pd.to_numeric(
                X["Mas Vnr Area"],
                errors="coerce"
            ).eq(0)

            X.loc[
                missing & zero_area,
                "Mas Vnr Type"
            ] = "None"

        X["Mas Vnr Type"] = X["Mas Vnr Type"].fillna(
            "Unknown"
        )

    for col in ["MS SubClass", "Mo Sold"]:
        if col in X.columns:
            X[col] = (
                X[col]
                .astype("string")
                .fillna("Unknown")
                .astype(str)
            )

    ordinal_maps = {
        "Bsmt Exposure": {
            "None": 0,
            "No": 1,
            "Mn": 2,
            "Av": 3,
            "Gd": 4
        },
        "Garage Finish": {
            "None": 0,
            "Unf": 1,
            "RFn": 2,
            "Fin": 3
        }
    }

    for col, mapping in ordinal_maps.items():
        if col in X.columns:
            unknown = set(
                X[col].dropna().unique()
            ) - set(mapping)

            if unknown:
                raise ValueError(
                    f"Categorii necunoscute in {col}: "
                    f"{sorted(map(str, unknown))}"
                )

            X[col] = X[col].map(mapping)

    categorical_cols = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    quality_cols = [
        "Exter Qual",
        "Exter Cond",
        "Bsmt Qual",
        "Bsmt Cond",
        "Heating QC",
        "Kitchen Qual",
        "Fireplace Qu",
        "Garage Qual",
        "Garage Cond",
        "Pool QC"
    ]

    ordinal_cols = [
        col for col in quality_cols
        if col in categorical_cols
    ]

    nominal_cols = [
        col for col in categorical_cols
        if col not in ordinal_cols
    ]

    quality_order = [
        "None", "Po", "Fa", "TA", "Gd", "Ex"
    ]

    for col in ordinal_cols:
        unknown = set(
            X[col].dropna().unique()
        ) - set(quality_order)

        if unknown:
            raise ValueError(
                f"Categorii necunoscute in {col}: "
                f"{sorted(map(str, unknown))}"
            )

    if nominal_cols:
        X[nominal_cols] = X[nominal_cols].fillna(
            "Unknown"
        )

    numeric_cols = X.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_cols:
        values = X[numeric_cols].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                "Caracteristicile numerice contin NaN sau infinit."
            )

    transformers = []

    if ordinal_cols:
        ordinal_encoder = OrdinalEncoder(
            categories=[quality_order] * len(ordinal_cols),
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        transformers.append(
            ("quality", ordinal_encoder, ordinal_cols)
        )

    if nominal_cols:
        if encoding == "onehot":
            nominal_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        else:
            nominal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )

        transformers.append(
            ("nominal", nominal_encoder, nominal_cols)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False
    )

    return X, y, preprocessor


def evaluate(model, X_test, y_test, show_plots=True):
    if len(X_test) != len(y_test) or len(y_test) == 0:
        raise ValueError(
            "X_test si y_test trebuie sa aiba aceeasi lungime nenula."
        )

    if not hasattr(model, "predict_proba") or not hasattr(
        model, "classes_"
    ):
        raise ValueError(
            "Modelul trebuie sa aiba predict_proba() si classes_."
        )

    if not np.array_equal(
        np.asarray(model.classes_), CLASSES
    ):
        raise ValueError(
            f"Clasele modelului trebuie sa fie 0..5; "
            f"primite: {model.classes_}"
        )

    y_true = np.asarray(y_test)

    if not np.isin(y_true, CLASSES).all():
        raise ValueError(
            "y_test contine clase in afara intervalului 0..5."
        )

    y_pred = np.asarray(model.predict(X_test))

    y_proba = np.asarray(
        model.predict_proba(X_test),
        dtype=float
    )

    if y_pred.shape != (len(y_true),) or not np.isin(
        y_pred, CLASSES
    ).all():
        raise ValueError(
            "Predictiile au dimensiuni sau clase invalide."
        )

    if y_proba.shape != (len(y_true), len(CLASSES)):
        raise ValueError(
            "Matricea de probabilitati trebuie sa aiba 6 coloane."
        )

    if (
        not np.isfinite(y_proba).all()
        or (y_proba < 0).any()
        or (y_proba > 1).any()
    ):
        raise ValueError("Probabilitatile sunt invalide.")

    if not np.allclose(
        y_proba.sum(axis=1), 1.0, atol=1e-5
    ):
        raise ValueError(
            "Probabilitatile pe fiecare rand nu insumeaza 1."
        )

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            labels=CLASSES,
            average="macro",
            zero_division=0
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            labels=CLASSES,
            average="macro",
            zero_division=0
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            labels=CLASSES,
            average="macro",
            zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true,
            y_pred,
            labels=CLASSES,
            average="weighted",
            zero_division=0
        )
    }

    print("Rezultatele modelului:")

    for name, value in results.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=CLASSES,
            zero_division=0
        )
    )

    if show_plots:
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            labels=CLASSES,
            cmap="Blues"
        )

        plt.title("Confusion Matrix")
        plt.show()

        y_binary = label_binarize(
            y_true,
            classes=CLASSES
        )

        plt.figure(figsize=(8, 6))

        for i in range(len(CLASSES)):
            if np.unique(y_binary[:, i]).size < 2:
                continue

            fpr, tpr, _ = roc_curve(
                y_binary[:, i],
                y_proba[:, i]
            )

            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr,
                tpr,
                label=f"Class {i} (AUC={roc_auc:.2f})"
            )

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 6))

        for i in range(len(CLASSES)):
            if y_binary[:, i].sum() == 0:
                continue

            precision, recall, _ = precision_recall_curve(
                y_binary[:, i],
                y_proba[:, i]
            )

            ap = average_precision_score(
                y_binary[:, i],
                y_proba[:, i]
            )

            plt.plot(
                recall,
                precision,
                label=f"Class {i} (AP={ap:.2f})"
            )

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves")
        plt.legend()
        plt.show()

    return results
