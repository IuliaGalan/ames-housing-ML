
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Instrumente sklearn pentru preprocesare si evaluarea clasificarii
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
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

# Limitele intervalelor de pret si identificatorii celor sase clase
PRICE_BINS = [-np.inf, 100000, 130000, 180000, 250000, 400000, np.inf]
CLASSES = np.arange(6)

# Encoder pentru varianta nativa XGBoost: pastreaza categoriile in format pandas
class NativeCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Keeps nominal columns as pandas category, learning levels on fit data only."""

    # Retinem coloanele si categoriile intalnite la antrenare
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NativeCategoricalEncoder asteapta un DataFrame pandas.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.categories_ = {
            col: pd.Index(X[col].dropna().unique()) for col in self.feature_names_in_
        }
        return self

    # Aplicam aceleasi categorii si pe date noi; cele necunoscute devin valori lipsa
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X = X.loc[:, list(self.feature_names_in_)].copy()
        for col in self.feature_names_in_:
            X[col] = pd.Categorical(X[col], categories=self.categories_[col])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_ if input_features is None else input_features, dtype=object)

# Transforma pretul de vanzare intr-una dintre cele sase clase
def make_classes(sale_price):
    prices = pd.Series(sale_price, copy=True)

    # Verificam ca preturile sa fie prezente si sa aiba valori numerice valide
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

    # Intervalele sunt inchise la stanga: de exemplu, 100000 intra in clasa 1
    return pd.cut(
        numeric,
        bins=PRICE_BINS,
        labels=CLASSES,
        right=False
    ).astype(int)

# Pregateste caracteristicile, etichetele si transformatorul folosit de model
def build_features(df, encoding="onehot"):
    # Verificam formatul datelor si optiunea de encoding primita
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df trebuie sa fie un pandas DataFrame.")

    if df.empty or not df.columns.is_unique:
        raise ValueError(
            "Dataset-ul este gol sau are nume de coloane duplicate."
        )

    if "SalePrice" not in df.columns:
        raise ValueError("Lipseste coloana obligatorie SalePrice.")

    if encoding not in {"onehot", "ordinal", "native"}:
        raise ValueError(
            "Encoding trebuie sa fie 'onehot', 'ordinal' sau 'native'."
        )

    # Variabila tinta este clasa de pret, nu pretul numeric
    y = make_classes(df["SalePrice"])

    # Excludem tinta si coloanele care ar putea dezvalui pretul sau identifica randul
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

    # La aceste caracteristici, lipsa valorii poate insemna lipsa dotarii
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

    # Pentru furnir, distingem lipsa efectiva de o valoare necunoscuta
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

    # Aceste coloane sunt coduri/categorii, chiar daca initial par numerice
    for col in ["MS SubClass", "Mo Sold"]:
        if col in X.columns:
            X[col] = (
                X[col]
                .astype("string")
                .fillna("Unknown")
                .astype(str)
            )

    # Unele categorii cu ordine clara sunt transformate direct in valori numerice
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
    # Nu acceptam valori din afara ordinii stabilite pentru aceste coloane
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

    # Identificam coloanele categorice ramase dupa transformarile anterioare
    categorical_cols = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    # Calitatea are o ordine naturala, spre deosebire de categoriile nominale
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

    # Ordinea folosita pentru nivelurile de calitate, de la lipsa la excelent
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

    # Completam valorile lipsa din coloanele nominale
    if nominal_cols:
        X[nominal_cols] = X[nominal_cols].fillna(
            "Unknown"
        )

    # Verificam sa nu existe NaN sau infinit in caracteristicile numerice
    numeric_cols = X.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_cols:
        values = X[numeric_cols].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                "Caracteristicile numerice contin NaN sau infinit."
            )

    # Construim separat transformarile pentru coloanele ordinale si nominale
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

    # Comparam trei moduri de tratare a categoriilor nominale
    if nominal_cols:
        if encoding == "onehot":
            nominal_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
            transformers.append(("nominal", nominal_encoder, nominal_cols))
        elif encoding == "ordinal":
            nominal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
            transformers.append(("nominal", nominal_encoder, nominal_cols))
        else:
            # Learn category levels only on the current training fold; unseen
            # validation/test categories are converted to missing values.
            transformers.append(("nominal", NativeCategoricalEncoder(), nominal_cols))

    # ColumnTransformer aplica encoderul potrivit fiecarei grupe de coloane
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    # Intoarcem datele necodificate, clasele si preprocesorul pentru pipeline
    return X, y, preprocessor

# Evalueaza modelul pe un set de test si poate afisa graficele aferente
def evaluate(model, X_test, y_test, show_plots=True):
    # Verificam datele de test si compatibilitatea modelului cu cele sase clase
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
    if y_true.ndim != 1:
        raise ValueError("y_test trebuie sa fie un vector unidimensional.")

    if not np.isin(y_true, CLASSES).all():
        raise ValueError(
            "y_test contine clase in afara intervalului 0..5."
        )

    # Obtinem atat clasa prezisa, cat si probabilitatile fiecarei clase
    y_pred = np.asarray(model.predict(X_test))

    y_proba = np.asarray(
        model.predict_proba(X_test),
        dtype=float
    )

    # Controlam forma predictiilor si validitatea probabilitatilor
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

    # Pentru ROC si PR tratam fiecare clasa separat fata de celelalte
    y_binary = label_binarize(y_true, classes=CLASSES)
    roc_auc_per_class = {}
    average_precision_per_class = {}
    for i, class_id in enumerate(CLASSES):
        # AUC is undefined if a class has no positive or no negative examples.
        if np.unique(y_binary[:, i]).size == 2:
            roc_auc_per_class[int(class_id)] = float(roc_auc_score(y_binary[:, i], y_proba[:, i]))
        else:
            roc_auc_per_class[int(class_id)] = float("nan")
        average_precision_per_class[int(class_id)] = float(
            average_precision_score(y_binary[:, i], y_proba[:, i])
        ) if y_binary[:, i].sum() > 0 else float("nan")

     # Calculam matricea de confuzie si mediile scorurilor pe clasele valide
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    defined_auc = [v for v in roc_auc_per_class.values() if np.isfinite(v)]
    defined_ap = [v for v in average_precision_per_class.values() if np.isfinite(v)]
    macro_auc = float(np.mean(defined_auc)) if defined_auc else float("nan")
    macro_ap = float(np.mean(defined_ap)) if defined_ap else float("nan")
    # Pastram metricile si predictiile intr-un dictionar reutilizabil
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
        ),
        "roc_auc_macro_ovr": macro_auc,
        "average_precision_macro": macro_ap,
        "roc_auc_per_class": roc_auc_per_class,
        "average_precision_per_class": average_precision_per_class,
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_true, y_pred, labels=CLASSES, zero_division=0, output_dict=True
        ),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    # Afisam scorurile numerice si raportul de clasificare
    print("Rezultatele modelului:")

    for name, value in results.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            print(f"{name}: {value:.4f}")
    print("ROC-AUC per clasa:", {k: round(v, 4) for k, v in roc_auc_per_class.items()})
    print("Average Precision per clasa:", {k: round(v, 4) for k, v in average_precision_per_class.items()})
    print("Matricea de confuzie:\n", cm)

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=CLASSES,
            zero_division=0
        )
    )

    # Graficele se afiseaza doar daca show_plots este True
    if show_plots:
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            labels=CLASSES,
            cmap="Blues"
        )

        plt.title("Confusion Matrix")
        plt.show()

    # Pregatim etichetele binare pentru curbele pe fiecare clasa
        y_binary = label_binarize(
            y_true,
            classes=CLASSES
        )

    # Curbele ROC arata raportul dintre true positive si false positive rate
        plt.figure(figsize=(8, 6))

        for i in range(len(CLASSES)):
            if np.unique(y_binary[:, i]).size < 2:
                continue

            fpr, tpr, _ = roc_curve(
                y_binary[:, i],
                y_proba[:, i]
            )

            roc_auc = roc_auc_per_class[int(i)]

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

    # Curbele Precision-Recall sunt utile si pentru clasele mai putin frecvente
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
