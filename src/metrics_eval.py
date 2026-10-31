"""
Partea B - modelul XGBoost, arborii, metricile, interpretarea.

Toate functiile de analiza primesc un MODEL (un sklearn Pipeline
[preprocessor -> XGBClassifier], ca cel returnat de fit_xgb) si datele de
evaluare, ca sa poata fi rerulate identic pe modelul final din Partea C:

    model = fit_xgb(X_train, y_train, preprocessor, params=best_params)
    full_report(model, X_train, y_train, X_test, y_test)

Depinde de common.py (make_classes, build_features, evaluate, CLASSES).
"""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from common import CLASSES, build_features, evaluate

SEED = 42
CLASS_NAMES = ["<100k", "100-130k", "130-180k", "180-250k", "250-400k", ">=400k"]

# feature-urile create in L1 (feature_engineering.ipynb) - evidentiate in grafice
FE_FEATURES = {
    "TotalSF", "HouseAge", "YearsSinceRemod", "Remodeled", "TotalBath",
    "TotalPorchSF", "HasPool", "HasFireplace", "Has2ndFlr", "HasBsmt",
    "QualxLivArea", "LivAreaPerRoom", "Has_Garage",
}

# hiperparametri de baza; n_estimators mare + early stopping decide numarul real
BASE_PARAMS = dict(
    objective="multi:softprob",
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0.0,
    reg_lambda=1.0,
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=SEED,
    n_jobs=-1,
)


# ----------------------------------------------------------------------------
# utilitare
# ----------------------------------------------------------------------------
def _finish(fig, name, save_dir=None):
    """Afiseaza figura si, optional, o salveaza ca PNG."""
    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.show()


def get_parts(model):
    """Intoarce (preprocessor, XGBClassifier) dintr-un Pipeline sau dintr-un clasificator simplu."""
    if isinstance(model, Pipeline):
        return model.named_steps["preprocessor"], model.named_steps["classifier"]
    return None, model


def best_n_rounds(model):
    """Numarul de runde de boosting folosite efectiv (best_iteration + 1 daca s-a folosit early stopping)."""
    _, clf = get_parts(model)
    bi = getattr(clf, "best_iteration", None)
    return int(bi) + 1 if bi is not None else int(clf.get_booster().num_boosted_rounds())


def get_booster(model):
    """Booster-ul taiat la runda optima (fara arborii adaugati dupa best_iteration in timpul patience)."""
    _, clf = get_parts(model)
    return clf.get_booster()[: best_n_rounds(model)]


def class_weights(y, use=True):
    return compute_sample_weight("balanced", y) if use else None


def _params_for_frame(params, frame):
    """Activeaza suportul categorical nativ cand frame contine dtype category."""
    out = dict(params)
    if isinstance(frame, pd.DataFrame) and any(
        isinstance(dtype, pd.CategoricalDtype) for dtype in frame.dtypes
    ):
        out["enable_categorical"] = True
    return out


# ----------------------------------------------------------------------------
# date, split, antrenare
# ----------------------------------------------------------------------------
def load_data(path="df_fe.csv", encoding="onehot"):
    """Citeste datasetul curatat si aplica build_features din common.py (aceeasi functie ca A si C)."""
    df = pd.read_csv(path)
    X, y, preprocessor = build_features(df, encoding=encoding)
    preprocessor.set_output(transform="pandas")  # pastreaza numele feature-urilor (pentru arbori / importance)
    return X, y, preprocessor


def make_split(X, y, test_size=0.2, stratify=True, seed=SEED):
    return train_test_split(
        X, y, test_size=test_size, stratify=y if stratify else None, random_state=seed
    )


def fit_xgb(
    X_train,
    y_train,
    preprocessor,
    params=None,
    early_stopping=True,
    early_stopping_rounds=30,
    val_size=0.15,
    use_class_weight=True,
    seed=SEED,
    refit_full=True,
):
    """Antreneaza XGBoost fara leakage si returneaza un Pipeline fitat.

    Cu early_stopping=True:
      1) imparte numai TRAIN in fit/validare;
      2) invata preprocesorul exclusiv pe partea fit;
      3) selecteaza numarul de boosting rounds pe validare;
      4) implicit, refit-uieste preprocesorul si modelul pe TOT X_train,
         folosind numarul de runde selectat. Testul nu este accesat.

    Curba si best_iteration din etapa de selectie sunt pastrate ca metadata
    pentru plot_training_curve(). Cu early_stopping=False, preprocesorul este
    fitat pe X_train inainte de transformare (fara NotFittedError).
    """
    p = {**BASE_PARAMS, **(params or {})}
    prep = clone(preprocessor).set_output(transform="pandas")

    if early_stopping:
        if not 0 < val_size < 1:
            raise ValueError("val_size trebuie sa fie intre 0 si 1.")
        if not isinstance(early_stopping_rounds, int) or early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds trebuie sa fie un intreg pozitiv.")
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train, test_size=val_size, stratify=y_train, random_state=seed
        )
        prep_select = clone(prep).set_output(transform="pandas")
        A_fit = prep_select.fit_transform(X_fit)
        A_val = prep_select.transform(X_val)
        selector_params = dict(p)
        if any(isinstance(dtype, pd.CategoricalDtype) for dtype in A_fit.dtypes):
            selector_params["enable_categorical"] = True
        selector = XGBClassifier(**selector_params, early_stopping_rounds=early_stopping_rounds)
        selector.fit(
            A_fit,
            y_fit,
            sample_weight=class_weights(y_fit, use_class_weight),
            # Early stopping uses the natural validation distribution. We do
            # not derive evaluation weights from validation labels.
            eval_set=[(A_fit, y_fit), (A_val, y_val)],
            verbose=False,
        )
        selected_rounds = int(selector.best_iteration) + 1
        history = selector.evals_result()
        selected_iteration = int(selector.best_iteration)

        if refit_full:
            prep.fit(X_train)
            A_train = prep.transform(X_train)
            final_params = {**p, "n_estimators": selected_rounds}
            if any(isinstance(dtype, pd.CategoricalDtype) for dtype in A_train.dtypes):
                final_params["enable_categorical"] = True
            clf = XGBClassifier(**final_params)
            clf.fit(
                A_train,
                y_train,
                sample_weight=class_weights(y_train, use_class_weight),
                verbose=False,
            )
        else:
            prep = prep_select
            clf = selector
    else:
        A = prep.fit_transform(X_train)
        fit_params = dict(p)
        if any(isinstance(dtype, pd.CategoricalDtype) for dtype in A.dtypes):
            fit_params["enable_categorical"] = True
        clf = XGBClassifier(**fit_params)
        clf.fit(A, y_train, sample_weight=class_weights(y_train, use_class_weight), verbose=False)
        history = None
        selected_iteration = None

    fitted = Pipeline([("preprocessor", prep), ("classifier", clf)])
    fitted._early_stopping_history = history
    fitted._early_stopping_best_iteration = selected_iteration
    fitted._early_stopping_selected_rounds = int(selected_iteration) + 1 if selected_iteration is not None else None
    fitted._refit_full = bool(early_stopping and refit_full)
    fitted._use_class_weight = bool(use_class_weight)
    return fitted


# ----------------------------------------------------------------------------
# 1. curba de invatare a modelului (train vs validare pe runde)
# ----------------------------------------------------------------------------
def plot_training_curve(model, save_dir=None):
    _, clf = get_parts(model)
    ev = getattr(model, "_early_stopping_history", None)
    if ev is None:
        try:
            ev = clf.evals_result()
        except (AttributeError, xgb.core.XGBoostError):
            print("Curba de boosting nu este disponibila: modelul nu a pastrat evals_result().")
            return None
    keys = list(ev)
    metric = list(ev[keys[0]])[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ev[keys[0]][metric], label="train")
    if len(keys) > 1:
        ax.plot(ev[keys[1]][metric], label="validare")
    bi = getattr(model, "_early_stopping_best_iteration", None)
    if bi is None:
        bi = getattr(clf, "best_iteration", None)
    if bi is not None:
        ax.axvline(bi, color="red", ls="--", label=f"best_iteration = {bi}")
    ax.set_xlabel("runda de boosting")
    ax.set_ylabel(metric)
    ax.set_title("Curba de invatare XGBoost (early stopping)")
    ax.legend()
    _finish(fig, "B1_training_curve", save_dir)


# ----------------------------------------------------------------------------
# 2. arbori: primul si "cel mai bun"
# ----------------------------------------------------------------------------
def tree_index(model, round_idx, class_idx):
    """Indexul arborelui in booster: in multiclass fiecare runda are cate un arbore per clasa."""
    _, clf = get_parts(model)
    return int(round_idx) * int(clf.n_classes_) + int(class_idx)


def first_tree_idx(model, class_idx=0):
    return tree_index(model, 0, class_idx)


def best_tree_idx(model, class_idx=0):
    """Arborele clasei `class_idx` din runda best_iteration (cea aleasa de early stopping)."""
    return tree_index(model, best_n_rounds(model) - 1, class_idx)


def tree_summary(model, indices):
    """Tabel cu marimea / castigul total al arborilor ceruti (indecsi absoluti in booster)."""
    df = get_parts(model)[1].get_booster().trees_to_dataframe()
    rows = []
    for i in indices:
        t = df[df["Tree"] == i]
        splits, leaves = t[t["Feature"] != "Leaf"], t[t["Feature"] == "Leaf"]
        rows.append(
            {
                "tree": i,
                "noduri": len(t),
                "frunze": len(leaves),
                "castig_total": splits["Gain"].sum(),
                "feature_radacina": t.loc[t["Node"] == 0, "Feature"].iloc[0],
            }
        )
    return pd.DataFrame(rows).set_index("tree")


def plot_tree_idx(model, tree_idx, title="", figsize=(26, 12), save_dir=None, name=None):
    """
    Deseneaza un arbore cu xgboost.plot_tree (necesita Graphviz instalat in sistem, comanda `dot`).
    Daca Graphviz lipseste, afiseaza dump-ul text al arborelui.
    """
    _, clf = get_parts(model)
    booster = clf.get_booster()
    try:
        fig, ax = plt.subplots(figsize=figsize)
        xgb.plot_tree(booster, tree_idx=tree_idx, ax=ax)
        ax.set_title(title or f"Arborele #{tree_idx}", fontsize=16)
        _finish(fig, name or f"B2_tree_{tree_idx}", save_dir)
    except Exception as exc:  # graphviz lipsa
        plt.close("all")
        warnings.warn(f"plot_tree a esuat ({exc}); afisez dump text. Instaleaza Graphviz (binarul `dot`).")
        print(title)
        print(booster.get_dump()[tree_idx])


# ----------------------------------------------------------------------------
# 3. feature importance (gain / split)
# ----------------------------------------------------------------------------
def _nominal_columns(model):
    prep, _ = get_parts(model)
    if prep is None:
        return []
    for name, _, cols in prep.transformers_:
        if name == "nominal":
            return list(cols)
    return []


def _origin_map(model, names):
    """Mapeaza fiecare nume one-hot 'Col_val' la coloana originala 'Col' (potrivire pe cel mai lung prefix)."""
    nominal = sorted(_nominal_columns(model), key=len, reverse=True)
    out = {}
    for n in names:
        out[n] = next((c for c in nominal if n.startswith(c + "_")), n)
    return out


def importance_table(model, aggregate=False):
    """
    Importanta pe tot ansamblul de arbori pana la best_iteration:
      gain        = castigul mediu pe split-urile care folosesc feature-ul
      total_gain  = castigul total
      split       = de cate ori e folosit ca split ('weight' in xgboost)
      cover       = nr. mediu de exemple afectate de split
    aggregate=True aduna coloanele one-hot inapoi la feature-ul original (total_gain, split se aduna).
    """
    booster = get_booster(model)
    names = list(booster.feature_names)
    d = {t: booster.get_score(importance_type=t) for t in ["gain", "total_gain", "weight", "cover"]}
    df = pd.DataFrame(index=names)
    df["gain"] = pd.Series(d["gain"])
    df["total_gain"] = pd.Series(d["total_gain"])
    df["split"] = pd.Series(d["weight"])
    df["cover"] = pd.Series(d["cover"])
    df = df.fillna(0.0).copy()

    if aggregate:
        origin = _origin_map(model, names)
        df["origin"] = pd.Series(origin)
        g = df.groupby("origin")
        agg = pd.DataFrame({"total_gain": g["total_gain"].sum(), "split": g["split"].sum()})
        agg["gain"] = (agg["total_gain"] / agg["split"].replace(0, np.nan)).fillna(0.0)
        df = agg[["gain", "total_gain", "split"]].copy()

    df["rank_gain"] = df["gain"].rank(ascending=False, method="min").astype(int)
    df["rank_split"] = df["split"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("gain", ascending=False)


def plot_importance(model, top=20, aggregate=True, save_dir=None):
    """Doua panouri: top feature-uri dupa gain si dupa split. Feature-urile create in L1 sunt colorate diferit."""
    tab = importance_table(model, aggregate=aggregate)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, col, title in zip(axes, ["gain", "split"], ["Importance - GAIN (mediu / split)", "Importance - SPLIT (nr. de utilizari)"]):
        t = tab[col].sort_values(ascending=False).head(top).iloc[::-1]
        colors = ["tab:orange" if n in FE_FEATURES else "tab:blue" for n in t.index]
        ax.barh(t.index, t.values, color=colors)
        ax.set_title(title)
    axes[0].text(
        0.98, 0.02, "portocaliu = feature creat in L1", transform=axes[0].transAxes,
        ha="right", color="tab:orange", fontsize=10,
    )
    _finish(fig, "B3_feature_importance", save_dir)

    rho = spearmanr(tab["gain"], tab["split"]).correlation
    print(f"Corelatie Spearman intre rangul dupa gain si dupa split: {rho:.3f}")
    return tab


def plot_permutation_importance(model, X_train, y_train, top=12, n_repeats=2, n_splits=3, save_dir=None):
    """Permutation importance pe folduri de validare CV, exclusiv in TRAIN.

    Pentru eficienta, in fiecare fold permutam cele mai importante `top`
    caracteristici brute identificate dupa gain pe foldul de antrenare.
    Fiecare fold antreneaza un model nou, iar scorul este masurat numai pe
    observatiile de validare ale acelui fold. Setul de test
    nu este utilizat in aceasta analiza.
    """
    if len(X_train) != len(y_train) or len(y_train) == 0:
        raise ValueError("X_train si y_train trebuie sa aiba aceeasi lungime nenula.")
    if n_splits < 2:
        raise ValueError("n_splits trebuie sa fie cel putin 2.")

    prep, fitted_clf = get_parts(model)
    if prep is None:
        raise ValueError("Permutation importance CV necesita un Pipeline preprocessor -> classifier.")

    params = fitted_clf.get_params().copy()
    params["n_estimators"] = best_n_rounds(model)
    params.pop("early_stopping_rounds", None)
    use_weights = bool(getattr(model, "_use_class_weight", False))
    fold_scores = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for train_idx, valid_idx in cv.split(X_train, y_train):
        X_fit, X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_fit, y_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        fold_prep = clone(prep).set_output(transform="pandas")
        A_fit = fold_prep.fit_transform(X_fit)
        fold_params = _params_for_frame(params, A_fit)
        fold_clf = XGBClassifier(**fold_params)
        fold_clf.fit(A_fit, y_fit, sample_weight=class_weights(y_fit, use_weights), verbose=False)
        fold_model = Pipeline([("preprocessor", fold_prep), ("classifier", fold_clf)])

        baseline = f1_score(y_valid, fold_model.predict(X_valid), average="macro", labels=CLASSES, zero_division=0)
        # Select candidates using this fold's training-only model, never the
        # held-out validation fold or the final model trained on all X_train.
        fold_gain = importance_table(fold_model, aggregate=True)
        candidates = [c for c in fold_gain.sort_values("gain", ascending=False).index if c in X_train.columns][:top]
        if not candidates:
            raise ValueError("Nu s-au gasit caracteristici brute pentru permutation importance.")
        rng = np.random.default_rng(SEED)
        fold_drop = {}
        for col in candidates:
            drops = []
            for _ in range(n_repeats):
                X_perm = X_valid.copy()
                X_perm[col] = rng.permutation(X_perm[col].to_numpy())
                perm_score = f1_score(y_valid, fold_model.predict(X_perm), average="macro", labels=CLASSES, zero_division=0)
                drops.append(baseline - perm_score)
            fold_drop[col] = float(np.mean(drops))
        fold_scores.append(fold_drop)

    fold_df = pd.DataFrame(fold_scores)
    result = pd.DataFrame({
        "importance_mean": fold_df.mean(axis=0),
        "importance_std_cv": fold_df.std(axis=0, ddof=1),
        "folds_selected": fold_df.count(axis=0),
    }).sort_values("importance_mean", ascending=False)

    shown = result.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(shown.index, shown["importance_mean"], xerr=shown["importance_std_cv"].fillna(0.0))
    ax.set_xlabel("scadere macro-F1 la permutare (medie CV)")
    ax.set_title(f"Permutation importance – CV {n_splits}-fold pe train (top {top} dupa gain in fiecare fold)")
    ax.axvline(0, color="black", linewidth=0.8)
    _finish(fig, "B3_permutation_importance", save_dir)
    return result



# ----------------------------------------------------------------------------
# 4. metrici + matrice de confuzie + ROC + PR (common.evaluate) + tabel per clasa
# ----------------------------------------------------------------------------
def per_class_table(y_true, y_pred):
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    return pd.DataFrame(
        {"clasa": CLASS_NAMES, "precision": p, "recall": r, "f1": f, "support": s}, index=CLASSES
    )


def evaluate_model(model, X_test, y_test, show_plots=True):
    """Foloseste evaluate() din common.py (identic pentru toti trei) + tabel per clasa."""
    res = evaluate(model, X_test, y_test, show_plots=show_plots)
    print(per_class_table(y_test, model.predict(X_test)).round(3).to_string())
    return res


# ----------------------------------------------------------------------------
# 5. praguri per clasa care maximizeaza macro-F1
#    (invatate pe predictii out-of-fold pe TRAIN, aplicate pe test)
# ----------------------------------------------------------------------------
def refit_params(model):
    """Parametrii modelului, cu n_estimators = nr. optim de runde si fara early stopping."""
    _, clf = get_parts(model)
    p = clf.get_params()
    p["n_estimators"] = best_n_rounds(model)
    p["early_stopping_rounds"] = None
    return p


def oof_proba(model, X_train, y_train, n_splits=5, use_class_weight=True, seed=SEED):
    """Probabilitati out-of-fold (CV stratificat) pe train, cu aceeasi configuratie ca modelul dat."""
    prep, _ = get_parts(model)
    params = refit_params(model)
    oof = np.zeros((len(X_train), len(CLASSES)))
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for a, b in skf.split(X_train, y_train):
        p = clone(prep).set_output(transform="pandas")
        A, B = p.fit_transform(X_train.iloc[a]), p.transform(X_train.iloc[b])
        clf = XGBClassifier(**params)
        clf.fit(A, y_train.iloc[a], sample_weight=class_weights(y_train.iloc[a], use_class_weight))
        oof[b] = clf.predict_proba(B)
    return oof


def per_class_pr_thresholds(y_true, proba):
    """Pentru fiecare clasa (one-vs-rest): pragul din curba Precision-Recall cu F1 maxim."""
    y_true = np.asarray(y_true)
    thr, best_f1 = np.zeros(len(CLASSES)), np.zeros(len(CLASSES))
    for c in CLASSES:
        p, r, t = precision_recall_curve(y_true == c, proba[:, c])
        f = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
        i = int(np.argmax(f))
        thr[c], best_f1[c] = t[i], f[i]
    return thr, best_f1


def apply_thresholds(proba, thr):
    """Regula de decizie multiclass cu praguri: argmax_c p_c / t_c (clasa cu probabilitatea cea mai mare relativ la pragul ei)."""
    return np.argmax(proba / np.clip(np.asarray(thr), 1e-6, None), axis=1)


def optimize_class_scales(y_true, proba, grid=None, n_rounds=3):
    """
    Cauta direct factori w_c (pred = argmax p_c * w_c) care maximizeaza macro-F1,
    prin cautare coordonata cu coordonata. Pragul echivalent pe clasa este 1 / w_c.
    """
    y_true = np.asarray(y_true)
    grid = np.geomspace(0.2, 5.0, 41) if grid is None else grid
    w = np.ones(len(CLASSES))
    best = f1_score(y_true, np.argmax(proba * w, 1), average="macro", labels=CLASSES, zero_division=0)
    for _ in range(n_rounds):
        for c in CLASSES:
            for g in grid:
                w2 = w.copy()
                w2[c] = g
                s = f1_score(y_true, np.argmax(proba * w2, 1), average="macro", labels=CLASSES, zero_division=0)
                if s > best + 1e-9:
                    best, w = s, w2
    return w, best


def _scores(y_true, y_pred):
    kw = dict(labels=CLASSES, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, **kw),
        "recall_macro": recall_score(y_true, y_pred, **kw),
        "f1_macro": f1_score(y_true, y_pred, **kw),
    }


def threshold_analysis(model, X_train, y_train, X_test, y_test, save_dir=None, n_splits=5, use_class_weight=None):
    """
    Compara 3 reguli de decizie pe TEST:
      (1) argmax simplu,
      (2) praguri PR OvR invatate separat pe OOF; pentru o eticheta multiclasă
          exclusiva folosim argmax(p_c / prag_c),
      (3) factori per clasa care maximizeaza direct macro-F1 pe OOF.
    Pragurile OvR sunt praguri binare de operare, nu garanteaza direct macro-F1
    dupa conversia la o singura clasa prezisa per observatie.
    Pragurile se invata DOAR pe predictii out-of-fold din train, ca sa nu injectam informatie din test.
    """
    y_test = np.asarray(y_test)
    if use_class_weight is None:
        use_class_weight = getattr(model, "_use_class_weight", False)
    oof = oof_proba(model, X_train, y_train, n_splits=n_splits, use_class_weight=use_class_weight)
    thr_pr, f1_pr_oof = per_class_pr_thresholds(y_train, oof)
    w_opt, f1_opt_oof = optimize_class_scales(y_train, oof)
    thr_opt = 1.0 / w_opt

    proba = model.predict_proba(X_test)
    preds = {
        "argmax": np.argmax(proba, 1),
        "praguri PR (OvR)": apply_thresholds(proba, thr_pr),
        "praguri optimizate macro-F1": apply_thresholds(proba, thr_opt),
    }
    summary = pd.DataFrame({k: _scores(y_test, v) for k, v in preds.items()}).T
    perclass = pd.DataFrame({k: per_class_table(y_test, v)["f1"] for k, v in preds.items()})
    perclass.index = CLASS_NAMES
    thr_df = pd.DataFrame(
        {"prag PR (OvR)": thr_pr, "F1 OvR pe OOF": f1_pr_oof, "prag optimizat macro-F1": thr_opt}, index=CLASS_NAMES
    )
    print(f"Macro-F1 pe OOF (train): argmax={f1_score(y_train, oof.argmax(1), average='macro'):.4f} | "
          f"praguri optimizate={f1_opt_oof:.4f}")

    # grafic 1: curbe PR pe test cu punctul de operare ales pe OOF
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for c, ax in zip(CLASSES, axes.ravel()):
        p, r, _ = precision_recall_curve(y_test == c, proba[:, c])
        ax.plot(r, p, color="tab:blue")
        # This point is the genuine one-vs-rest operating point p(class)>=threshold.
        # It is not the final mutually-exclusive multiclass decision rule.
        pred_c = proba[:, c] >= thr_pr[c]
        tp = np.sum(pred_c & (y_test == c))
        pp = tp / max(pred_c.sum(), 1)
        rr = tp / max((y_test == c).sum(), 1)
        ax.scatter([rr], [pp], color="red", zorder=3, label=f"prag={thr_pr[c]:.2f}")
        ax.set_title(f"Clasa {c} ({CLASS_NAMES[c]})")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left")
    fig.suptitle("Curbe PR OvR: punctul arata p(clasa)>=prag, prag invatat pe OOF")
    _finish(fig, "B5_pr_thresholds", save_dir)

    # grafic 2: F1 per clasa pentru cele 3 reguli
    fig, ax = plt.subplots(figsize=(10, 5))
    perclass.plot.bar(ax=ax)
    ax.set_ylabel("F1 pe test")
    ax.set_title("F1 per clasa: argmax vs praguri")
    ax.tick_params(axis="x", rotation=0)
    _finish(fig, "B5_f1_per_class_thresholds", save_dir)

    # grafic 3: matricea de confuzie argmax vs praguri optimizate
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, key in zip(axes, ["argmax", "praguri optimizate macro-F1"]):
        ConfusionMatrixDisplay.from_predictions(y_test, preds[key], labels=CLASSES, cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(key)
    _finish(fig, "B5_confusion_thresholds", save_dir)

    return {"summary": summary, "per_class_f1": perclass, "thresholds": thr_df, "preds": preds}


# ----------------------------------------------------------------------------
# 5b. cu vs fara sample_weight 'balanced' pe ACELASI model
# ----------------------------------------------------------------------------
def _full_scores(y_true, y_pred):
    kw = dict(labels=CLASSES, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", **kw),
        "recall_macro": recall_score(y_true, y_pred, average="macro", **kw),
        "f1_macro": f1_score(y_true, y_pred, average="macro", **kw),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", **kw),
    }


def compare_class_weights(X_train, y_train, X_test, y_test, preprocessor, folds=None, n_splits=5, params=None, save_dir=None):
    """
    Compara antrenarea FARA ponderi cu sample_weight='balanced':
      (a) CV stratificat pe TRAIN cu numar fix de runde;
      (b) test cu acelasi numar fix de runde si aceeasi configuratie.
    Astfel diferenta izoleaza efectul ponderilor. Testul este doar pentru raportare.
    """
    folds = make_folds(X_train, y_train, preprocessor, n_splits=n_splits) if folds is None else folds
    base = {**SWEEP_BASE, **(params or {})}
    names = {False: "fara ponderi", True: "balanced"}

    rows, rec = [], []
    for k, (A, y, Bv, yv) in enumerate(folds):
        for use_w in (False, True):
            clf = XGBClassifier(**_params_for_frame(base, A))
            clf.fit(A, y, sample_weight=class_weights(y, use_w))
            pred = clf.predict(Bv)
            rows.append({"fold": k, "varianta": names[use_w], **_full_scores(yv, pred)})
            r = recall_score(yv, pred, labels=CLASSES, average=None, zero_division=0)
            rec.append({"fold": k, "varianta": names[use_w], **{f"recall_{CLASS_NAMES[c]}": r[c] for c in CLASSES}})
    cv = pd.DataFrame(rows)
    g = cv.groupby("varianta")[["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]]
    cv_summary = g.mean().join(g.std().add_suffix("_std"))
    cv_recall = pd.DataFrame(rec).groupby("varianta").mean().drop(columns="fold")

    # Test comparison uses the same fixed-round protocol as CV, so the
    # difference isolates the effect of class weights rather than early stopping.
    test_rows, test_recall = {}, {}
    for use_w in (False, True):
        prep_test = clone(preprocessor).set_output(transform="pandas")
        A_train = prep_test.fit_transform(X_train)
        A_test = prep_test.transform(X_test)
        clf = XGBClassifier(**_params_for_frame(base, A_train))
        clf.fit(A_train, y_train, sample_weight=class_weights(y_train, use_w), verbose=False)
        pred = clf.predict(A_test)
        test_rows[names[use_w]] = {**_full_scores(y_test, pred), "runde": int(base["n_estimators"])}
        r = recall_score(y_test, pred, labels=CLASSES, average=None, zero_division=0)
        test_recall[names[use_w]] = {f"recall_{CLASS_NAMES[c]}": r[c] for c in CLASSES}
    test = pd.DataFrame(test_rows).T
    test_recall = pd.DataFrame(test_recall).T

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    cv_recall.T.plot.bar(ax=axes[0], rot=30)
    axes[0].set_title("Recall per clasa, CV pe train (medie)")
    test_recall.T.plot.bar(ax=axes[1], rot=30)
    axes[1].set_title("Recall per clasa, test")
    for ax in axes:
        ax.set_ylim(0, 1)
    _finish(fig, "B5b_class_weights_recall", save_dir)
    return {"cv_summary": cv_summary, "cv_recall": cv_recall, "test": test, "test_recall": test_recall}


# ----------------------------------------------------------------------------
# 6. hiperparametri manuali (CV pe train): diagnostic over/underfitting
# ----------------------------------------------------------------------------
# baza pentru baleiaj: numar fix de runde (fara early stopping) ca sa fie comparabil intre valori
SWEEP_BASE = {**BASE_PARAMS, "n_estimators": 150}

SWEEP_GRID = {
    "max_depth": [2, 3, 4, 6, 8],
    "min_child_weight": [1, 3, 5, 10, 20],
    "gamma": [0.0, 0.5, 1.0, 2.0, 5.0],
    "subsample": [0.5, 0.7, 0.85, 1.0],
    "colsample_bytree": [0.3, 0.5, 0.7, 1.0],
    "reg_lambda": [0.0, 1.0, 10.0, 50.0],
}


def make_folds(X_train, y_train, preprocessor, n_splits=5, seed=SEED):
    """Pre-calculeaza folduri CV stratificate cu encoding-ul fitat DOAR pe partea de train a fiecarui fold."""
    folds = []
    for a, b in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X_train, y_train):
        p = clone(preprocessor).set_output(transform="pandas")
        A, B = p.fit_transform(X_train.iloc[a]), p.transform(X_train.iloc[b])
        folds.append((A, y_train.iloc[a].to_numpy(), B, y_train.iloc[b].to_numpy()))
    return folds


def _metrics_at(clf, A, y, rounds=None):
    kw = {} if rounds is None else {"iteration_range": (0, rounds)}
    proba = clf.predict_proba(A, **kw)
    pred = proba.argmax(1)
    return (
        f1_score(y, pred, average="macro", labels=CLASSES, zero_division=0),
        # Report natural-distribution logloss on both train and validation.
        # Training sample weights affect fitting, not the evaluation definition.
        log_loss(y, proba, labels=CLASSES),
    )


def sweep_param(name, values, folds, base=None, use_class_weight=True):
    """Variaza un singur hiperparametru; intoarce train / CV (medie, std) pentru macro-F1 si mlogloss."""
    base = {**SWEEP_BASE, **(base or {})}
    rows = []
    for v in values:
        tr_f1, va_f1, tr_ll, va_ll = [], [], [], []
        for A, y, B, yv in folds:
            wA = class_weights(y, use_class_weight)
            clf = XGBClassifier(**_params_for_frame({**base, name: v}, A))
            clf.fit(A, y, sample_weight=wA)
            f, l = _metrics_at(clf, A, y)
            tr_f1.append(f), tr_ll.append(l)
            f, l = _metrics_at(clf, B, yv)
            va_f1.append(f), va_ll.append(l)
        rows.append(
            {
                "param": name, "value": v,
                "train_f1": np.mean(tr_f1), "cv_f1": np.mean(va_f1), "cv_f1_std": np.std(va_f1, ddof=1),
                "train_logloss": np.mean(tr_ll), "cv_logloss": np.mean(va_ll), "cv_logloss_std": np.std(va_ll, ddof=1),
            }
        )
    return pd.DataFrame(rows)


def sweep_lr_nestimators(folds, learning_rates=(0.03, 0.1, 0.3), max_rounds=400, step=25, base=None, use_class_weight=True):
    """
    learning_rate si n_estimators sunt cuplati, deci se studiaza impreuna: pentru fiecare learning_rate se
    antreneaza o singura data max_rounds runde si se evalueaza prefixele (iteration_range) -> curba vs n_estimators.
    """
    base = {**SWEEP_BASE, **(base or {})}
    grid = list(range(step, max_rounds + 1, step))
    rows = []
    for lr in learning_rates:
        acc = {n: {"tf": [], "vf": [], "tl": [], "vl": []} for n in grid}
        for A, y, B, yv in folds:
            wA = class_weights(y, use_class_weight)
            clf = XGBClassifier(**_params_for_frame({**base, "learning_rate": lr, "n_estimators": max_rounds}, A))
            clf.fit(A, y, sample_weight=wA)
            for n in grid:
                f, l = _metrics_at(clf, A, y, n)
                acc[n]["tf"].append(f), acc[n]["tl"].append(l)
                f, l = _metrics_at(clf, B, yv, n)
                acc[n]["vf"].append(f), acc[n]["vl"].append(l)
        for n in grid:
            a = acc[n]
            rows.append(
                {
                    "learning_rate": lr, "n_estimators": n,
                    "train_f1": np.mean(a["tf"]), "cv_f1": np.mean(a["vf"]), "cv_f1_std": np.std(a["vf"], ddof=1),
                    "train_logloss": np.mean(a["tl"]), "cv_logloss": np.mean(a["vl"]),
                    "cv_logloss_std": np.std(a["vl"], ddof=1),
                }
            )
    return pd.DataFrame(rows)


def plot_sweep(df, save_dir=None):
    """Doua panouri per hiperparametru: macro-F1 si logloss, train vs CV (banda = +-1 std intre foldurile CV)."""
    name = df["param"].iloc[0]
    x = df["value"].astype(float).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x, df["train_f1"], "o--", label="train")
    axes[0].errorbar(x, df["cv_f1"], yerr=df["cv_f1_std"], fmt="o-", capsize=3, label="CV (medie +- std)")
    axes[0].set_ylabel("macro-F1")
    axes[1].plot(x, df["train_logloss"], "o--", label="train")
    axes[1].errorbar(x, df["cv_logloss"], yerr=df["cv_logloss_std"], fmt="o-", capsize=3, label="CV (medie +- std)")
    axes[1].set_ylabel("mlogloss")
    for ax in axes:
        ax.set_xlabel(name)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"Efectul lui {name} (restul hiperparametrilor la valorile de baza)")
    _finish(fig, f"B6_sweep_{name}", save_dir)


def plot_lr_nestimators(df, save_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for lr, g in df.groupby("learning_rate"):
        l, = axes[0].plot(g["n_estimators"], g["cv_f1"], "-o", ms=3, label=f"lr={lr} (CV)")
        axes[0].fill_between(g["n_estimators"], g["cv_f1"] - g["cv_f1_std"], g["cv_f1"] + g["cv_f1_std"], color=l.get_color(), alpha=0.15)
        axes[0].plot(g["n_estimators"], g["train_f1"], "--", color=l.get_color(), alpha=0.6)
        l, = axes[1].plot(g["n_estimators"], g["cv_logloss"], "-o", ms=3, label=f"lr={lr} (CV)")
        axes[1].fill_between(g["n_estimators"], g["cv_logloss"] - g["cv_logloss_std"], g["cv_logloss"] + g["cv_logloss_std"], color=l.get_color(), alpha=0.15)
        axes[1].plot(g["n_estimators"], g["train_logloss"], "--", color=l.get_color(), alpha=0.6)
    axes[0].set_ylabel("macro-F1 (continuu = CV, intrerupt = train)")
    axes[1].set_ylabel("mlogloss (continuu = CV, intrerupt = train)")
    for ax in axes:
        ax.set_xlabel("n_estimators")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("learning_rate x n_estimators")
    _finish(fig, "B6_sweep_lr_nestimators", save_dir)


def best_by_sweep(df):
    """Valoarea cu cel mai bun macro-F1 CV + cea mai mica valoare aflata la <= 1 std de optim (regula 1-SE, mai simpla = mai robusta)."""
    i = df["cv_f1"].idxmax()
    thr = df.loc[i, "cv_f1"] - df.loc[i, "cv_f1_std"]
    return df.loc[i, "value"], df[df["cv_f1"] >= thr]["value"].tolist()


# ----------------------------------------------------------------------------
# 7. raport complet pe un model (pentru rerulare pe modelul final din C)
# ----------------------------------------------------------------------------
def full_report(model, X_train, y_train, X_test, y_test, class_idx=0, save_dir=None, with_thresholds=True, use_class_weight=None):
    """Reproduce toata Partea B pe un model dat: curba, primul/cel mai bun arbore, importance, metrici, praguri."""
    out = {}
    plot_training_curve(model, save_dir)

    f_idx, b_idx = first_tree_idx(model, class_idx), best_tree_idx(model, class_idx)
    print(tree_summary(model, [f_idx, b_idx]).round(2).to_string())
    plot_tree_idx(model, f_idx, f"Primul arbore (runda 0, clasa {class_idx})", save_dir=save_dir, name="B2_first_tree")
    plot_tree_idx(model, b_idx, f"Arborele din runda optima {best_n_rounds(model) - 1} (clasa {class_idx})",
                  save_dir=save_dir, name="B2_best_tree")

    out["importance"] = plot_importance(model, save_dir=save_dir)
    out["permutation"] = plot_permutation_importance(model, X_train, y_train, save_dir=save_dir)
    out["metrics"] = evaluate_model(model, X_test, y_test)
    if with_thresholds:
        if use_class_weight is None:
            use_class_weight = getattr(model, "_use_class_weight", False)
        out["thresholds"] = threshold_analysis(
            model, X_train, y_train, X_test, y_test,
            save_dir=save_dir, use_class_weight=use_class_weight,
        )
    return out


# ----------------------------------------------------------------------------
# spatiul de cautare propus pentru Optuna (Partea C) - de actualizat dupa rezultatele din notebook
# ----------------------------------------------------------------------------
SEARCH_SPACE = {
    "max_depth": (2, 8),                 # zona buna: 3-6
    "learning_rate": (0.01, 0.3, "log"),
    "n_estimators": (100, 800),          # sau: n_estimators mare + early stopping / pruning
    "subsample": (0.5, 1.0),             # efect mic
    "colsample_bytree": (0.3, 1.0),
    "min_child_weight": (1, 10, "log"),  # 20 duce la underfitting
    "gamma": (0.0, 2.0),                 # 5 duce la underfitting
    "reg_lambda": (1e-2, 20.0, "log"),   # 50 duce la underfitting
    "sample_weight_balanced": [False, True],  # categoric: cu / fara ponderi (decizia din A.9)
}

def compare_onehot_native_cv(X_train, y_train, preprocessor, n_splits=5, n_estimators=150, seed=SEED):
    """Comparatie echitabila One-Hot vs native categorical pe aceleasi folduri stratificate.

    Ambele variante folosesc acelasi numar fix de boosting rounds si aceiasi hiperparametri.
    Encoderul One-Hot este fitat numai pe foldul de train. Pentru varianta nativa,
    feature-urile de calitate raman ordinale, iar nominalele devin pandas category.
    """
    quality_order = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
    quality_cols = [c for c in [
        "Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond", "Heating QC",
        "Kitchen Qual", "Fireplace Qu", "Garage Qual", "Garage Cond", "Pool QC"
    ] if c in X_train.columns]
    rows = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    base = {**BASE_PARAMS, "n_estimators": n_estimators}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
        Xtr, Xva = X_train.iloc[tr_idx].copy(), X_train.iloc[va_idx].copy()
        ytr, yva = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        # One-hot: fit preprocessing only on the training partition of this fold.
        p = clone(preprocessor).set_output(transform="pandas")
        A_oh = p.fit_transform(Xtr)
        B_oh = p.transform(Xva)
        clf_oh = XGBClassifier(**base)
        clf_oh.fit(A_oh, ytr, verbose=False)
        pred_oh = clf_oh.predict(B_oh)
        rows.append({"encoding": "onehot", "fold": fold,
                     "macro_f1": f1_score(yva, pred_oh, average="macro", labels=CLASSES, zero_division=0)})

        # Native: ordinal quality columns + categorical nominal columns.
        for col in quality_cols:
            Xtr[col] = Xtr[col].map({v: i for i, v in enumerate(quality_order)}).astype("float32")
            Xva[col] = Xva[col].map({v: i for i, v in enumerate(quality_order)}).astype("float32")
        nominal_cols = Xtr.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        for col in nominal_cols:
            categories = pd.Index(Xtr[col].dropna().unique()).sort_values()
            Xtr[col] = pd.Categorical(Xtr[col], categories=categories)
            Xva[col] = pd.Categorical(Xva[col], categories=categories)
        clf_native = XGBClassifier(**base, enable_categorical=True)
        clf_native.fit(Xtr, ytr, verbose=False)
        pred_native = clf_native.predict(Xva)
        rows.append({"encoding": "native", "fold": fold,
                     "macro_f1": f1_score(yva, pred_native, average="macro", labels=CLASSES, zero_division=0)})

    raw = pd.DataFrame(rows)
    summary = raw.groupby("encoding")["macro_f1"].agg(["mean", "std", "min", "max"])
    return {"fold_scores": raw, "summary": summary}


# ============================================================================
# BONUS (Partea C): LightGBM pe aceleasi features
# ----------------------------------------------------------------------------
# Echivalentul pentru LightGBM al functiilor XGBoost de mai sus. Folosesc
# aceleasi date (load_data, make_split), aceleasi folduri (make_folds),
# aceleasi ponderi (class_weights) si aceleasi metrici (evaluate_model),
# ca cele doua modele sa poata fi comparate direct.
# tree_frame, plot_tree_generic si plot_structure_comparison functioneaza
# atat pentru XGBoost, cat si pentru LightGBM (pentru comparatia arborilor).
# ============================================================================
import time

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except ImportError:  # restul modulului ramane utilizabil si fara lightgbm instalat
    lgb = None
    LGBMClassifier = None


def _require_lgb():
    if lgb is None:
        raise ImportError("Functiile LightGBM necesita pachetul lightgbm: pip install lightgbm")


# hiperparametri de baza, alesi paralel cu BASE_PARAMS de la XGBoost
LGB_BASE_PARAMS = dict(
    objective="multiclass",
    n_estimators=1000,          # numarul real de runde il decide early stopping
    learning_rate=0.1,
    num_leaves=31,              # parametrul principal de complexitate (arbori crescuti frunza cu frunza)
    max_depth=-1,               # fara limita de adancime; complexitatea e limitata de num_leaves
    min_child_samples=20,       # numarul minim de case dintr-o frunza
    subsample=0.8,
    subsample_freq=1,           # LightGBM aplica subsample doar daca subsample_freq > 0
    colsample_bytree=0.8,
    min_split_gain=0.0,         # echivalentul lui gamma din XGBoost
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,                 # fara mesaje la fiecare antrenare
    deterministic=True,         # rezultate reproductibile
    force_row_wise=True,
)

# spatiul de cautare pentru Optuna (stabilit pe baza sweep-urilor din notebook-ul LightGBM)
LGB_SEARCH_SPACE = {
    "num_leaves": (4, 64, "log"),
    "learning_rate": (0.01, 0.3, "log"),
    "n_estimators": (100, 800),
    "min_child_samples": (5, 40, "log"),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.2, 1.0),
    "min_split_gain": (0.0, 1.0),
    "reg_lambda": (1e-2, 20.0, "log"),
    "sample_weight_balanced": [False, True],
}


def fit_lgb(
    X_train,
    y_train,
    preprocessor,
    params=None,
    early_stopping=True,
    early_stopping_rounds=30,
    val_size=0.15,
    use_class_weight=True,
    seed=SEED,
    refit_full=True,
):
    """Antreneaza LightGBM fara leakage si returneaza un Pipeline fitat (aceeasi logica ca fit_xgb).

    Cu early_stopping=True: imparte numai TRAIN in fit/validare, alege numarul de runde
    pe validare, apoi (implicit) reantreneaza pe tot X_train cu numarul de runde ales.
    Testul nu este accesat.
    """
    _require_lgb()
    p = {**LGB_BASE_PARAMS, **(params or {})}
    prep = clone(preprocessor).set_output(transform="pandas")
    history, selected_iteration = None, None

    if early_stopping:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train, test_size=val_size, stratify=y_train, random_state=seed
        )
        prep_select = clone(prep).set_output(transform="pandas")
        A_fit = prep_select.fit_transform(X_fit)
        A_val = prep_select.transform(X_val)
        selector = LGBMClassifier(**p)
        selector.fit(
            A_fit,
            y_fit,
            sample_weight=class_weights(y_fit, use_class_weight),
            eval_X=(A_fit, A_val),
            eval_y=(y_fit, y_val),
            eval_names=["train", "validare"],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        selected_rounds = int(selector.best_iteration_) or int(p["n_estimators"])
        history = selector.evals_result_
        selected_iteration = selected_rounds - 1

        if refit_full:
            A_train = prep.fit_transform(X_train)
            clf = LGBMClassifier(**{**p, "n_estimators": selected_rounds})
            clf.fit(A_train, y_train, sample_weight=class_weights(y_train, use_class_weight))
        else:
            prep, clf = prep_select, selector
    else:
        A_train = prep.fit_transform(X_train)
        clf = LGBMClassifier(**p)
        clf.fit(A_train, y_train, sample_weight=class_weights(y_train, use_class_weight))

    fitted = Pipeline([("preprocessor", prep), ("classifier", clf)])
    fitted._early_stopping_history = history
    fitted._early_stopping_best_iteration = selected_iteration
    fitted._early_stopping_selected_rounds = selected_iteration + 1 if selected_iteration is not None else None
    fitted._use_class_weight = bool(use_class_weight)
    return fitted


def lgb_cross_validation(params, folds, use_class_weight=True):
    """Macro-F1 mediu si deviatia standard pe foldurile din make_folds (folosita de Optuna)."""
    _require_lgb()
    p = {**LGB_BASE_PARAMS, **params}
    scores = []
    for A, y, B, yv in folds:
        clf = LGBMClassifier(**p)
        clf.fit(A, y, sample_weight=class_weights(y, use_class_weight))
        scores.append(f1_score(yv, clf.predict(B), average="macro", labels=CLASSES, zero_division=0))
    return float(np.mean(scores)), float(np.std(scores))


def plot_lgb_training_curve(model, save_dir=None):
    """Curba de invatare (multi_logloss pe train si validare, pe runde) din etapa de early stopping."""
    ev = getattr(model, "_early_stopping_history", None)
    if not ev:
        print("Curba nu este disponibila: modelul a fost antrenat fara early stopping.")
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, style in (("train", "--"), ("validare", "-")):
        if name in ev:
            ax.plot(ev[name]["multi_logloss"], style, label=name)
    bi = getattr(model, "_early_stopping_best_iteration", None)
    if bi is not None:
        ax.axvline(bi, color="red", ls="--", label=f"best_iteration = {bi}")
    ax.set_xlabel("runda de boosting")
    ax.set_ylabel("multi_logloss")
    ax.set_title("Curba de invatare LightGBM (early stopping)")
    ax.legend()
    ax.grid(alpha=0.3)
    _finish(fig, "LGB1_training_curve", save_dir)


# ---------------------------------------------------------------- sweep-uri manuale
LGB_SWEEP_BASE = {**LGB_BASE_PARAMS, "n_estimators": 150}

LGB_SWEEP_GRID = {
    "num_leaves": [4, 8, 15, 31, 63, 127],
    "min_child_samples": [5, 10, 20, 40, 80],
    "min_split_gain": [0.0, 0.25, 0.5, 1.0, 2.0],
    "subsample": [0.5, 0.7, 0.85, 1.0],
    "colsample_bytree": [0.2, 0.3, 0.5, 0.7, 1.0],
    "reg_lambda": [0.0, 1.0, 10.0, 50.0],
}


def _lgb_metrics_at(clf, A, y, rounds=None):
    proba = clf.predict_proba(A, num_iteration=rounds)
    pred = proba.argmax(1)
    return (
        f1_score(y, pred, average="macro", labels=CLASSES, zero_division=0),
        log_loss(y, proba, labels=CLASSES),
    )


def lgb_sweep_param(name, values, folds, base=None, use_class_weight=True):
    """Variaza un singur hiperparametru LightGBM; acelasi format ca sweep_param (merge cu plot_sweep)."""
    _require_lgb()
    base = {**LGB_SWEEP_BASE, **(base or {})}
    rows = []
    for v in values:
        tr_f1, va_f1, tr_ll, va_ll = [], [], [], []
        for A, y, B, yv in folds:
            clf = LGBMClassifier(**{**base, name: v})
            clf.fit(A, y, sample_weight=class_weights(y, use_class_weight))
            f, l = _lgb_metrics_at(clf, A, y)
            tr_f1.append(f), tr_ll.append(l)
            f, l = _lgb_metrics_at(clf, B, yv)
            va_f1.append(f), va_ll.append(l)
        rows.append(
            {
                "param": name, "value": v,
                "train_f1": np.mean(tr_f1), "cv_f1": np.mean(va_f1), "cv_f1_std": np.std(va_f1, ddof=1),
                "train_logloss": np.mean(tr_ll), "cv_logloss": np.mean(va_ll), "cv_logloss_std": np.std(va_ll, ddof=1),
            }
        )
    return pd.DataFrame(rows)


def lgb_sweep_lr_nestimators(folds, learning_rates=(0.03, 0.1, 0.3), max_rounds=400, step=25, base=None, use_class_weight=True):
    """learning_rate x n_estimators pentru LightGBM; acelasi format ca sweep_lr_nestimators (merge cu plot_lr_nestimators)."""
    _require_lgb()
    base = {**LGB_SWEEP_BASE, **(base or {})}
    grid = list(range(step, max_rounds + 1, step))
    rows = []
    for lr in learning_rates:
        acc = {n: {"tf": [], "vf": [], "tl": [], "vl": []} for n in grid}
        for A, y, B, yv in folds:
            clf = LGBMClassifier(**{**base, "learning_rate": lr, "n_estimators": max_rounds})
            clf.fit(A, y, sample_weight=class_weights(y, use_class_weight))
            for n in grid:
                f, l = _lgb_metrics_at(clf, A, y, n)
                acc[n]["tf"].append(f), acc[n]["tl"].append(l)
                f, l = _lgb_metrics_at(clf, B, yv, n)
                acc[n]["vf"].append(f), acc[n]["vl"].append(l)
        for n in grid:
            a = acc[n]
            rows.append(
                {
                    "learning_rate": lr, "n_estimators": n,
                    "train_f1": np.mean(a["tf"]), "cv_f1": np.mean(a["vf"]), "cv_f1_std": np.std(a["vf"], ddof=1),
                    "train_logloss": np.mean(a["tl"]), "cv_logloss": np.mean(a["vl"]),
                    "cv_logloss_std": np.std(a["vl"], ddof=1),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- arbori (XGBoost si LightGBM)
def _model_kind(model):
    _, clf = get_parts(model)
    if LGBMClassifier is not None and isinstance(clf, LGBMClassifier):
        return "lightgbm"
    if isinstance(clf, XGBClassifier):
        return "xgboost"
    raise TypeError("Modelul trebuie sa fie XGBClassifier sau LGBMClassifier (eventual intr-un Pipeline).")


def _lgb_name_map(model):
    """LightGBM inlocuieste spatiile din numele coloanelor cu '_'; refacem numele originale (aceeasi ordine)."""
    prep, clf = get_parts(model)
    internal = clf.booster_.feature_name()
    original = list(prep.get_feature_names_out()) if prep is not None else internal
    return dict(zip(internal, original))


def tree_frame(model, trees=None):
    """Toate nodurile arborilor intr-un tabel comun pentru XGBoost si LightGBM.

    Coloane: tree, node_id, left, right, is_leaf, feature, op, threshold, gain, cover, value, depth.
    La ambele modele, in clasificarea multiclasa indexul arborelui este runda * 6 + clasa.
    """
    kind = _model_kind(model)
    if kind == "xgboost":
        df = get_booster(model).trees_to_dataframe()
        if trees is not None:
            df = df[df["Tree"].isin(trees)]
        depth = {}
        rows = []
        for r in df.itertuples(index=False):
            d = depth.get(r.ID, 0)
            leaf = r.Feature == "Leaf"
            if not leaf:
                depth[r.Yes] = d + 1
                depth[r.No] = d + 1
            rows.append({
                "tree": r.Tree, "node_id": r.ID,
                "left": None if leaf else r.Yes, "right": None if leaf else r.No,
                "is_leaf": leaf, "feature": None if leaf else r.Feature, "op": "<",
                "threshold": np.nan if leaf else r.Split,
                "gain": 0.0 if leaf else r.Gain, "cover": r.Cover,
                "value": r.Gain if leaf else np.nan, "depth": d,
            })
        return pd.DataFrame(rows)

    _, clf = get_parts(model)
    df = clf.booster_.trees_to_dataframe()
    if trees is not None:
        df = df[df["tree_index"].isin(trees)]
    names = _lgb_name_map(model)
    leaf = df["left_child"].isna()
    return pd.DataFrame({
        "tree": df["tree_index"].to_numpy(),
        "node_id": df["node_index"].to_numpy(),
        "left": df["left_child"].where(~leaf, None).to_numpy(),
        "right": df["right_child"].where(~leaf, None).to_numpy(),
        "is_leaf": leaf.to_numpy(),
        "feature": df["split_feature"].map(names).where(~leaf, None).to_numpy(),
        "op": "<=",
        "threshold": df["threshold"].to_numpy(),
        "gain": df["split_gain"].fillna(0.0).to_numpy(),
        "cover": df["count"].to_numpy(),
        "value": df["value"].where(leaf).to_numpy(),
        "depth": (df["node_depth"] - 1).to_numpy(),
    })


def tree_structure_stats(model):
    """Un rand per arbore: runda, clasa, numar de noduri si frunze, adancime maxima, feature-ul radacinii, gain total."""
    fr = tree_frame(model)
    g = fr.groupby("tree")
    stats = pd.DataFrame({
        "noduri": g.size(),
        "frunze": g["is_leaf"].sum(),
        "adancime_max": g["depth"].max(),
        "gain_total": g["gain"].sum(),
    })
    roots = fr[fr["depth"] == 0].set_index("tree")["feature"]
    stats["radacina"] = roots
    stats["runda"] = stats.index // len(CLASSES)
    stats["clasa"] = stats.index % len(CLASSES)
    return stats[["runda", "clasa", "noduri", "frunze", "adancime_max", "radacina", "gain_total"]]


def first_tree_idx_any(model, class_idx=0):
    """Indexul primului arbore al clasei date (runda 0), pentru XGBoost si LightGBM."""
    return class_idx


def last_tree_idx_any(model, class_idx=0):
    """Indexul arborelui clasei date din ultima runda pastrata de model."""
    if _model_kind(model) == "xgboost":
        n_rounds = best_n_rounds(model)
    else:
        _, clf = get_parts(model)
        n_rounds = clf.booster_.num_trees() // len(CLASSES)
    return (n_rounds - 1) * len(CLASSES) + class_idx


def plot_tree_generic(model, tree_idx, title="", save_dir=None, name=None, max_label=22):
    """Deseneaza un arbore cu matplotlib (nu necesita Graphviz). Merge pentru XGBoost si LightGBM.

    Nodurile interne arata conditia de split (ramura stanga = conditia adevarata),
    frunzele arata valoarea adaugata la scorul clasei (verde > 0, rosu < 0).
    """
    fr = tree_frame(model, trees=[tree_idx]).set_index("node_id")
    if fr.empty:
        raise ValueError(f"Arborele {tree_idx} nu exista in model.")
    root = fr.index[fr["depth"] == 0][0]
    pos, counter = {}, [0]

    def place(n):
        r = fr.loc[n]
        if r["is_leaf"]:
            pos[n] = (counter[0], -r["depth"])
            counter[0] += 1
        else:
            place(r["left"])
            place(r["right"])
            pos[n] = ((pos[r["left"]][0] + pos[r["right"]][0]) / 2, -r["depth"])

    place(root)
    n_leaves, max_d = counter[0], int(fr["depth"].max())
    fig, ax = plt.subplots(figsize=(max(10, 1.25 * n_leaves), max(4, 1.3 * (max_d + 1))))
    for n, r in fr.iterrows():
        if not r["is_leaf"]:
            for child, lab in ((r["left"], "da"), (r["right"], "nu")):
                (x0, y0), (x1, y1) = pos[n], pos[child]
                ax.plot([x0, x1], [y0, y1], color="gray", lw=0.8, zorder=1)
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, lab, fontsize=6, color="gray", ha="center", va="center")
    for n, r in fr.iterrows():
        x, yy = pos[n]
        if r["is_leaf"]:
            txt, color = f"{r['value']:+.3f}", ("#c7e9c0" if r["value"] > 0 else "#fcbba1")
        else:
            feat = str(r["feature"])
            feat = feat if len(feat) <= max_label else feat[: max_label - 1] + "…"
            thr = 0.0 if abs(r["threshold"]) < 1e-6 else r["threshold"]  # LightGBM scrie pragul 0 ca 1e-35
            txt, color = f"{feat}\n{r['op']} {thr:.4g}", "#c6dbef"
        ax.text(x, yy, txt, ha="center", va="center", fontsize=6.5, zorder=2,
                bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="gray", lw=0.6))
    ax.set_xlim(-0.8, n_leaves - 0.2)
    ax.set_ylim(-max_d - 0.6, 0.6)
    ax.axis("off")
    ax.set_title(title or f"Arborele {tree_idx} ({n_leaves} frunze, adancime {max_d})")
    _finish(fig, name or f"tree_{tree_idx}", save_dir)


def plot_structure_comparison(models, save_dir=None):
    """Compara forma arborilor mai multor modele: frunze per arbore, adancime maxima, adancimea frunzelor.

    models: dict {nume: model fitat}. Returneaza un tabel cu statistici medii.
    """
    frames = {name: tree_frame(m) for name, m in models.items()}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    summary = []
    for name, fr in frames.items():
        g = fr.groupby("tree")
        leaves = g["is_leaf"].sum()
        depth = g["depth"].max()
        leaf_depths = fr.loc[fr["is_leaf"], "depth"]
        axes[0].hist(leaves, bins=30, alpha=0.55, density=True, label=name)
        axes[1].hist(depth, bins=np.arange(depth.max() + 2) - 0.5, alpha=0.55, density=True, label=name)
        axes[2].hist(leaf_depths, bins=np.arange(leaf_depths.max() + 2) - 0.5, alpha=0.55, density=True, label=name)
        summary.append({
            "model": name, "arbori": int(fr["tree"].nunique()),
            "frunze/arbore (medie)": leaves.mean(), "adancime max (medie)": depth.mean(),
            "adancime max (maxim)": int(depth.max()),
            "adancimea frunzelor (medie)": leaf_depths.mean(), "adancimea frunzelor (std)": leaf_depths.std(),
        })
    axes[0].set_xlabel("numar de frunze per arbore")
    axes[1].set_xlabel("adancimea maxima a arborelui")
    axes[2].set_xlabel("adancimea la care se afla o frunza")
    for ax in axes:
        ax.set_ylabel("proportie")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Structura arborilor: XGBoost (nivel cu nivel) vs LightGBM (frunza cu frunza)")
    _finish(fig, "LGB2_structure_comparison", save_dir)
    return pd.DataFrame(summary).set_index("model")


# ---------------------------------------------------------------- importanta features
def lgb_importance_table(model, aggregate=False):
    """Importanta LightGBM: gain (mediu pe split), total_gain, split (numar de split-uri).

    aggregate=True aduna coloanele one-hot inapoi la feature-ul original, ca la importance_table.
    """
    _, clf = get_parts(model)
    booster = clf.booster_
    name_map = _lgb_name_map(model)
    names = [name_map[n] for n in booster.feature_name()]
    df = pd.DataFrame({
        "total_gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split").astype(float),
    }, index=names)
    if aggregate:
        origin = _origin_map(model, names)
        df = df.groupby(df.index.map(origin))[["total_gain", "split"]].sum()
    df["gain"] = (df["total_gain"] / df["split"].replace(0, np.nan)).fillna(0.0)
    df["rank_gain"] = df["gain"].rank(ascending=False, method="min").astype(int)
    df["rank_split"] = df["split"].rank(ascending=False, method="min").astype(int)
    return df[["gain", "total_gain", "split", "rank_gain", "rank_split"]].sort_values("gain", ascending=False)


def plot_lgb_importance(model, top=20, aggregate=True, save_dir=None):
    """Doua grafice: gain mediu si numar de split-uri (feature-urile create in L1 sunt evidentiate)."""
    imp = lgb_importance_table(model, aggregate=aggregate)
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.35 * top + 2))
    for ax, col, label in ((axes[0], "gain", "gain mediu pe split"), (axes[1], "split", "numar de split-uri")):
        s = imp[col].sort_values(ascending=False).head(top)[::-1]
        colors = ["#e6550d" if f in FE_FEATURES else "#3182bd" for f in s.index]
        ax.barh(s.index, s.values, color=colors)
        ax.set_xlabel(label)
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle(f"Importanta features LightGBM (top {top}; portocaliu = feature-uri create in L1)")
    _finish(fig, "LGB3_feature_importance", save_dir)
    return imp


def compare_importance(xgb_model, lgb_model, top=15, save_dir=None):
    """Compara importanta XGBoost vs LightGBM pe aceeasi scara: procentul din gain-ul total al modelului.

    Gain-urile absolute ale celor doua biblioteci nu sunt direct comparabile, procentele da.
    """
    x = importance_table(xgb_model, aggregate=True)
    l = lgb_importance_table(lgb_model, aggregate=True)
    tab = pd.DataFrame({
        "XGBoost % gain": 100 * x["total_gain"] / x["total_gain"].sum(),
        "LightGBM % gain": 100 * l["total_gain"] / l["total_gain"].sum(),
        "XGBoost % split": 100 * x["split"] / x["split"].sum(),
        "LightGBM % split": 100 * l["split"] / l["split"].sum(),
    }).fillna(0.0)
    tab = tab.sort_values("XGBoost % gain", ascending=False)
    show = tab.loc[tab[["XGBoost % gain", "LightGBM % gain"]].max(axis=1).sort_values(ascending=False).index[:top]]
    show = show.sort_values("XGBoost % gain")
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.4 * top + 2), sharey=True)
    yy = np.arange(len(show))
    for ax, kind in ((axes[0], "gain"), (axes[1], "split")):
        ax.barh(yy - 0.2, show[f"XGBoost % {kind}"], 0.4, label="XGBoost")
        ax.barh(yy + 0.2, show[f"LightGBM % {kind}"], 0.4, label="LightGBM")
        ax.set_xlabel(f"% din {'gain-ul total' if kind == 'gain' else 'numarul total de split-uri'}")
        ax.legend()
        ax.grid(alpha=0.3, axis="x")
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(show.index)
    fig.suptitle("Importanta features: XGBoost vs LightGBM (agregat pe feature-ul original)")
    _finish(fig, "LGB4_importance_comparison", save_dir)
    rho = spearmanr(tab["XGBoost % gain"], tab["LightGBM % gain"]).correlation
    print(f"Corelatia Spearman intre clasamentele dupa gain: {rho:.3f}")
    return tab


# ---------------------------------------------------------------- procesul de invatare comparat
def compare_learning_curves(X_train, y_train, preprocessor, configs, val_size=0.15, step=10, seed=SEED, save_dir=None):
    """Antreneaza fiecare model pe aceeasi impartire fit/validare din TRAIN si inregistreaza, pe runde,
    logloss-ul (train si validare) si macro-F1 pe validare.

    configs: dict {nume: (tip, params, use_class_weight)}, tip = "xgboost" sau "lightgbm".
    Returneaza (tabel pe runde, tabel cu timpii de antrenare).
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=seed
    )
    prep = clone(preprocessor).set_output(transform="pandas")
    A, B = prep.fit_transform(X_fit), prep.transform(X_val)
    rows, times = [], {}
    for name, (kind, params, use_w) in configs.items():
        w = class_weights(y_fit, use_w)
        # timpul de antrenare masurat separat, fara evaluarea pe runde
        t0 = time.perf_counter()
        if kind == "xgboost":
            XGBClassifier(**{**BASE_PARAMS, **params}).fit(A, y_fit, sample_weight=w, verbose=False)
        else:
            _require_lgb()
            LGBMClassifier(**{**LGB_BASE_PARAMS, **params}).fit(A, y_fit, sample_weight=w)
        times[name] = time.perf_counter() - t0

        if kind == "xgboost":
            clf = XGBClassifier(**{**BASE_PARAMS, **params})
            clf.fit(A, y_fit, sample_weight=w, eval_set=[(A, y_fit), (B, y_val)], verbose=False)
            ev = clf.evals_result()
            tr_ll, va_ll = ev["validation_0"]["mlogloss"], ev["validation_1"]["mlogloss"]
            proba = lambda X, n: clf.predict_proba(X, iteration_range=(0, n))
        else:
            clf = LGBMClassifier(**{**LGB_BASE_PARAMS, **params})
            clf.fit(A, y_fit, sample_weight=w, eval_X=(A, B), eval_y=(y_fit, y_val), eval_names=["train", "validare"])
            ev = clf.evals_result_
            tr_ll, va_ll = ev["train"]["multi_logloss"], ev["validare"]["multi_logloss"]
            proba = lambda X, n: clf.predict_proba(X, num_iteration=n)
        n_rounds = len(va_ll)
        for n in sorted(set(list(range(step, n_rounds + 1, step)) + [n_rounds])):
            rows.append({
                "model": name, "runda": n,
                "train_logloss": tr_ll[n - 1], "val_logloss": va_ll[n - 1],
                "train_f1": f1_score(y_fit, proba(A, n).argmax(1), average="macro", labels=CLASSES, zero_division=0),
                "val_f1": f1_score(y_val, proba(B, n).argmax(1), average="macro", labels=CLASSES, zero_division=0),
            })
    curves = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for name, g in curves.groupby("model", sort=False):
        l, = axes[0].plot(g["runda"], g["val_logloss"], "-", label=f"{name} (validare)")
        axes[0].plot(g["runda"], g["train_logloss"], "--", color=l.get_color(), alpha=0.7, label=f"{name} (train)")
        axes[1].plot(g["runda"], g["val_f1"], "-", color=l.get_color(), label=f"{name} (validare)")
        axes[1].plot(g["runda"], g["train_f1"], "--", color=l.get_color(), alpha=0.7, label=f"{name} (train)")
    axes[0].set_ylabel("logloss")
    axes[1].set_ylabel("macro-F1")
    for ax in axes:
        ax.set_xlabel("runda de boosting")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Procesul de invatare: XGBoost vs LightGBM (hiperparametrii optimizati, aceeasi validare)")
    _finish(fig, "LGB5_learning_curves", save_dir)
    return curves, pd.Series(times, name="timp antrenare (s)")
