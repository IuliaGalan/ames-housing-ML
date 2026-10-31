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
