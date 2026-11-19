"""
Lab 4 - cod comun pentru toata echipa (Pasul 0).

Toate experimentele (Partea 1, 2, 3) folosesc aceleasi functii de aici, ca rezultatele sa fie comparabile:
  - aceleasi imagini, aceeasi dimensiune, aceeasi normalizare
  - aceleasi 5 fold-uri (StratifiedKFold, seed 42), salvate o singura data in results/lab4/folds.csv
  - acelasi antrenament (early stopping pe macro-F1 de validare)
  - acelasi mod de salvare: fiecare experiment are folderul lui, results/lab4/runs/<name>/
    (asa nu apar conflicte in git cand fiecare coleg adauga experimente pe branch-ul lui);
    load_log() aduna toate experimentele terminate intr-un singur tabel

Utilizare tipica intr-un notebook:
    import lab4_common
    cfg = lab4_common.make_config(name="fc_1x256_relu", conv_layers=[], fc_layers=[256])
    rezultat = lab4_common.run_experiment(cfg)

Un experiment = un dictionar de configurare (vezi DEFAULT_CONFIG). Numele ("name") trebuie sa fie unic.
Daca un experiment (sau un fold din el) a fost deja rulat, nu se mai ruleaza din nou: se citeste rezultatul salvat.
Asa putem relua lucrul dupa o deconectare in Colab fara sa pierdem ce s-a terminat.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# 1. Cai (foldere) si setari generale
# ---------------------------------------------------------------------------

# Radacina proiectului = folderul care contine pyproject.toml
ROOT = Path(__file__).resolve().parents[1]

# Implicit: pozele in data/raw/flower_photos, rezultatele in results/lab4.
# In Colab putem schimba folderele cu set_paths(...) (de ex. rezultatele direct in Google Drive).
DATA_DIR = Path(os.environ.get("LAB4_DATA_DIR", ROOT / "data" / "raw" / "flower_photos"))
OUT_DIR = Path(os.environ.get("LAB4_OUT_DIR", ROOT / "results" / "lab4"))

SEED = 42
N_FOLDS = 5
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


def set_paths(data_dir=None, out_dir=None):
    """Schimba folderul cu poze si/sau folderul unde se salveaza rezultatele."""
    global DATA_DIR, OUT_DIR
    if data_dir is not None:
        DATA_DIR = Path(data_dir)
    if out_dir is not None:
        OUT_DIR = Path(out_dir)
    print(f"DATA_DIR = {DATA_DIR}\nOUT_DIR  = {OUT_DIR}")


def get_device(prefer=None):
    """Alege automat GPU (cuda) daca exista, altfel CPU. prefer='cpu' forteaza CPU (pt. comparatia CPU vs GPU)."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 2. Date: lista de imagini, incarcare + redimensionare, fold-uri
# ---------------------------------------------------------------------------

def list_images(data_dir=None):
    """Tabel cu toate imaginile: path (relativ la DATA_DIR), clasa (text) si label (0-4)."""
    data_dir = Path(data_dir or DATA_DIR)
    rows = []
    for label, cls in enumerate(CLASSES):
        for p in sorted((data_dir / cls).glob("*.jpg")):
            rows.append({"path": f"{cls}/{p.name}", "class": cls, "label": label})
    if not rows:
        raise FileNotFoundError(f"Nu am gasit imagini in {data_dir}. Ai dezarhivat flower_photos.tgz?")
    return pd.DataFrame(rows)


def _load_one(path, img_size):
    """Deschide o imagine, taie un patrat din centru si o micsoreaza la img_size x img_size."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((img_size, img_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_dataset(img_size=64):
    """
    Incarca TOATE imaginile in memorie, deja redimensionate.
    Intoarce (df, X, y):
      df - tabelul de la list_images
      X  - tensor uint8 de forma (N, 3, img_size, img_size)
      y  - tensor long cu etichetele (N,)
    Prima data dureaza putin (citeste ~3670 de poze); apoi se citeste din cache (results/lab4/cache/).
    """
    df = list_images()
    cache = OUT_DIR / "cache" / f"images_{img_size}.pt"
    if cache.exists():
        saved = torch.load(cache)
        if saved["paths"] == df["path"].tolist():
            return df, saved["X"], saved["y"]

    X = np.stack([_load_one(DATA_DIR / p, img_size) for p in df["path"]])
    X = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()  # (N, H, W, 3) -> (N, 3, H, W)
    y = torch.tensor(df["label"].values, dtype=torch.long)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"paths": df["path"].tolist(), "X": X, "y": y}, cache)
    return df, X, y


def get_folds(df):
    """
    Cele 5 fold-uri comune (StratifiedKFold, shuffle, seed 42).
    Se calculeaza o singura data si se salveaza in results/lab4/folds.csv; dupa aceea se citesc din fisier,
    ca toti colegii sa foloseasca exact aceeasi impartire. Intoarce un array: fold-ul (0-4) fiecarei imagini.
    """
    path = OUT_DIR / "folds.csv"
    if path.exists():
        saved = pd.read_csv(path)
        if saved["path"].tolist() == df["path"].tolist():
            return saved["fold"].values
        raise ValueError("folds.csv nu corespunde imaginilor gasite. Verifica folderul DATA_DIR.")

    fold = np.zeros(len(df), dtype=int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, val_idx) in enumerate(skf.split(df, df["label"])):
        fold[val_idx] = k
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": df["path"], "fold": fold}).to_csv(path, index=False)
    return fold


def to_float(xb):
    """uint8 [0, 255] -> float [-1, 1] (normalizare fixa, aceeasi pentru toate experimentele)."""
    return xb.float().div(127.5).sub(1.0)


def augment(xb, strength=1.0):
    """
    Augmentare simpla, aplicata DOAR pe batch-urile de antrenare (fiecare imagine primeste alte modificari):
      - oglindire orizontala (50% sanse)
      - mutare aleatoare cu cativa pixeli (ca un crop aleator)
      - luminozitate si contrast putin modificate
    xb: tensor float (B, 3, H, W) deja normalizat in [-1, 1].
    """
    B, _, H, W = xb.shape
    dev = xb.device

    flip = torch.rand(B, device=dev) < 0.5
    xb = torch.where(flip[:, None, None, None], xb.flip(-1), xb)

    pad = max(1, int(round(H * 0.1 * strength)))
    padded = nn.functional.pad(xb, (pad, pad, pad, pad), mode="reflect")
    dx = torch.randint(0, 2 * pad + 1, (B,)).tolist()
    dy = torch.randint(0, 2 * pad + 1, (B,)).tolist()
    xb = torch.stack([padded[i, :, dy[i]:dy[i] + H, dx[i]:dx[i] + W] for i in range(B)])

    brightness = (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.4 * strength
    contrast = 1 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.4 * strength
    mean = xb.mean(dim=(1, 2, 3), keepdim=True)
    xb = (xb - mean) * contrast + mean + brightness
    return xb.clamp(-1, 1)


# ---------------------------------------------------------------------------
# 3. Configurare si constructia modelului
# ---------------------------------------------------------------------------

# Configurarea de baza. Fiecare experiment schimba doar ce testeaza (restul ramane la fel).
DEFAULT_CONFIG = {
    "name": "baseline",
    "img_size": 64,
    # straturi convolutionale: lista cu numarul de filtre pe fiecare strat; [] = retea doar fully connected
    "conv_layers": [32, 64, 128],
    "kernel_size": 3,
    "stride": 1,
    "pooling": "max",        # "max", "avg" sau None (pooling 2x2 dupa fiecare strat conv)
    # strat recurent optional dupa partea conv: None, "lstm" sau "gru" (citeste imaginea rand cu rand)
    "recurrent": None,
    "rnn_hidden": 128,
    # straturi fully connected ascunse: lista cu numarul de neuroni; [] = direct stratul de iesire
    "fc_layers": [128],
    "activation": "relu",    # "relu", "leaky_relu", "tanh", "gelu", "sigmoid"
    "batch_norm": False,
    "dropout": 0.0,          # probabilitatea de dropout (0 = fara)
    # antrenare
    "augment": True,
    "optimizer": "adam",     # "adam", "adamw" sau "sgd"
    "lr": 1e-3,
    "weight_decay": 0.0,     # regularizare L2 (prin optimizer)
    "l1": 0.0,               # regularizare L1 (adaugata manual la loss)
    "batch_size": 64,
    "epochs": 30,
    "patience": 5,           # early stopping: oprim daca F1 de validare nu creste 5 epoci la rand
}


def make_config(**changes):
    """Copie a DEFAULT_CONFIG cu modificarile date. Ex: make_config(name="fc_256", conv_layers=[], fc_layers=[256])."""
    unknown = set(changes) - set(DEFAULT_CONFIG)
    if unknown:
        raise KeyError(f"Chei necunoscute in config: {unknown}")
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(changes)
    return cfg


def _activation(name):
    return {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
        "sigmoid": nn.Sigmoid(),
    }[name]


class FlowerNet(nn.Module):
    """
    Reteaua construita din config, in 3 parti (fiecare poate lipsi):
      1. features:  [Conv -> (BatchNorm) -> activare -> (pooling)] x len(conv_layers)
      2. recurent:  LSTM/GRU care citeste harta de trasaturi rand cu rand (secventa = randurile imaginii)
      3. classifier: [Linear -> (BatchNorm) -> activare -> (Dropout)] x len(fc_layers) -> Linear(5)
    """

    def __init__(self, cfg):
        super().__init__()
        layers, ch = [], 3
        for out_ch in cfg["conv_layers"]:
            layers.append(nn.Conv2d(ch, out_ch, cfg["kernel_size"], stride=cfg["stride"],
                                    padding=cfg["kernel_size"] // 2))
            if cfg["batch_norm"]:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(_activation(cfg["activation"]))
            if cfg["pooling"] == "max":
                layers.append(nn.MaxPool2d(2))
            elif cfg["pooling"] == "avg":
                layers.append(nn.AvgPool2d(2))
            ch = out_ch
        self.features = nn.Sequential(*layers)

        # Aflam forma iesirii din partea conv trecand o imagine "goala" prin ea
        with torch.no_grad():
            f = self.features(torch.zeros(1, 3, cfg["img_size"], cfg["img_size"]))
        _, C, H, W = f.shape

        self.rnn = None
        if cfg["recurrent"]:
            rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cfg["recurrent"]]
            self.rnn = rnn_cls(input_size=C * W, hidden_size=cfg["rnn_hidden"], batch_first=True)
            n_in = cfg["rnn_hidden"]
        else:
            n_in = C * H * W

        fc = []
        for n in cfg["fc_layers"]:
            fc.append(nn.Linear(n_in, n))
            if cfg["batch_norm"]:
                fc.append(nn.BatchNorm1d(n))
            fc.append(_activation(cfg["activation"]))
            if cfg["dropout"] > 0:
                fc.append(nn.Dropout(cfg["dropout"]))
            n_in = n
        fc.append(nn.Linear(n_in, len(CLASSES)))
        self.classifier = nn.Sequential(*fc)

    def forward(self, x):
        x = self.features(x)
        if self.rnn is not None:
            B, C, H, W = x.shape
            seq = x.permute(0, 2, 1, 3).reshape(B, H, C * W)  # secventa de H randuri, fiecare cu C*W valori
            out, _ = self.rnn(seq)
            x = out[:, -1]                                     # starea de la ultimul rand
        else:
            x = x.flatten(1)
        return self.classifier(x)


def build_model(cfg):
    return FlowerNet(cfg)


def count_params(model):
    """Numarul de parametri antrenabili."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(cfg, depth=3):
    """Tabel cu straturile, forma tensorilor la iesirea fiecaruia si nr. de parametri (torchinfo)."""
    from torchinfo import summary
    model = build_model(cfg)
    return summary(model, input_size=(1, 3, cfg["img_size"], cfg["img_size"]), depth=depth,
                   col_names=("output_size", "num_params"), verbose=0)


# ---------------------------------------------------------------------------
# 4. Antrenare si evaluare
# ---------------------------------------------------------------------------

def _make_optimizer(cfg, model):
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
    raise ValueError(cfg["optimizer"])


@torch.no_grad()
def predict(model, X, device, batch_size=256):
    """Probabilitatile pentru fiecare clasa (N, 5). X: tensor uint8."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = to_float(X[i:i + batch_size].to(device))
        out.append(torch.softmax(model(xb), dim=1).cpu())
    return torch.cat(out).numpy()


def train_fold(cfg, X, y, train_idx, val_idx, device, fold=0, verbose=True):
    """
    Antreneaza un model pe un fold. Dupa fiecare epoca calculeaza loss si macro-F1 pe train si pe validare.
    Early stopping: pastram greutatile din epoca cu cel mai bun F1 de validare.
    Intoarce (model cu cele mai bune greutati, istoric pe epoci, timpul de antrenare in secunde).
    """
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    model = build_model(cfg).to(device)
    opt = _make_optimizer(cfg, model)
    loss_fn = nn.CrossEntropyLoss()

    Xtr, ytr = X[train_idx].to(device), y[train_idx].to(device)
    Xva, yva = X[val_idx], y[val_idx].numpy()

    history, best_f1, best_state, bad_epochs = [], -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        tot_loss, preds, trues = 0.0, [], []
        for i in range(0, len(perm), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            if len(idx) < 2:          # BatchNorm nu merge cu un batch de o singura imagine
                continue
            xb, yb = to_float(Xtr[idx]), ytr[idx]
            if cfg["augment"]:
                xb = augment(xb)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if cfg["l1"] > 0:
                loss = loss + cfg["l1"] * sum(p.abs().sum() for n, p in model.named_parameters() if "weight" in n)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            preds.append(logits.argmax(1).cpu())
            trues.append(yb.cpu())
        tr_pred, tr_true = torch.cat(preds).numpy(), torch.cat(trues).numpy()

        proba = predict(model, Xva, device)
        va_loss = nn.functional.nll_loss(torch.log(torch.tensor(proba) + 1e-9), torch.tensor(yva)).item()
        va_f1 = f1_score(yva, proba.argmax(1), average="macro")
        history.append({
            "fold": fold, "epoch": epoch,
            "train_loss": tot_loss / len(tr_true), "val_loss": va_loss,
            "train_f1": f1_score(tr_true, tr_pred, average="macro"), "val_f1": va_f1,
            "seconds": time.time() - t0,
        })
        if verbose:
            h = history[-1]
            print(f"  fold {fold} epoca {epoch:2d}: loss {h['train_loss']:.3f}/{h['val_loss']:.3f}  "
                  f"F1 {h['train_f1']:.3f}/{h['val_f1']:.3f}  ({h['seconds']:.0f}s)")

        if va_f1 > best_f1:
            best_f1, bad_epochs = va_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                if verbose:
                    print(f"  early stopping (cel mai bun F1 validare: {best_f1:.3f})")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), time.time() - t0


def _run_dir(name):
    return OUT_DIR / "runs" / name


def run_experiment(cfg, folds=None, device=None, verbose=True):
    """
    Ruleaza un experiment complet (cross-validation pe cele 5 fold-uri) si salveaza tot:
      results/lab4/runs/<name>/config.json      - configurarea
      results/lab4/runs/<name>/fold<k>.json     - rezultatele fold-ului k (F1, acuratete, timp, epoci)
      results/lab4/runs/<name>/history.csv      - loss si F1 pe fiecare epoca (pt. grafice)
      results/lab4/runs/<name>/predictions.csv  - predictia pentru fiecare imagine (cand era in validare)
      results/lab4/models/<name>.pt             - modelul din fold-ul 0 (pt. dimensiunea pe disc; nu intra in git)
    Cand toate 5 fold-urile sunt gata, experimentul apare in tabelul comun load_log().
    Fold-urile deja terminate nu se mai ruleaza (se pot relua experimentele dupa o intrerupere).

    folds:  lista de fold-uri de rulat (implicit toate 5). Ex: folds=[0] pentru un test rapid;
            in acest caz experimentul NU apare in load_log() (rezultatul nu e complet).
    device: None = automat (GPU daca exista), sau "cpu" / "cuda".
    """
    cfg = make_config(**cfg)
    name = cfg["name"]
    device = get_device(device)
    run_dir = _run_dir(name)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        if json.loads(cfg_path.read_text()) != cfg:
            raise ValueError(f"Exista deja un experiment '{name}' cu alta configurare. Alege alt nume.")
    else:
        cfg_path.write_text(json.dumps(cfg, indent=2))

    df, X, y = load_dataset(cfg["img_size"])
    fold_of = get_folds(df)
    folds = list(range(N_FOLDS)) if folds is None else list(folds)

    for k in folds:
        fold_path = run_dir / f"fold{k}.json"
        if fold_path.exists():
            if verbose:
                print(f"[{name}] fold {k}: deja rulat, il sar.")
            continue
        if verbose:
            print(f"[{name}] fold {k} pe {device} ...")
        train_idx = np.where(fold_of != k)[0]
        val_idx = np.where(fold_of == k)[0]
        model, hist, secs = train_fold(cfg, X, y, train_idx, val_idx, device, fold=k, verbose=verbose)

        proba = predict(model, X[val_idx], device)
        pred = proba.argmax(1)
        true = y[val_idx].numpy()
        best = hist.loc[hist["val_f1"].idxmax()]

        # timpul de predictie (inferenta) per imagine, pe acelasi device
        t0 = time.time()
        predict(model, X[val_idx], device)
        infer_ms = (time.time() - t0) / len(val_idx) * 1000

        model_mb = None
        if k == 0:
            mpath = OUT_DIR / "models" / f"{name}.pt"
            mpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), mpath)
            model_mb = mpath.stat().st_size / 1024 ** 2

        hist.to_csv(run_dir / f"history_fold{k}.csv", index=False)
        pd.DataFrame({
            "path": df["path"].values[val_idx], "fold": k, "true": true, "pred": pred,
            **{f"p_{c}": proba[:, j] for j, c in enumerate(CLASSES)},
        }).to_csv(run_dir / f"predictions_fold{k}.csv", index=False)
        fold_path.write_text(json.dumps({
            "fold": k, "val_f1": f1_score(true, pred, average="macro"),
            "val_acc": accuracy_score(true, pred), "train_f1": float(best["train_f1"]),
            "best_epoch": int(best["epoch"]), "epochs_run": int(hist["epoch"].max()),
            "train_seconds": secs, "infer_ms_per_image": infer_ms,
            "n_params": count_params(model), "model_mb": model_mb, "device": str(device),
        }, indent=2))
        if verbose:
            print(f"[{name}] fold {k}: macro-F1 validare = {f1_score(true, pred, average='macro'):.3f}")

    done = sorted(run_dir.glob("fold*.json"))
    if len(done) == N_FOLDS:
        _merge_run_files(name)
        row = _summary(name)
        print(f"[{name}] gata: macro-F1 = {row['val_f1_mean']:.3f} ± {row['val_f1_std']:.3f}")
        return row
    print(f"[{name}] {len(done)}/{N_FOLDS} fold-uri terminate (rezultatul nu apare inca in load_log()).")
    return None


def _merge_run_files(name):
    """Uneste fisierele pe fold-uri in history.csv si predictions.csv."""
    run_dir = _run_dir(name)
    for kind in ["history", "predictions"]:
        if (run_dir / f"{kind}.csv").exists():
            continue
        files = [run_dir / f"{kind}_fold{k}.csv" for k in range(N_FOLDS)]
        pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(run_dir / f"{kind}.csv", index=False)
        for f in files:
            f.unlink()


def _summary(name):
    """Rezumatul unui experiment terminat (o linie): media pe fold-uri + configurarea."""
    run_dir = _run_dir(name)
    cfg = json.loads((run_dir / "config.json").read_text())
    res = pd.DataFrame([json.loads((run_dir / f"fold{k}.json").read_text()) for k in range(N_FOLDS)])
    row = {
        "name": name,
        "val_f1_mean": res["val_f1"].mean(), "val_f1_std": res["val_f1"].std(),
        "val_acc_mean": res["val_acc"].mean(), "train_f1_mean": res["train_f1"].mean(),
        "best_epoch_mean": res["best_epoch"].mean(),
        "train_minutes_total": res["train_seconds"].sum() / 60,
        "infer_ms_per_image": res["infer_ms_per_image"].mean(),
        "n_params": int(res["n_params"].iloc[0]), "model_mb": res["model_mb"].iloc[0],
        "device": res["device"].iloc[0],
        **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in cfg.items() if k != "name"},
    }
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


# ---------------------------------------------------------------------------
# 5. Citirea rezultatelor si grafice
# ---------------------------------------------------------------------------

def load_log():
    """
    Tabelul comun cu toate experimentele terminate (o linie pe experiment), sortat dupa macro-F1.
    Se construieste de fiecare data din folderele results/lab4/runs/, deci contine si experimentele
    colegilor, dupa ce le aducem din git (merge / pull).
    """
    runs = OUT_DIR / "runs"
    names = [d.name for d in sorted(runs.glob("*"))
             if d.is_dir() and all((d / f"fold{k}.json").exists() for k in range(N_FOLDS))]
    if not names:
        return pd.DataFrame()
    for n in names:
        _merge_run_files(n)
    log = pd.DataFrame([_summary(n) for n in names])
    return log.sort_values("val_f1_mean", ascending=False, ignore_index=True)


def load_history(name):
    return pd.read_csv(_run_dir(name) / "history.csv")


def load_predictions(name):
    return pd.read_csv(_run_dir(name) / "predictions.csv")


def plot_history(name, ax=None):
    """Loss si macro-F1 (media pe fold-uri) pe fiecare epoca: train vs validare."""
    import matplotlib.pyplot as plt
    h = load_history(name).groupby("epoch")[["train_loss", "val_loss", "train_f1", "val_f1"]].mean()
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(h.index, h["train_loss"], label="train")
    ax[0].plot(h.index, h["val_loss"], label="validare")
    ax[0].set(title=f"{name} - loss", xlabel="epoca", ylabel="cross-entropy")
    ax[1].plot(h.index, h["train_f1"], label="train")
    ax[1].plot(h.index, h["val_f1"], label="validare")
    ax[1].set(title=f"{name} - macro-F1", xlabel="epoca", ylabel="macro-F1")
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    plt.tight_layout()
    return ax


def plot_confusion(name, ax=None, normalize=True):
    """Matricea de confuzie din predictiile de validare (toate 5 fold-urile = toate imaginile)."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    p = load_predictions(name)
    cm = confusion_matrix(p["true"], p["pred"], normalize="true" if normalize else None)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set(title=f"{name} - matrice de confuzie", xlabel="prezis", ylabel="real")
    plt.tight_layout()
    return ax


def plot_compare(names=None, ax=None):
    """Bare cu macro-F1 (media ± abaterea pe fold-uri) pentru experimentele date (implicit toate)."""
    import matplotlib.pyplot as plt
    log = load_log()
    if names is not None:
        log = log.set_index("name").loc[names].reset_index()
    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, 0.7 * len(log)), 4))
    ax.bar(log["name"], log["val_f1_mean"], yerr=log["val_f1_std"], capsize=4, color="#4C72B0")
    ax.set(ylabel="macro-F1 validare", title="Comparatie experimente")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return ax


def show_images(paths, titles=None, ncols=6, size=2.2):
    """Afiseaza imaginile (path relativ la DATA_DIR), optional cu titluri."""
    import matplotlib.pyplot as plt
    nrows = int(np.ceil(len(paths) / ncols))
    _, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < len(paths):
            img = Image.open(DATA_DIR / paths[i]).convert("RGB")
            img.thumbnail((200, 200))   # micsoram doar pentru afisare (notebook-ul ramane mic)
            ax.imshow(img)
            if titles is not None:
                ax.set_title(titles[i], fontsize=8)
    plt.tight_layout()
