# main_liver_binary_3D.py
import os
import time
from pathlib import Path
import csv
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset3D_multi_binary_splitcorregido import Couinaud3DDataset
from unet_multiclass_3D import UNet3D

# =========================
# CONFIG
# =========================
DATA_DIR = "/home/lvelez/PI_Velez/data/32slices"

RUN_NAME = os.environ.get("SLURM_JOB_NAME", "Liver_UNet3D_32slices_v2_BCE_DiceES")
RUN_DIR = Path("EXPERIMENTOS") / RUN_NAME

BATCH_SIZE = 1
EPOCHS = 300
LR = 1e-4
BASE_FILTERS = 16
VAL_SPLIT = 0.1
SEED = 42

PATIENCE = 30
MIN_DELTA = 0.0

USE_AMP = True
RESUME_FROM_LAST = True

USE_POS_WEIGHT = True
THR = 0.5

# estabilización numérica
MAX_POS_WEIGHT = 50.0          # cap para evitar bombas con FP16
GRAD_CLIP_NORM = 1.0           # clipping para AMP
DEBUG_FINITE = True            # check NaN/Inf con pid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Debug opcional desde shell:
# DEBUG_CUDA=1 CUDA_LAUNCH_BLOCKING=1 python3 main_liver_binary_3D.py
if os.environ.get("DEBUG_CUDA", "0") == "1":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# =========================
# REPRODUCIBILIDAD
# =========================
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

# =========================
# DIRECTORIOS
# =========================
RUN_DIR.mkdir(parents=True, exist_ok=True)

CKPT_LAST = RUN_DIR / "last.pth"
CKPT_BEST = RUN_DIR / "best.pth"
CSV_METRICS = RUN_DIR / "metrics.csv"
HPARAMS_TXT = RUN_DIR / "hparams.txt"
TIME_TXT = RUN_DIR / "training_time.txt"

# =========================
# GUARDAR HPARAMS
# =========================
with open(HPARAMS_TXT, "w") as f:
    f.write(f"RUN_NAME={RUN_NAME}\n")
    f.write(f"DATA_DIR={DATA_DIR}\n")
    f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
    f.write(f"EPOCHS={EPOCHS}\n")
    f.write(f"LR={LR}\n")
    f.write(f"BASE_FILTERS={BASE_FILTERS}\n")
    f.write(f"VAL_SPLIT={VAL_SPLIT}\n")
    f.write(f"SEED={SEED}\n")
    f.write(f"PATIENCE={PATIENCE}\n")
    f.write(f"MIN_DELTA={MIN_DELTA}\n")
    f.write(f"USE_AMP={USE_AMP}\n")
    f.write("TASK=binary_liver\n")
    f.write("LOSS=BCEWithLogitsLoss\n")
    f.write(f"USE_POS_WEIGHT={USE_POS_WEIGHT}\n")
    f.write(f"THR={THR}\n")
    f.write(f"MAX_POS_WEIGHT={MAX_POS_WEIGHT}\n")
    f.write(f"GRAD_CLIP_NORM={GRAD_CLIP_NORM}\n")

# =========================
# DATASET + SPLIT POR PACIENTE (SIN LEAKAGE)
# =========================
full_ds = Couinaud3DDataset(DATA_DIR, task="binary", return_binary_as_float=True)

if len(full_ds) == 0:
    raise RuntimeError(f"No hay samples. Revisá DATA_DIR={DATA_DIR} y el matching de máscaras.")

s0 = full_ds.samples[0]
if not (isinstance(s0, (tuple, list)) and len(s0) == 4):
    raise RuntimeError(
        "ERROR: full_ds.samples NO tiene 4 campos.\n"
        f"Recibí sample tipo={type(s0)} len={len(s0) if hasattr(s0,'__len__') else 'NA'} value={s0}\n"
        "Esto significa que NO se está usando el dataset corregido.\n"
        "Asegurate de estar importando dataset3D_multi_binary_splitcorregido.py"
    )

patient_to_indices = {}
for i, s in enumerate(full_ds.samples):
    patient_id = s[3]  # (ct, mask, pid, patient_id)
    patient_to_indices.setdefault(patient_id, []).append(i)

patients = sorted(patient_to_indices.keys())
rng = random.Random(SEED)
rng.shuffle(patients)

n_val_patients = max(1, int(len(patients) * VAL_SPLIT))
val_patients = set(patients[:n_val_patients])
train_patients = set(patients[n_val_patients:])

train_indices, val_indices = [], []
for p in patients:
    if p in val_patients:
        val_indices.extend(patient_to_indices[p])
    else:
        train_indices.extend(patient_to_indices[p])

train_ds = Subset(full_ds, train_indices)
val_ds = Subset(full_ds, val_indices)

print(
    f"[INFO] Patients total: {len(patients)} | "
    f"Train patients: {len(train_patients)} | "
    f"Val patients: {len(val_patients)}"
)
print(f"[INFO] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

train_p = set(full_ds.samples[i][3] for i in train_indices)
val_p   = set(full_ds.samples[i][3] for i in val_indices)
overlap = train_p & val_p
print(f"[CHECK] patient overlap train/val = {len(overlap)} (debería ser 0)")
if len(overlap) > 0:
    print("Ej overlap:", list(sorted(overlap))[:10])

# sanity
try:
    it = full_ds[0]
    print("[SANITY] image:", tuple(it["image"].shape),
          "mask:", tuple(it["mask"].shape),
          "mask dtype:", it["mask"].dtype,
          "mask unique:", torch.unique(it["mask"])[:5].tolist())
    print("[SANITY] pid:", it["pid"], "| patient_id:", it["patient_id"])
except Exception as e:
    print("[SANITY WARN]", e)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
)

# =========================
# POS_WEIGHT (opcional)
# =========================
def compute_pos_weight_from_subset(subset: Subset):
    pos = 0
    neg = 0
    for i in range(len(subset)):
        y = subset[i]["mask"]  # [Z,H,W] float 0/1
        y = y.reshape(-1)
        pos += int((y > 0.5).sum().item())
        neg += int((y <= 0.5).sum().item())
    pos = max(1, pos)
    neg = max(1, neg)
    return float(neg) / float(pos), pos, neg

pos_weight_tensor = None
pos_w_value = None

if USE_POS_WEIGHT:
    pos_w_value, pos_count, neg_count = compute_pos_weight_from_subset(train_ds)
    raw = pos_w_value
    pos_w_value = float(min(pos_w_value, MAX_POS_WEIGHT))
    pos_weight_tensor = torch.tensor([pos_w_value], dtype=torch.float32, device=DEVICE)
    print(f"[INFO] pos/neg voxels (train): pos={pos_count} neg={neg_count}")
    print(f"[INFO] pos_weight raw={raw:.4f} | capped={pos_w_value:.4f}")

# =========================
# MODEL + AMP
# =========================
model = UNet3D(
    in_channels=1,
    num_classes=1,
    base_filters=BASE_FILTERS,
    norm="in",
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

from torch.amp import GradScaler, autocast
scaler = GradScaler("cuda", enabled=(USE_AMP and DEVICE == "cuda"))

# =========================
# METRICAS 3D BINARIAS
# =========================
@torch.no_grad()
def metrics_binary_3d_from_logits(logits, y_b1zhw, thr=0.5, eps=1e-8):
    prob = torch.sigmoid(logits)
    pred = (prob >= thr).to(torch.int64)
    gt   = (y_b1zhw >= 0.5).to(torch.int64)

    pred_sum = int(pred.sum().item())
    gt_sum   = int(gt.sum().item())
    inter    = int((pred & gt).sum().item())
    union    = int((pred | gt).sum().item())

    denom = pred_sum + gt_sum
    dice = 1.0 if denom == 0 else (2.0 * inter) / (denom + eps)
    iou  = 1.0 if union == 0 else inter / (union + eps)

    return float(dice), float(iou), gt_sum, pred_sum, inter, union

# =========================
# EARLY STOP STATE
# =========================
best_val_dice_global = -1.0
early_counter = 0
start_epoch = 1

# =========================
# RESUME
# =========================
if RESUME_FROM_LAST and CKPT_LAST.exists():
    print(f"[RESUME] Cargando checkpoint: {CKPT_LAST}")
    ckpt = torch.load(CKPT_LAST, map_location=DEVICE, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None and (USE_AMP and DEVICE == "cuda"):
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception as e:
            print(f"[WARN] No pude cargar scaler_state_dict: {e}")

    best_val_dice_global = ckpt.get("best_val_dice_global", -1.0)
    early_counter = ckpt.get("early_counter", 0)
    start_epoch = ckpt["epoch"] + 1

    print(
        f"[RESUME] Continúo desde epoch {start_epoch} | "
        f"best_val_dice_global={best_val_dice_global:.6f} | "
        f"early_counter={early_counter}"
    )

# =========================
# CSV HEADER
# =========================
if not CSV_METRICS.exists():
    with open(CSV_METRICS, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_dice_macro",
            "val_dice_global",
            "val_iou_macro",
            "val_iou_global",
            "used_batches",
            "gt_empty_batches",
            "dice_zero_batches",
            "best_val_dice_global",
            "early_counter",
            "epoch_time_sec",
        ])

# =========================
# TRAIN LOOP
# =========================
t_start = time.time()

for epoch in range(start_epoch, EPOCHS + 1):
    epoch_t0 = time.time()

    # -------- TRAIN --------
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        x = batch["image"].to(DEVICE, non_blocking=True)                  # [B,1,Z,H,W]
        y = batch["mask"].to(DEVICE, non_blocking=True).unsqueeze(1)      # [B,1,Z,H,W]
        pid = batch["pid"][0] if isinstance(batch["pid"], (list, tuple)) else batch["pid"]

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP and DEVICE == "cuda":
            with autocast("cuda", enabled=True):
                logits = model(x)
                loss = criterion(logits, y)

            if DEBUG_FINITE and (not torch.isfinite(loss)):
                print(f"[NAN/INF] loss no finita en pid={pid} | loss={loss.item()}")
                print("x[min,max]=", float(x.min()), float(x.max()), " y[sum]=", float(y.sum()))
                raise RuntimeError("Loss NaN/Inf")

            scaler.scale(loss).backward()

            # ---- CLIP ----
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)

            if DEBUG_FINITE and (not torch.isfinite(loss)):
                print(f"[NAN/INF] loss no finita en pid={pid} | loss={loss.item()}")
                print("x[min,max]=", float(x.min()), float(x.max()), " y[sum]=", float(y.sum()))
                raise RuntimeError("Loss NaN/Inf")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

        train_loss += float(loss.item())

    train_loss /= max(1, len(train_loader))

    # -------- VALID --------
    model.eval()
    val_loss = 0.0

    dice_sum = 0.0
    iou_sum  = 0.0
    count    = 0

    g_inter = 0
    g_pred  = 0
    g_gt    = 0
    g_union = 0

    gt_empty_pids = []
    dice_zero_pids = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = batch["mask"].to(DEVICE, non_blocking=True).unsqueeze(1)
            pid = batch["pid"][0] if isinstance(batch["pid"], (list, tuple)) else batch["pid"]

            if USE_AMP and DEVICE == "cuda":
                with autocast("cuda", enabled=True):
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)

            val_loss += float(loss.item())

            dice_b, iou_b, gt_sum, pred_sum, inter, union = metrics_binary_3d_from_logits(
                logits, y, thr=THR
            )

            if gt_sum == 0:
                gt_empty_pids.append(str(pid))
                continue

            if dice_b == 0.0:
                dice_zero_pids.append(str(pid))

            dice_sum += dice_b
            iou_sum  += iou_b
            count    += 1

            g_inter += inter
            g_pred  += pred_sum
            g_gt    += gt_sum
            g_union += union

    val_loss /= max(1, len(val_loader))

    val_dice_macro = (dice_sum / count) if count > 0 else 0.0
    val_iou_macro  = (iou_sum  / count) if count > 0 else 0.0
    val_dice_global = (2.0 * g_inter) / (g_pred + g_gt + 1e-8) if (g_pred + g_gt) > 0 else 0.0
    val_iou_global  = (g_inter) / (g_union + 1e-8) if g_union > 0 else 0.0

    # -------- EARLY STOP --------
    improved = (val_dice_global - best_val_dice_global) > MIN_DELTA
    if improved:
        best_val_dice_global = val_dice_global
        early_counter = 0
    else:
        early_counter += 1

    epoch_time = time.time() - epoch_t0

    print(
        f"[Epoch {epoch:03d}] "
        f"TrainLoss={train_loss:.6f} | ValLoss={val_loss:.6f} | "
        f"DiceMacro={val_dice_macro:.6f} | DiceGlobal={val_dice_global:.6f} | "
        f"IoUMacro={val_iou_macro:.6f} | IoUGlobal={val_iou_global:.6f} | "
        f"Used={count} | GTEmpty={len(gt_empty_pids)} | Dice0={len(dice_zero_pids)} | "
        f"BestDiceGlobal={best_val_dice_global:.6f} | Early={early_counter}/{PATIENCE} | "
        f"Time={epoch_time:.1f}s"
    )

    ckpt_payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if (USE_AMP and DEVICE == "cuda") else None,
        "best_val_dice_global": float(best_val_dice_global),
        "early_counter": int(early_counter),
        "loss": "BCEWithLogitsLoss",
        "pos_weight": float(pos_w_value) if pos_w_value is not None else None,
        "thr": float(THR),
        "val_debug": {
            "gt_empty_pids": gt_empty_pids,
            "dice_zero_pids": dice_zero_pids,
        }
    }

    torch.save(ckpt_payload, CKPT_LAST)

    if improved:
        torch.save(ckpt_payload, CKPT_BEST)
        print(f" ✔ Nuevo mejor ValDiceGlobal: {best_val_dice_global:.6f}")

    with open(CSV_METRICS, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            train_loss,
            val_loss,
            val_dice_macro,
            val_dice_global,
            val_iou_macro,
            val_iou_global,
            count,
            len(gt_empty_pids),
            len(dice_zero_pids),
            best_val_dice_global,
            early_counter,
            epoch_time,
        ])

    if early_counter >= PATIENCE:
        print(f"[EARLY STOP] No mejora en ValDiceGlobal por {PATIENCE} epochs. Corto en epoch {epoch}.")
        break

# =========================
# FIN
# =========================
total_time = time.time() - t_start

with open(TIME_TXT, "w") as f:
    f.write(f"Tiempo total entrenamiento (s): {total_time:.1f}\n")
    f.write(f"Tiempo total entrenamiento (h): {total_time / 3600:.2f}\n")

print("Entrenamiento finalizado.")