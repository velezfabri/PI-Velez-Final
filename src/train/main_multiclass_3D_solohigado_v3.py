# main_multiclass_3D_solohigado_v2.py
import os
import time
from pathlib import Path
import csv
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset3D_solohigado_splitcorregido import Couinaud3DDataset
from unet_multiclass_3D import UNet3D

# =========================
# CONFIG
# =========================
CT_DIR = "/home/lvelez/PI_Velez/data/32slices/ct_higado_solo"
MASK_DIR = "/home/lvelez/PI_Velez/data/32slices"

RUN_NAME = os.environ.get("SLURM_JOB_NAME", "Couinaud_UNet3D_solohigado_v2_DiceES")
RUN_DIR = Path("EXPERIMENTOS") / RUN_NAME

BATCH_SIZE = 1
EPOCHS = 300
LR = 1e-4
NUM_CLASSES = 9
BASE_FILTERS = 16
VAL_SPLIT = 0.1
SEED = 42

PATIENCE = 30
MIN_DELTA = 0.0

USE_AMP = True
RESUME_FROM_LAST = True

LAMBDA_CE = 0.5
LAMBDA_DICE = 0.5
BACKGROUND_WEIGHT_SCALE = 0.05

# estabilización numérica
GRAD_CLIP_NORM = 1.0
DEBUG_FINITE = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    f.write(f"CT_DIR={CT_DIR}\n")
    f.write(f"MASK_DIR={MASK_DIR}\n")
    f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
    f.write(f"EPOCHS={EPOCHS}\n")
    f.write(f"LR={LR}\n")
    f.write(f"NUM_CLASSES={NUM_CLASSES}\n")
    f.write(f"BASE_FILTERS={BASE_FILTERS}\n")
    f.write(f"VAL_SPLIT={VAL_SPLIT}\n")
    f.write(f"SEED={SEED}\n")
    f.write(f"PATIENCE={PATIENCE}\n")
    f.write(f"MIN_DELTA={MIN_DELTA}\n")
    f.write(f"USE_AMP={USE_AMP}\n")
    f.write(f"LAMBDA_CE={LAMBDA_CE}\n")
    f.write(f"LAMBDA_DICE={LAMBDA_DICE}\n")
    f.write(f"BACKGROUND_WEIGHT_SCALE={BACKGROUND_WEIGHT_SCALE}\n")
    f.write("VAL_METRIC=binary_liver_fg>0 (dice macro/global + iou macro/global)\n")
    f.write("EARLY_STOP=val_dice_global\n")
    f.write(f"GRAD_CLIP_NORM={GRAD_CLIP_NORM}\n")

# =========================
# helpers: soporta samples dict o tuple
# =========================
def sample_patient_id(s):
    if isinstance(s, dict):
        return s["patient_id"]
    return s[3]  # (ct, mask, pid, patient_id)

def sample_pid(s):
    if isinstance(s, dict):
        return s["pid"]
    return s[2]

# =========================
# DATASET + SPLIT POR PACIENTE
# =========================
full_ds = Couinaud3DDataset(CT_DIR, MASK_DIR)

patient_to_indices = {}
for i, s in enumerate(full_ds.samples):
    pr = sample_patient_id(s)
    patient_to_indices.setdefault(pr, []).append(i)

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

assert len(train_patients & val_patients) == 0, "Leakage: hay pacientes en train y val a la vez."

train_ds = Subset(full_ds, train_indices)
val_ds = Subset(full_ds, val_indices)

train_pids = {sample_patient_id(full_ds.samples[i]) for i in train_indices}
val_pids   = {sample_patient_id(full_ds.samples[i]) for i in val_indices}
assert len(train_pids & val_pids) == 0, "Leakage: se mezclaron volúmenes de un mismo paciente entre splits."

print(
    f"[INFO] Patients total: {len(patients)} | "
    f"Train patients: {len(train_patients)} | "
    f"Val patients: {len(val_patients)}"
)
print(f"[INFO] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

# sanity
try:
    item0 = full_ds[0]
    print("[SANITY] image:", tuple(item0["image"].shape), "mask:", tuple(item0["mask"].shape))
    print("[SANITY] pid:", item0["pid"], "| patient_id:", item0["patient_id"])
    print("[SANITY] mask unique (muestra):", torch.unique(item0["mask"])[:15].tolist())
except Exception as e:
    print("[SANITY WARN] No pude leer el primer sample:", e)

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
# LOSS: pesos por clase desde TRAIN
# =========================
def compute_class_weights_from_subset(subset: Subset, num_classes: int, bg_scale: float = 0.05):
    counts = torch.zeros(num_classes, dtype=torch.long)
    for i in range(len(subset)):
        item = subset[i]
        y = item["mask"].reshape(-1)
        bc = torch.bincount(y.cpu(), minlength=num_classes)
        counts += bc

    counts = counts.clamp(min=1)
    freq = counts.float() / counts.sum().float()

    w = 1.0 / torch.sqrt(freq)
    w = w / w.mean()
    w[0] = w[0] * float(bg_scale)
    return w.float(), counts

class GeneralizedDiceLoss(nn.Module):
    def __init__(self, num_classes: int, exclude_bg: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.exclude_bg = exclude_bg
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)

        target_1h = F.one_hot(target, num_classes=self.num_classes)
        target_1h = target_1h.permute(0, 4, 1, 2, 3).float()

        if self.exclude_bg:
            probs = probs[:, 1:, ...]
            target_1h = target_1h[:, 1:, ...]

        dims = (0, 2, 3, 4)
        g_c = target_1h.sum(dims)
        w_c = 1.0 / (g_c * g_c + self.eps)

        intersect = (probs * target_1h).sum(dims)
        denom = (probs + target_1h).sum(dims)

        dice_per_class = (2.0 * intersect + self.eps) / (denom + self.eps)
        dice = (w_c * dice_per_class).sum() / (w_c.sum() + self.eps)

        return 1.0 - dice

ce_weights, class_counts = compute_class_weights_from_subset(
    train_ds, NUM_CLASSES, bg_scale=BACKGROUND_WEIGHT_SCALE
)
print("[INFO] Class counts (train):", class_counts.tolist())
print("[INFO] CE weights:", [float(x) for x in ce_weights])

# =========================
# MODEL + AMP
# =========================
model = UNet3D(
    in_channels=1,
    num_classes=NUM_CLASSES,
    base_filters=BASE_FILTERS,
    norm="in",
).to(DEVICE)

criterion_ce = nn.CrossEntropyLoss(weight=ce_weights.to(DEVICE))
criterion_dice = GeneralizedDiceLoss(num_classes=NUM_CLASSES, exclude_bg=True)

def compound_loss(logits, y):
    return (LAMBDA_CE * criterion_ce(logits, y)) + (LAMBDA_DICE * criterion_dice(logits, y))

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

from torch.amp import GradScaler, autocast
scaler = GradScaler("cuda", enabled=(USE_AMP and DEVICE == "cuda"))

# =========================
# MÉTRICAS BINARIAS (hígado = label>0) EN VALID
# =========================
@torch.no_grad()
def metrics_liver_fg_from_logits_multiclass(logits, y_1zhw, eps=1e-8):
    pred_lbl = torch.argmax(logits, dim=1)  # [1,Z,H,W]
    pred_fg = (pred_lbl > 0).to(torch.int64)
    gt_fg   = (y_1zhw > 0).to(torch.int64)

    pred_sum = int(pred_fg.sum().item())
    gt_sum   = int(gt_fg.sum().item())
    inter    = int((pred_fg & gt_fg).sum().item())
    union    = int((pred_fg | gt_fg).sum().item())

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

    if "scaler_state_dict" in ckpt and (USE_AMP and DEVICE == "cuda") and ckpt["scaler_state_dict"] is not None:
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
        x = batch["image"].to(DEVICE, non_blocking=True)  # [B,1,Z,H,W]
        y = batch["mask"].to(DEVICE, non_blocking=True)   # [B,Z,H,W] long
        pid = batch["pid"][0] if isinstance(batch["pid"], (list, tuple)) else batch["pid"]

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP and DEVICE == "cuda":
            with autocast("cuda", enabled=True):
                logits = model(x)
                loss = compound_loss(logits, y)

            if DEBUG_FINITE and (not torch.isfinite(loss)):
                print(f"[NAN/INF] loss no finita en pid={pid} | loss={loss.item()}")
                raise RuntimeError("Loss NaN/Inf")

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = compound_loss(logits, y)

            if DEBUG_FINITE and (not torch.isfinite(loss)):
                print(f"[NAN/INF] loss no finita en pid={pid} | loss={loss.item()}")
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
    iou_sum = 0.0
    count = 0

    g_inter = 0
    g_pred = 0
    g_gt = 0
    g_union = 0

    gt_empty_pids = []
    dice_zero_pids = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = batch["mask"].to(DEVICE, non_blocking=True)  # [1,Z,H,W] long
            pid = batch["pid"][0] if isinstance(batch["pid"], (list, tuple)) else batch["pid"]

            if USE_AMP and DEVICE == "cuda":
                with autocast("cuda", enabled=True):
                    logits = model(x)
                    loss = compound_loss(logits, y)
            else:
                logits = model(x)
                loss = compound_loss(logits, y)

            val_loss += float(loss.item())

            dice_b, iou_b, gt_sum, pred_sum, inter, union = metrics_liver_fg_from_logits_multiclass(logits, y)

            if gt_sum == 0:
                gt_empty_pids.append(str(pid))
                continue

            if dice_b == 0.0:
                dice_zero_pids.append(str(pid))

            dice_sum += dice_b
            iou_sum += iou_b
            count += 1

            g_inter += inter
            g_pred  += pred_sum
            g_gt    += gt_sum
            g_union += union

    val_loss /= max(1, len(val_loader))

    val_dice_macro = (dice_sum / count) if count > 0 else 0.0
    val_iou_macro  = (iou_sum / count) if count > 0 else 0.0

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
        f"TrainLoss={train_loss:.6f} | "
        f"ValLoss={val_loss:.6f} | "
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
        "ce_weights": ce_weights.cpu(),
        "class_counts": class_counts.cpu(),
        "loss_cfg": {
            "LAMBDA_CE": LAMBDA_CE,
            "LAMBDA_DICE": LAMBDA_DICE,
            "BACKGROUND_WEIGHT_SCALE": BACKGROUND_WEIGHT_SCALE,
        },
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