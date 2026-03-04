#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import time

import numpy as np
import nibabel as nib
import torch

from unet_multiclass_3D import UNet3D


# -------------------------
# Helpers: ckpt
# -------------------------
def strip_state_dict(ckpt):
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def resolve_best_ckpt(exp_dir: Path) -> Path:
    """
    exp_dir: carpeta EXPERIMENTOS/<RUN_NAME>
    devuelve exp_dir/best.pth (si no existe, cae a last.pth)
    """
    best = exp_dir / "best.pth"
    last = exp_dir / "last.pth"
    if best.exists():
        return best
    if last.exists():
        return last
    raise FileNotFoundError(f"No encontré best.pth ni last.pth en: {exp_dir}")


# -------------------------
# Windowing / normalization
# -------------------------
def apply_window_and_norm(x: np.ndarray, wl: float, wh: float, norm: str = "01") -> np.ndarray:
    """
    x: float32 HU array [D,H,W]
    clip to [wl,wh] then normalize
    norm:
      - "none": only clip
      - "01": scale to [0,1]
      - "m11": scale to [-1,1]
    """
    x = np.clip(x, wl, wh)
    if norm == "none":
        return x.astype(np.float32)
    if norm == "01":
        x = (x - wl) / (wh - wl + 1e-8)
        return x.astype(np.float32)
    if norm == "m11":
        x = (x - wl) / (wh - wl + 1e-8)
        x = x * 2.0 - 1.0
        return x.astype(np.float32)
    raise ValueError(f"norm desconocida: {norm}")


def should_window(mode: str, filename: str) -> bool:
    """
    mode:
      - "always": siempre aplica ventaneo
      - "never": nunca aplica
      - "auto": si el nombre contiene 'ventaneo' -> no aplica, si no -> aplica
    """
    mode = mode.lower()
    if mode == "always":
        return True
    if mode == "never":
        return False
    if mode == "auto":
        return ("ventaneo" not in filename.lower())
    raise ValueError(f"window_mode inválido: {mode}")


# -------------------------
# Sliding windows depth
# -------------------------
def make_depth_windows(D: int, win: int, stride: int):
    if D <= win:
        return [(0, D)]
    starts = list(range(0, D - win + 1, stride))
    if starts[-1] != D - win:
        starts.append(D - win)
    return [(s, s + win) for s in starts]


@torch.no_grad()
def infer_liver_prob_sliding(model, vol_1dhw: torch.Tensor, win: int, stride: int, amp: bool):
    """
    vol_1dhw: [1,D,H,W] float (CPU)
    devuelve prob [D,H,W] cpu float32
    """
    device = next(model.parameters()).device
    _, D, H, W = vol_1dhw.shape

    windows = make_depth_windows(D, win, stride)
    prob_acc = torch.zeros((D, H, W), dtype=torch.float32, device=device)
    cnt_acc  = torch.zeros((D, H, W), dtype=torch.float32, device=device)

    for (a, b) in windows:
        x = vol_1dhw[:, a:b].unsqueeze(0).to(device, non_blocking=True)  # [1,1,d,H,W]
        with torch.amp.autocast(device_type="cuda", enabled=(amp and device.type == "cuda")):
            logits = model(x)                  # [1,1,d,H,W]
            p = torch.sigmoid(logits)[0, 0]    # [d,H,W]
        prob_acc[a:b] += p
        cnt_acc[a:b]  += 1.0

    return (prob_acc / torch.clamp(cnt_acc, min=1.0)).detach().cpu()  # [D,H,W]


@torch.no_grad()
def infer_couinaud_logits_sliding(model, vol_1dhw: torch.Tensor, win: int, stride: int, amp: bool, num_classes: int):
    """
    vol_1dhw: [1,D,H,W] float (CPU)
    devuelve logits [C,D,H,W] cpu float32
    """
    device = next(model.parameters()).device
    _, D, H, W = vol_1dhw.shape

    windows = make_depth_windows(D, win, stride)
    log_acc = torch.zeros((num_classes, D, H, W), dtype=torch.float32, device=device)
    cnt_acc = torch.zeros((D, H, W), dtype=torch.float32, device=device)

    for (a, b) in windows:
        x = vol_1dhw[:, a:b].unsqueeze(0).to(device, non_blocking=True)  # [1,1,d,H,W]
        with torch.amp.autocast(device_type="cuda", enabled=(amp and device.type == "cuda")):
            logits = model(x)[0]  # [C,d,H,W]
        log_acc[:, a:b] += logits
        cnt_acc[a:b] += 1.0

    return (log_acc / torch.clamp(cnt_acc.unsqueeze(0), min=1.0)).detach().cpu()  # [C,D,H,W]


# -------------------------
# Save nifti like reference
# -------------------------
def save_like(ref_nii: nib.Nifti1Image, arr_hwd: np.ndarray, out_path: Path, dtype=None):
    if dtype is not None:
        arr_hwd = arr_hwd.astype(dtype)
    out = nib.Nifti1Image(arr_hwd, affine=ref_nii.affine, header=ref_nii.header)
    nib.save(out, str(out_path))


# -------------------------
# ORIENT helpers
# (usar *_orientada para inferir, pero guardar outputs en el espacio ORIGINAL)
# -------------------------
def pick_oriented_if_exists(ct_path: Path, suffix: str = "_orientada") -> Path:
    name = ct_path.name
    base = name
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]

    cand1 = ct_path.parent / f"{base}{suffix}.nii.gz"
    cand2 = ct_path.parent / f"{base}{suffix}.nii"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    return ct_path


def reorient_arr_like(src_nii: nib.Nifti1Image, arr_src_hwd: np.ndarray, ref_nii: nib.Nifti1Image) -> np.ndarray:
    """
    Reorienta arr_src_hwd (en espacio src_nii) para matchear orientación de ref_nii.
    OJO: esto SOLO cambia ejes/flip por orientación (no resamplea voxel sizes).
    """
    ornt_src = nib.orientations.io_orientation(src_nii.affine)
    ornt_ref = nib.orientations.io_orientation(ref_nii.affine)
    transform = nib.orientations.ornt_transform(ornt_src, ornt_ref)
    arr_out = nib.orientations.apply_orientation(arr_src_hwd, transform)
    return arr_out


# -------------------------
# File collection helpers
# -------------------------
def list_nifti_files(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz"))])


def collect_ct_files(folder: Path):
    """
    Si hay archivos con 'ventaneo' => tomamos SOLO esos (CTs de tu dataset).
    Si no hay ninguno con 'ventaneo' => tomamos TODOS los nifti (casos reales / nombres raros).
    """
    all_files = list_nifti_files(folder)
    vent = [p for p in all_files if "ventaneo" in p.name.lower()]
    if len(vent) > 0:
        return vent
    return all_files


def get_depth_slices(ct_path: Path) -> int:
    ref = nib.load(str(ct_path))
    arr = ref.get_fdata()
    if arr.ndim != 3:
        return -1
    return int(arr.shape[-1])  # [H,W,D]


def safe_case_folder_name(ct_path: Path) -> str:
    n = ct_path.name
    if n.lower().endswith(".nii.gz"):
        return n[:-7]
    if n.lower().endswith(".nii"):
        return n[:-4]
    return ct_path.stem


def write_times_txt(case_out_dir: Path, info: dict, status: str, err_msg: str = ""):
    """
    Guarda el tiempo y metadata en la MISMA carpeta donde se guardan outputs del caso.
    """
    time_txt = case_out_dir / "times.txt"
    with open(time_txt, "w") as f:
        f.write(f"status: {status}\n")
        if err_msg:
            f.write(f"error: {err_msg}\n")
        f.write(f"case: {info.get('case','')}\n")
        f.write(f"ct_used_for_infer: {info.get('ct_used_for_infer','')}\n")
        f.write(f"D: {info.get('D','')}\n")
        f.write(f"H: {info.get('H','')}\n")
        f.write(f"W: {info.get('W','')}\n")
        f.write(f"t_liver_s: {float(info.get('t_liver_s',0.0)):.6f}\n")
        f.write(f"t_couinaud_s: {float(info.get('t_couinaud_s',0.0)):.6f}\n")
        f.write(f"t_total_s: {float(info.get('t_total_s',0.0)):.6f}\n")


# -------------------------
# Run one case
# -------------------------
def run_one_case(ct_path: Path, out_case_dir: Path, liver_model, cou_model, args):
    out_case_dir.mkdir(parents=True, exist_ok=True)

    device = next(liver_model.parameters()).device
    t_case0 = time.time()

    # 1) ORIGINAL (para guardar outputs en su espacio)
    ref_orig = nib.load(str(ct_path))

    # 2) Para inferencia: usar *_orientada si el flag está activo y existe
    ct_infer_path = pick_oriented_if_exists(ct_path, suffix=args.oriented_suffix) if args.prefer_oriented else ct_path
    ref_infer = nib.load(str(ct_infer_path))

    ct = ref_infer.get_fdata().astype(np.float32)  # [H,W,D]
    if ct.ndim != 3:
        raise ValueError(f"{ct_infer_path.name}: esperaba 3D, shape={ct.shape}")

    ct_dhw = np.moveaxis(ct, -1, 0)  # [D,H,W]
    D, H, W = ct_dhw.shape

    # Ventaneo / normalización si corresponde
    if should_window(args.window_mode, ct_infer_path.name):
        ct_dhw = apply_window_and_norm(ct_dhw, args.wl, args.wh, norm=args.norm)
    else:
        ct_dhw = ct_dhw.astype(np.float32)

    if D < args.win:
        raise RuntimeError(f"{ct_infer_path.name}: D={D} < win={args.win} (omitido)")

    # ---- a torch [1,D,H,W]
    ct_t = torch.from_numpy(ct_dhw)[None, ...].float()

    # ======================
    # LIVER
    # ======================
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    liver_prob = infer_liver_prob_sliding(liver_model, ct_t, args.win, args.stride, args.amp).numpy()  # [D,H,W]

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_liver = time.time() - t0

    liver_mask = (liver_prob >= args.thr_liver).astype(np.uint8)  # [D,H,W]

    # Guardar liver en espacio ORIGINAL
    liver_mask_hwd_infer = np.moveaxis(liver_mask, 0, -1)  # [H,W,D] en ref_infer
    liver_mask_hwd_orig = reorient_arr_like(ref_infer, liver_mask_hwd_infer, ref_orig)
    save_like(ref_orig, liver_mask_hwd_orig, out_case_dir / "liver_mask.nii.gz", dtype=np.uint8)

    if args.save_liver_prob:
        liver_prob_hwd_infer = np.moveaxis(liver_prob, 0, -1).astype(np.float32)
        liver_prob_hwd_orig = reorient_arr_like(ref_infer, liver_prob_hwd_infer, ref_orig)
        save_like(ref_orig, liver_prob_hwd_orig, out_case_dir / "liver_prob.nii.gz", dtype=np.float32)

    # ======================
    # CT solo hígado
    # ======================
    if args.soft_gating:
        ct_masked = ct_dhw * liver_prob
    else:
        ct_masked = ct_dhw * liver_mask

    if args.save_ct_masked:
        ct_masked_hwd_infer = np.moveaxis(ct_masked, 0, -1).astype(np.float32)
        ct_masked_hwd_orig = reorient_arr_like(ref_infer, ct_masked_hwd_infer, ref_orig)
        save_like(ref_orig, ct_masked_hwd_orig, out_case_dir / "ct_masked.nii.gz", dtype=np.float32)

    # ======================
    # COUINAUD
    # ======================
    ct_masked_t = torch.from_numpy(ct_masked)[None, ...].float()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    cou_logits = infer_couinaud_logits_sliding(
        cou_model, ct_masked_t, args.win, args.stride, args.amp, args.num_classes
    ).numpy()  # [C,D,H,W]

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_cou = time.time() - t1

    cou_pred = np.argmax(cou_logits, axis=0).astype(np.uint8)  # [D,H,W]

    if args.mask_couinaud_with_liver:
        cou_pred = cou_pred * liver_mask

    # Guardar couinaud en espacio ORIGINAL
    cou_pred_hwd_infer = np.moveaxis(cou_pred, 0, -1)  # [H,W,D] en ref_infer
    cou_pred_hwd_orig = reorient_arr_like(ref_infer, cou_pred_hwd_infer, ref_orig)
    save_like(ref_orig, cou_pred_hwd_orig, out_case_dir / "couinaud_pred.nii.gz", dtype=np.uint8)

    t_total = time.time() - t_case0

    return {
        "case": ct_path.name,
        "ct_used_for_infer": str(ct_infer_path),
        "D": D, "H": H, "W": W,
        "t_liver_s": t_liver,
        "t_couinaud_s": t_cou,
        "t_total_s": t_total,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_path", required=True,
                    help="CT única (.nii/.nii.gz) o carpeta con CTs (dataset o pacientes reales).")
    ap.add_argument("--out_dir", required=True)

    # --------- Checkpoints (modo viejo, compatible) ----------
    ap.add_argument("--liver_ckpt", default=None,
                    help="Ruta directa al checkpoint del hígado (best.pth o last.pth). Si lo pasás, tiene prioridad.")
    ap.add_argument("--couinaud_ckpt", default=None,
                    help="Ruta directa al checkpoint Couinaud (best.pth o last.pth). Si lo pasás, tiene prioridad.")

    # --------- Experimentos (modo nuevo) ----------
    ap.add_argument("--liver_exp_dir", default="/home/lvelez/PI_Velez/U-Net Couinaud/3D/EXPERIMENTOS/Liver_binary_3D_v3_HP4_splitcorregido")
    ap.add_argument("--couinaud_exp_dir", default="/home/lvelez/PI_Velez/U-Net Couinaud/3D/EXPERIMENTOS/multiclass_3D_solohigadomask_v3_HP4_splitcorregido")

    # Arquitectura (tiene que coincidir con training)
    ap.add_argument("--base_filters_liver", type=int, default=16)
    ap.add_argument("--base_filters_cou", type=int, default=16)
    ap.add_argument("--num_classes", type=int, default=9)

    # Infer params
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--win", type=int, default=32)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--thr_liver", type=float, default=0.5)
    ap.add_argument("--soft_gating", action="store_true",
                    help="CT_solo_higado = CT * prob (en vez de CT * mask binaria).")
    ap.add_argument("--mask_couinaud_with_liver", action="store_true",
                    help="Pone predicción Couinaud fuera del hígado (predicho) en 0.")

    # Orientación
    ap.add_argument("--prefer_oriented", action="store_true",
                    help="Si existe *_orientada, la usa para inferir (pero guarda outputs en el espacio ORIGINAL).")
    ap.add_argument("--oriented_suffix", default="_orientada")

    # Ventaneo
    ap.add_argument("--window_mode", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--wl", type=float, default=-200.0)
    ap.add_argument("--wh", type=float, default=250.0)
    ap.add_argument("--norm", default="01", choices=["none", "01", "m11"])

    # Outputs extra
    ap.add_argument("--save_liver_prob", action="store_true")
    ap.add_argument("--save_ct_masked", action="store_true")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # --------- Resolver paths de checkpoints ----------
    if args.liver_ckpt is not None:
        liver_ckpt_path = Path(args.liver_ckpt)
        if not liver_ckpt_path.exists():
            raise FileNotFoundError(f"No existe --liver_ckpt: {liver_ckpt_path}")
    else:
        liver_exp = Path(args.liver_exp_dir)
        liver_ckpt_path = resolve_best_ckpt(liver_exp)

    if args.couinaud_ckpt is not None:
        cou_ckpt_path = Path(args.couinaud_ckpt)
        if not cou_ckpt_path.exists():
            raise FileNotFoundError(f"No existe --couinaud_ckpt: {cou_ckpt_path}")
    else:
        cou_exp = Path(args.couinaud_exp_dir)
        cou_ckpt_path = resolve_best_ckpt(cou_exp)

    print(f"[CKPT] Liver:   {liver_ckpt_path}")
    print(f"[CKPT] Couinaud:{cou_ckpt_path}")

    # --------- Load models ----------
    liver_model = UNet3D(in_channels=1, num_classes=1, base_filters=args.base_filters_liver, norm="in")
    ckpt_l = torch.load(str(liver_ckpt_path), map_location="cpu", weights_only=False)
    liver_model.load_state_dict(strip_state_dict(ckpt_l), strict=True)
    liver_model.to(device).eval()

    cou_model = UNet3D(in_channels=1, num_classes=args.num_classes, base_filters=args.base_filters_cou, norm="in")
    ckpt_c = torch.load(str(cou_ckpt_path), map_location="cpu", weights_only=False)
    cou_model.load_state_dict(strip_state_dict(ckpt_c), strict=True)
    cou_model.to(device).eval()

    in_path = Path(args.in_path)
    single_input = not in_path.is_dir()

    # Collect CTs
    if in_path.is_dir():
        ct_files = collect_ct_files(in_path)
        if len(ct_files) == 0:
            raise RuntimeError(f"No encontré NIfTI en {in_path}")
        print(f"[INFO] En carpeta: {len(ct_files)} archivos seleccionados para inferencia.")
        all_files = list_nifti_files(in_path)
        if len(all_files) != len(ct_files):
            print(f"[INFO] Se ignoraron {len(all_files) - len(ct_files)} archivos (probablemente GT sin 'ventaneo').")
    else:
        ct_files = [in_path]

    # Warm-up (solo si carpeta; para archivo único no te ensucia out_dir)
    if (not single_input) and device.type == "cuda" and len(ct_files) > 0:
        try:
            D0 = get_depth_slices(ct_files[0])
            if D0 >= args.win:
                print("[WARMUP] warm-up 1 caso...")
                _ = run_one_case(ct_files[0], out_dir / "_warmup", liver_model, cou_model, args)
                torch.cuda.synchronize()
                print("[WARMUP] OK")
            else:
                print(f"[WARMUP] Salteo: primer caso D={D0} < win={args.win}")
        except Exception as e:
            print(f"[WARMUP WARN] No pude hacer warmup: {e}")

    # Loop
    for ct_path in ct_files:
        # ✅ Si es 1 archivo: guardamos TODO directo en out_dir (incluye times.txt)
        # ✅ Si es carpeta: una subcarpeta por caso
        if single_input:
            case_out = out_dir
        else:
            case_folder = safe_case_folder_name(ct_path)
            case_out = out_dir / case_folder

        try:
            D = get_depth_slices(ct_path)
            if D < args.win:
                print(f"[SKIP] {ct_path.name}: D={D} < win={args.win}")
                info = {
                    "case": ct_path.name,
                    "ct_used_for_infer": "",
                    "D": D, "H": "", "W": "",
                    "t_liver_s": 0.0, "t_couinaud_s": 0.0, "t_total_s": 0.0,
                }
                case_out.mkdir(parents=True, exist_ok=True)
                write_times_txt(case_out, info, status="SKIP_D_LT_WIN")
                continue

            info = run_one_case(ct_path, case_out, liver_model, cou_model, args)

            print(
                f"[TIME] {info['case']} | Liver={info['t_liver_s']:.2f}s | "
                f"Couinaud={info['t_couinaud_s']:.2f}s | Total={info['t_total_s']:.2f}s | "
                f"D={info['D']} | InferCT={info['ct_used_for_infer']}"
            )

            write_times_txt(case_out, info, status="OK")
            print(f"[OK] {ct_path.name} -> {case_out}")

        except Exception as e:
            print(f"[FAIL] {ct_path.name}: {e}")
            info = {
                "case": ct_path.name,
                "ct_used_for_infer": "",
                "D": -1, "H": "", "W": "",
                "t_liver_s": 0.0, "t_couinaud_s": 0.0, "t_total_s": 0.0,
            }
            case_out.mkdir(parents=True, exist_ok=True)
            write_times_txt(case_out, info, status="FAIL", err_msg=str(e))

    print("[DONE] Listo. Se guardó times.txt junto a los outputs del caso.")


if __name__ == "__main__":
    main()