# dataset3D_solohigado.py
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


def _strip_ext(fname: str) -> str:
    if fname.endswith(".nii.gz"):
        return fname[:-7]
    if fname.endswith(".nii"):
        return fname[:-4]
    return fname


def _patient_root(pid: str) -> str:
    """
    Devuelve el ID de paciente base quitando SOLO el sufijo de repetición,
    sin romper el número de paciente.

    Convención esperada:
      - 1 solo archivo: hepaticvessel_001          -> hepaticvessel_001
      - múltiples:      hepaticvessel_001_1        -> hepaticvessel_001
                       hepaticvessel_001_2        -> hepaticvessel_001
                       hepaticvessel_001_3        -> hepaticvessel_001

    También soporta:
      - ABC-2 -> ABC
    """
    m = re.match(r"^(.+_\d+)_\d+$", pid)  # (base_con_numPaciente)_(rep)
    if m:
        return m.group(1)

    pid = re.sub(r"-\d+$", "", pid)
    return pid


def _pid_from_ct_filename(ct_name: str) -> str:
    """
    De CT filename con ventaneo a pid base (sin ventaneo, sin extensión).

    Ej:
      hepaticvessel_001_ventaneo_1.nii -> hepaticvessel_001_1
      hepaticvessel_141_ventaneo.nii   -> hepaticvessel_141
      hepaticvessel_141_ventaneo_2.nii -> hepaticvessel_141_2
    """
    stem = _strip_ext(ct_name)

    # manejar primero "_ventaneo_" para no dejar "__"
    stem = stem.replace("_ventaneo_", "_")
    stem = stem.replace("_ventaneo", "")

    # por si alguien guardó "ventaneo" sin underscore (raro, pero)
    stem = stem.replace("ventaneo_", "_").replace("ventaneo", "")

    stem = re.sub(r"__+", "_", stem).strip("_-")
    return stem


class Couinaud3DDataset(Dataset):
    """
    CT viene de ct_dir: contiene 'ventaneo' en el nombre
    Máscara viene de mask_dir: mismo nombre pero sin 'ventaneo'
    """

    def __init__(self, ct_dir: str, mask_dir: str):
        self.ct_dir = Path(ct_dir)
        self.mask_dir = Path(mask_dir)

        ct_files = sorted([
            f for f in os.listdir(self.ct_dir)
            if (f.endswith(".nii") or f.endswith(".nii.gz")) and ("ventaneo" in f.lower())
        ])

        self.samples = []
        for ct in ct_files:
            pid = _pid_from_ct_filename(ct)
            patient_id = _patient_root(pid)

            # máscara: mismo pid con .nii o .nii.gz
            mask1 = self.mask_dir / f"{pid}.nii"
            mask2 = self.mask_dir / f"{pid}.nii.gz"

            if mask1.exists():
                ms_name = mask1.name
            elif mask2.exists():
                ms_name = mask2.name
            else:
                print(f"[WARN] máscara no encontrada para {ct} -> esperaba {pid}.nii o {pid}.nii.gz")
                continue

            self.samples.append({
                "ct_name": ct,
                "ms_name": ms_name,
                "pid": pid,
                "patient_id": patient_id,
            })

        print(f"[Couinaud3DDataset] CT dir:   {self.ct_dir}")
        print(f"[Couinaud3DDataset] MASK dir: {self.mask_dir}")
        print(f"[Couinaud3DDataset] CT detectadas: {len(ct_files)}")
        print(f"[Couinaud3DDataset] muestras (pares): {len(self.samples)}")

        if len(self.samples) == 0 and len(ct_files) > 0:
            print("[ERROR] Se detectaron CT pero no matchearon máscaras. Ej CT:", ct_files[:5])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ct_path = self.ct_dir / s["ct_name"]
        ms_path = self.mask_dir / s["ms_name"]

        ct = nib.load(str(ct_path)).get_fdata().astype(np.float32)  # [H,W,D]
        ms = nib.load(str(ms_path)).get_fdata().astype(np.int16)    # [H,W,D]

        # [H,W,D] -> [D,H,W]
        ct = np.moveaxis(ct, -1, 0)
        ms = np.moveaxis(ms, -1, 0)

        # canal
        ct = ct[None, ...]  # [1,D,H,W]

        return {
            "image": torch.from_numpy(ct).float(),
            "mask": torch.from_numpy(ms).long(),
            "pid": s["pid"],
            "patient_id": s["patient_id"],
        }