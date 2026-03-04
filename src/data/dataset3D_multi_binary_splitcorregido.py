# dataset3D_liver_binary.py
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


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
    # Si termina con _<rep> y antes también termina con _<numPaciente>
    m = re.match(r"^(.+_\d+)_\d+$", pid)  # (base_con_numPaciente)_(rep)
    if m:
        return m.group(1)

    # Soporte opcional para guión
    pid = re.sub(r"-\d+$", "", pid)
    return pid


def _pid_from_ct(ct_name: str) -> str:
    """
    Convierte nombre de CT -> pid base para buscar máscara.

    Ejemplos:
      hepaticvessel_141_ventaneo.nii     -> hepaticvessel_141
      hepaticvessel_141_ventaneo_2.nii   -> hepaticvessel_141_2
      hepaticvessel_141_ventaneo_3.nii   -> hepaticvessel_141_3
    """
    stem = ct_name[:-4] if ct_name.endswith(".nii") else ct_name

    # IMPORTANTE: manejar primero "_ventaneo_" para no dejar "__"
    stem = stem.replace("_ventaneo_", "_")
    stem = stem.replace("_ventaneo", "")

    # limpiar dobles underscores por las dudas
    stem = re.sub(r"__+", "_", stem).strip("_")
    return stem


class Couinaud3DDataset(Dataset):
    """
    Espera en data_dir:
      - CT:   *_ventaneo*.nii   (ej: hepaticvessel_141_ventaneo_2.nii)
      - MASK: misma base sin '_ventaneo' (ej: hepaticvessel_141_2.nii)

    Devuelve:
      image: [1,Z,H,W]
      mask:
        - task="binary": {0,1} float (si return_binary_as_float=True) o long
        - task="multiclass": labels 0..8 long
      pid, patient_id

    IMPORTANTE:
      - self.samples GUARDA 4 CAMPOS:
        (ct_name, mask_name, pid, patient_id)
      - así el split por paciente se hace SIN cargar NIfTI.
    """

    def __init__(self, data_dir: str, task: str = "multiclass", return_binary_as_float: bool = True):
        self.data_dir = Path(data_dir)
        assert task in ("multiclass", "binary"), f"task inválido: {task}"
        self.task = task
        self.return_binary_as_float = return_binary_as_float

        ct_files = sorted([
            f for f in os.listdir(self.data_dir)
            if f.endswith(".nii") and ("ventaneo" in f.lower())
        ])

        self.samples = []
        for ct in ct_files:
            # base sin extensión y sin "_ventaneo"
            pid = _pid_from_ct(ct)
            mask = f"{pid}.nii"
            mask_path = self.data_dir / mask

            if mask_path.exists():
                patient_id = _patient_root(pid)
                self.samples.append((ct, mask, pid, patient_id))
            else:
                print(f"[WARN] máscara no encontrada para {ct} -> esperaba {mask}")

        print(f"[Couinaud3DDataset] task={self.task} | CT detectadas: {len(ct_files)} | pares: {len(self.samples)}")

        if len(self.samples) == 0 and len(ct_files) > 0:
            print("[ERROR] Se detectaron CT pero no matchearon máscaras. Ej CT:", ct_files[:5])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ct_name, ms_name, pid, patient_id = self.samples[idx]

        ct_path = self.data_dir / ct_name
        ms_path = self.data_dir / ms_name

        ct = nib.load(str(ct_path)).get_fdata().astype(np.float32)   # [H,W,Z]
        ms = nib.load(str(ms_path)).get_fdata().astype(np.int16)     # [H,W,Z]

        # [H,W,Z] -> [Z,H,W]
        ct = np.moveaxis(ct, -1, 0)
        ms = np.moveaxis(ms, -1, 0)

        # canal
        ct = ct[None, ...]  # [1,Z,H,W]

        if self.task == "binary":
            ms_bin = (ms > 0).astype(np.uint8)  # {0,1}
            if self.return_binary_as_float:
                mask_t = torch.from_numpy(ms_bin).float()
            else:
                mask_t = torch.from_numpy(ms_bin).long()
        else:
            mask_t = torch.from_numpy(ms).long()

        return {
            "image": torch.from_numpy(ct).float(),
            "mask": mask_t,
            "pid": pid,
            "patient_id": patient_id,
        }