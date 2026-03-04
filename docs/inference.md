VERSION 3
python3 concatenado_v3.py \
  --in_path "/ruta/a/las/imagenes_del_paciente" \
  --out_dir "/ruta/donde/guardar/las/predicciones" \
  --liver_ckpt "/ruta/al/best_model_liver.pth" \
  --couinaud_ckpt "/ruta/al/best_model_couinaud.pth" \
  --window_mode auto --wl -200 --wh 250 --norm 01 \
  --win 32 --stride 8 \
  --mask_couinaud_with_liver \
  --prefer_oriented \
  --amp

