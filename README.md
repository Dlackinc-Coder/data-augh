# mioclinic-aug

Aumento de dataset y segmentación para MioClinic: corte cardíaco teñido con TTC + Evans Blue.

Base: **Mouse Cardiac Infarction Image Dataset** (243 pares imagen/máscara, 6000×4000,
4 clases 0–3, Apache-2.0). Fuente: https://github.com/wwz4416/mouse-cardiac-dataset

## Estructura

```
mioclinic-aug/
├── data/mouse_cardiac/     # NO va al git (ver "Dataset")
│   ├── train/{img,mask}/   # 171 pares
│   ├── val/{img,mask}/     # 35 pares
│   └── test/{img,mask}/    # 37 pares
├── src/
│   ├── dataset.py          # tiling 512², augmentación, DataLoader
│   ├── seg_model.py        # U-Net efficientnet-b0 (segmentation-models-pytorch)
│   ├── train_seg.py        # fine-tuning con AMP + grad accumulation + TensorBoard
│   ├── eval.py             # Dice / IoU multiclase
│   └── pix2pix/
│       ├── networks.py     # UnetGenerator (n_down=7) + PatchGAN (n_layers=3)
│       ├── data.py         # máscara one-hot (4ch) → imagen real en [-1,1]
│       ├── model.py        # pix2pix (LSGAN + L1×100)
│       └── train_pix2pix.py
├── configs/
│   ├── seg.yaml
│   └── pix2pix.yaml
├── runs/                   # tensorboard + muestras pix2pix
└── checkpoints/            # best.pt / last.pt
```

## Setup (PC con RTX 4060, Ubuntu + Tailscale)

```bash
git clone https://github.com/TU_USUARIO/mioclinic-aug.git
cd mioclinic-aug

# dataset (si data/mouse_cardiac no existe)
git clone https://github.com/wwz4416/mouse-cardiac-dataset.git /tmp/mcd
mkdir -p data/mouse_cardiac
unzip /tmp/mcd/dataset.zip -d data/mouse_cardiac   # requiere git-lfs instalado
cp /tmp/mcd/LICENSE data/mouse_cardiac/

# entorno
conda create -n mioaug python=3.11 -y && conda activate mioaug
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

## Entrenamiento

### 1. Segmentador (U-Net efficientnet-b0)

```bash
cd src
python train_seg.py ../configs/seg.yaml
tensorboard --logdir ../runs
python eval.py ../configs/seg.yaml        # Dice/IoU en test
```

Batch 2 a 512² con AMP + grad_accum 4 (batch efectivo 8) ≈ 4–5 GB VRAM.
~10–15 min/epoch en RTX 4060.

### 2. Generador pix2pix (máscara → imagen)

```bash
cd src/pix2pix
python train_pix2pix.py ../../configs/pix2pix.yaml
```

Batch 1 a 512² (G y D por separado) ≈ 5–7 GB VRAM. Muestras cada 200 steps en
`runs/pix2pix_samples/`. Checkpoints en `checkpoints/pix2pix/`.

## Mask de clases

Máscaras en gris, valores 0–3 (recorte/limpieza de ruido JPEG aplicado en código):

| valor | probable significado |
|-------|----------------------|
| 0 | background |
| 1–3   | LV / AAR / IS (no verificado) |

## Verificación

Smoke tests pasados (CPU, 08-oct-2026):

- `TileDataset`/`make_loaders`: tiles 512² con clases {0..3} y ruido JPEG limpio
- `train_seg.py`: 1 época mini → checkpoint `best.pt`/`last.pt` + logs TensorBoard
- `eval.py`: carga checkpoint y calcula Dice/IoU
- pix2pix: forward 512² (G 41.8M + D 2.8M params), `train_step` (LSGAN+L1),
  loop 1 época → `pix2pix_last.pt` + muestras PNG en `runs/pix2pix_samples/`

Pendiente de probar en GPU (RTX 4060): VRAM batch 2 + AMP (seg) y batch 1 (pix2pix).

## Licencia de código

Sobre Apache-2.0 (dataset original también Apache-2.0).
