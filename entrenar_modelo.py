"""Entrena un modelo de segmentación de partes de insectos con el conjunto que exporta Entometría.

Cómo usarlo en Google Colab (gratis, con GPU):
  1. Entorno de ejecución → Cambiar tipo de entorno → GPU.
  2. Suba el zip que descargó en Entometría (pestaña Etiquetar → «Descargar conjunto para entrenar»).
  3. En una celda:   !pip install -q segmentation-models-pytorch onnx
                     !python entrenar_modelo.py entometria_entrenamiento_AAAA-MM-DD.zip
  4. Descargue modelo.onnx y cárguelo en Entometría: Configuración del insecto → Modelo propio → Cargar .onnx,
     con las clases de clases.txt separadas por coma y el lado indicado al final (512 de fábrica).

También corre en una computadora con Python 3.10+ y PyTorch (más lento sin GPU).

El modelo recibe la foto entera llevada a LADO × LADO (sin conservar la proporción, igual que la app), normalizada
como ImageNet, y entrega 1 × clases × LADO × LADO: es lo que espera Entometría.
"""
import argparse
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

try:
    import segmentation_models_pytorch as smp
except ImportError as e:  # pragma: no cover
    raise SystemExit("Falta segmentation-models-pytorch:  pip install segmentation-models-pytorch onnx") from e

MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESV = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def leer_conjunto(origen: Path):
    """Abre el zip (o una carpeta ya descomprimida) y devuelve las parejas foto-máscara y las clases."""
    carpeta = origen
    if origen.suffix.lower() == ".zip":
        carpeta = origen.with_suffix("")
        with zipfile.ZipFile(origen) as z:
            z.extractall(carpeta)
    clases = [c.strip() for c in (carpeta / "clases.txt").read_text(encoding="utf-8").splitlines() if c.strip()]
    parejas = []
    for img in sorted((carpeta / "imagenes").glob("*.jpg")):
        mascara = carpeta / "mascaras" / (img.stem + ".png")
        if mascara.exists():
            parejas.append((img, mascara))
    if not parejas:
        raise SystemExit("El conjunto no trae imágenes con máscara.")
    return parejas, clases


class Conjunto(torch.utils.data.Dataset):
    def __init__(self, parejas, lado: int, aumentar: bool):
        self.parejas = parejas
        self.lado = lado
        self.aumentar = aumentar

    def __len__(self):
        return len(self.parejas)

    def __getitem__(self, i):
        img = Image.open(self.parejas[i][0]).convert("RGB")
        mas = Image.open(self.parejas[i][1]).convert("L")
        if self.aumentar:
            # Giros, espejos, un recorte y cambios de luz: las fotos reales varían en todo eso.
            if random.random() < 0.5:
                img, mas = img.transpose(Image.FLIP_LEFT_RIGHT), mas.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img, mas = img.transpose(Image.FLIP_TOP_BOTTOM), mas.transpose(Image.FLIP_TOP_BOTTOM)
            angulo = random.uniform(-25, 25)
            img = img.rotate(angulo, resample=Image.BILINEAR, fillcolor=(235, 235, 230))
            mas = mas.rotate(angulo, resample=Image.NEAREST, fillcolor=0)
            w, h = img.size
            k = random.uniform(0.75, 1.0)
            cw, ch = int(w * k), int(h * k)
            x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
            img, mas = img.crop((x0, y0, x0 + cw, y0 + ch)), mas.crop((x0, y0, x0 + cw, y0 + ch))
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
            img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
        img = img.resize((self.lado, self.lado), Image.BILINEAR)
        mas = mas.resize((self.lado, self.lado), Image.NEAREST)
        x = (np.asarray(img, dtype=np.float32) / 255 - MEDIA) / DESV
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(np.asarray(mas, dtype=np.int64))


def dice(logits, objetivo, clases):
    """Pérdida Dice promedio de las clases (sin el fondo): ayuda con las partes chicas, como patas y antenas."""
    p = logits.softmax(1)
    uno = F.one_hot(objetivo, clases).permute(0, 3, 1, 2).float()
    inter = (p * uno).sum((0, 2, 3))
    union = p.sum((0, 2, 3)) + uno.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (union + 1))[1:].mean()


def iou_por_clase(modelo, cargador, clases, dispositivo):
    inter = np.zeros(clases)
    union = np.zeros(clases)
    modelo.eval()
    with torch.no_grad():
        for x, y in cargador:
            pred = modelo(x.to(dispositivo)).argmax(1).cpu()
            for c in range(clases):
                inter[c] += ((pred == c) & (y == c)).sum().item()
                union[c] += ((pred == c) | (y == c)).sum().item()
    return inter / np.maximum(union, 1)


def main():
    # En la consola de Windows los acentos fallan si la salida no es UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser(description="Entrena un modelo de partes de insectos para Entometría")
    a.add_argument("conjunto", type=Path, help="zip exportado por Entometría (o su carpeta)")
    a.add_argument("--lado", type=int, default=512, help="lado de entrada del modelo (múltiplo de 32)")
    a.add_argument("--epocas", type=int, default=80)
    a.add_argument("--lote", type=int, default=4)
    a.add_argument("--codificador", default="mobilenet_v2", help="mobilenet_v2 (liviano, para el teléfono) o resnet34")
    a.add_argument("--salida", type=Path, default=Path("modelo.onnx"))
    args = a.parse_args()

    parejas, clases = leer_conjunto(args.conjunto)
    random.seed(1)
    random.shuffle(parejas)
    n_val = max(1, len(parejas) // 6)
    validacion, entrenamiento = parejas[:n_val], parejas[n_val:] or parejas
    print(f"{len(entrenamiento)} imágenes para entrenar, {len(validacion)} para validar; clases: {', '.join(clases)}")

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = smp.Unet(args.codificador, encoder_weights="imagenet", classes=len(clases)).to(dispositivo)
    cargar = lambda p, aum: torch.utils.data.DataLoader(Conjunto(p, args.lado, aum), batch_size=args.lote, shuffle=aum, num_workers=2)
    entrenar, validar = cargar(entrenamiento, True), cargar(validacion, False)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=3e-4, weight_decay=1e-4)
    calendario = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, args.epocas)

    mejor, estado = -1.0, None
    for epoca in range(1, args.epocas + 1):
        modelo.train()
        total = 0.0
        for x, y in entrenar:
            x, y = x.to(dispositivo), y.to(dispositivo)
            logits = modelo(x)
            perdida = F.cross_entropy(logits, y) + dice(logits, y, len(clases))
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()
            total += perdida.item()
        calendario.step()
        iou = iou_por_clase(modelo, validar, len(clases), dispositivo)
        media = iou[1:].mean() if len(clases) > 1 else iou.mean()
        if media > mejor:
            mejor, estado = media, {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        detalle = " ".join(f"{c}={v:.2f}" for c, v in zip(clases[1:], iou[1:]))
        print(f"época {epoca:3d}  pérdida {total / len(entrenar):.3f}  IoU {media:.3f}  ({detalle})")

    modelo.load_state_dict(estado)
    modelo.eval().cpu()
    entrada = torch.zeros(1, 3, args.lado, args.lado)
    torch.onnx.export(modelo, entrada, str(args.salida), input_names=["entrada"], output_names=["salida"], opset_version=17)
    print()
    print(f"Listo: {args.salida} (mejor IoU de validación {mejor:.3f}).")
    print("En Entometría: Configuración del insecto > Modelo propio > Cargar .onnx, con:")
    print(f"  Clases: {', '.join(clases)}")
    print(f"  Lado:   {args.lado}")


if __name__ == "__main__":
    main()
