from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_dataset(root: str, image_size: int = 32):
    """Carrega root/planta/classe/imagem, preservando a ordem das classes."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset não encontrado: {root_path}")
    class_dirs = sorted({p for p in root_path.glob('*/*') if p.is_dir()})
    classes = sorted({p.name for p in class_dirs})
    if len(classes) < 2:
        raise ValueError("São necessárias ao menos duas pastas de classes em dataset/planta/classe/.")
    images, labels = [], []
    for directory in class_dirs:
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            try:
                with Image.open(path) as image:
                    images.append(np.asarray(image.convert('RGB').resize((image_size, image_size)), dtype=np.uint8))
                labels.append(classes.index(directory.name))
            except OSError as exc:
                print(f"Imagem ignorada ({path}): {exc}")
    if not images:
        raise ValueError("Nenhuma imagem válida foi encontrada.")
    return np.stack(images), np.asarray(labels, dtype=np.int64), classes


def split_dataset(images, labels, seed=42):
    """Divide em treino (64%), validação (16%) e teste (20%)."""
    try:
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=seed, stratify=labels
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, test_size=0.2, random_state=seed,
            stratify=y_train_val
        )
    except ValueError as exc:
        raise ValueError("Cada classe precisa de imagens suficientes para as três divisões estratificadas.") from exc
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
