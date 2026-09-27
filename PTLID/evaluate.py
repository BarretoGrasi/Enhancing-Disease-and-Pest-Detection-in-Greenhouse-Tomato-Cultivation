import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .data import load_dataset, split_dataset
from .ensemble import predict_ensemble
from .models import prepare_images


def report(y_true, y_pred, classes):
    indices = list(range(len(classes)))
    print(f'Acurácia: {accuracy_score(y_true, y_pred):.4f}')
    print('Matriz de confusão:\n', confusion_matrix(y_true, y_pred, labels=indices))
    print(classification_report(y_true, y_pred, labels=indices, target_names=classes,
                                digits=4, zero_division=0))


def main():
    parser = argparse.ArgumentParser(description='Avalia modelos ou um ensemble no conjunto de teste.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='outputs')
    parser.add_argument('--mode', choices=['hard', 'soft'], default='hard')
    parser.add_argument('--weights', type=float, nargs='+', help='Pesos fixos para votação soft.')
    args = parser.parse_args()
    output = Path(args.output)
    metadata = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
    images, labels, classes = load_dataset(args.data, metadata['image_size'])
    if classes != metadata['classes']:
        raise ValueError('As classes atuais diferem das classes registradas durante o treino.')
    _, _, (x_test, y_test) = split_dataset(images, labels, metadata['seed'])
    names = metadata['models']
    models = [tf.keras.models.load_model(output / f'{name}.keras', compile=False) for name in names]
    class PreparedModel:
        def __init__(self, model, name):
            self.model, self.name = model, name

        def predict(self, images, **kwargs):
            return self.model.predict(prepare_images(images, self.name), **kwargs)

    models = [PreparedModel(model, name) for model, name in zip(models, names)]
    for name, model in zip(names, models):
        print(f'\n{name}')
        report(y_test, model.predict(x_test, verbose=0).argmax(axis=1), classes)
    if len(models) > 1:
        print(f'\nEnsemble {args.mode}')
        y_pred, _ = predict_ensemble(models, x_test, args.mode, args.weights)
        report(y_test, y_pred, classes)


if __name__ == '__main__':
    main()
