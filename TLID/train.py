import argparse
import json
from pathlib import Path

import tensorflow as tf

from .data import load_dataset, split_dataset
from .models import BACKBONES, build_model, prepare_images


def main():
    parser = argparse.ArgumentParser(description='Treina classificadores de patches de tomate.')
    parser.add_argument('--data', required=True, help='Pasta com subpastas planta/classe/imagens.')
    parser.add_argument('--output', default='outputs')
    parser.add_argument('--dataset', choices=['tlid'], default='tlid')
    parser.add_argument('--models', nargs='+', choices=['cnn', 'vgg19', 'resnet50'], default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.image_size is None:
        args.image_size = 224 if args.dataset == 'tlid' else 32
    if args.models is None:
        args.models = ['cnn', 'vgg19', 'resnet50'] if args.dataset == 'tlid' else list(BACKBONES)
    if args.image_size < 32:
        parser.error('image-size precisa ser no mínimo 32.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    images, labels, classes = load_dataset(args.data, args.image_size)
    (x_train, y_train), (x_val, y_val), _ = split_dataset(images, labels, args.seed)
    metadata = {'dataset': args.dataset, 'classes': classes, 'models': args.models, 'seed': args.seed,
                'image_size': args.image_size, 'preprocessing': 'RGB: CNN /255, demais keras.applications',
                'split': '80/20 treino+val/teste; 80/20 treino/val estratificados'}
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    for name in args.models:
        print(f'\nTreinando {name} ({len(classes)} classes)')
        model = build_model(name, len(classes), args.image_size)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                       restore_best_weights=True)]
        if args.dataset == 'tlid':
            augmenter = tf.keras.preprocessing.image.ImageDataGenerator(
                width_shift_range=0.1, height_shift_range=0.1,
                shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                rotation_range=25, fill_mode='nearest')
            train_data = augmenter.flow(prepare_images(x_train, name), y_train,
                                        batch_size=args.batch_size, seed=args.seed)
            model.fit(train_data, validation_data=(prepare_images(x_val, name), y_val),
                      epochs=args.epochs, callbacks=callbacks)
        else:
            model.fit(prepare_images(x_train, name), y_train,
                      validation_data=(prepare_images(x_val, name), y_val),
                      epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)
        model.save(output / f'{name}.keras')


if __name__ == '__main__':
    main()
