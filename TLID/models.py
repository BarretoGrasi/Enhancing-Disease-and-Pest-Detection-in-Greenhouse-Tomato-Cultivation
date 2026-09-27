import tensorflow as tf


BACKBONES = {
    'mobilenet': (tf.keras.applications.MobileNet, tf.keras.applications.mobilenet.preprocess_input),
    'vgg19': (tf.keras.applications.VGG19, tf.keras.applications.vgg19.preprocess_input),
    'resnet50': (tf.keras.applications.ResNet50, tf.keras.applications.resnet50.preprocess_input),
}


def prepare_images(images, name):
    """Converte RGB 0..255 para a convenção ImageNet da arquitetura."""
    if name == 'cnn':
        return images.astype('float32') / 255.0
    return BACKBONES[name][1](images.astype('float32').copy())


def build_model(name: str, num_classes: int, image_size: int = 32):
    if name == 'cnn':
        regularizer = tf.keras.regularizers.l2(0.001)
        model = tf.keras.Sequential(name='cnn_patches_or_leaves')
        model.add(tf.keras.Input(shape=(image_size, image_size, 3)))
        for filters, kernel in [(32, 5), (64, 3), (128, 3)]:
            model.add(tf.keras.layers.Conv2D(filters, kernel, activation='relu',
                                              padding='same', kernel_regularizer=regularizer))
            model.add(tf.keras.layers.Conv2D(filters, kernel, activation='relu',
                                              kernel_regularizer=regularizer))
            model.add(tf.keras.layers.MaxPooling2D(2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizer))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    if name not in BACKBONES:
        raise ValueError(f"Modelo desconhecido: {name}")
    architecture, _ = BACKBONES[name]
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name='rgb_image')
    backbone = architecture(weights='imagenet', include_top=False,
                            input_shape=(image_size, image_size, 3))
    features = backbone(inputs)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dense(512, activation='relu')(features)
    features = tf.keras.layers.Dropout(0.5)(features)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(features)
    model = tf.keras.Model(inputs, outputs, name=f'{name}_patches')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
