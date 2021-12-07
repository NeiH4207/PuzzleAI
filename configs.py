
configs = {
    'preprocess': True,
    'test-size': 0.2,
    'validation-size': 0.1,
    'train-size': 0.7,
    'test-size': 0.2,
    'batch-size': 32,
    'epochs': 10,
    'learning-rate': 0.001,
    'momentum': 0.9,
    'decay': 0.0,
    'nesterov': True,
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy'],
    'optimizer': 'adam',
    'activation': 'relu',
    'dropout': 0.5,
    'regularizer': 'l2',
    'regularizer-rate': 0.01,
}

img_configs = {
    'image-size': (32, 32),
    'image-channels': 3,
    'block-size': (16, 16),
    'block-dim': (2, 2),
    'drop-rate': 0.5,
    'image-type': 'float32',
    'image-normalize': True,
    'image-mean': [0.485, 0.456, 0.406],
    'image-std': [0.229, 0.224, 0.225],
}