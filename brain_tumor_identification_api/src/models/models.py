import tensorflow as tf

MODELS = {
    'vgg19_imbalanced': 'models/vgg19_imbalanced.h5',
    'vgg19_balanced': 'models/vgg19_balanced.h5',
    'vgg16_imbalanced': 'models/vgg16_imbalanced.h5',
    'vgg16_balanced': 'models/vgg16_balanced.h5',
    'propose_imbalanced': 'models/propose_imbalanced.h5',
    'propose_balanced': 'models/propose_balanced.h5',
    'ResNet50_imbalanced': 'models/ResNet50_imbalanced.h5',
    'ResNet50_balanced': 'models/ResNet50_balanced.h5',
    'MobileVNet_imbalanced': 'models/MobileVNet_imbalanced.h5',
    'MobileVNet_balanced': 'models/MobileVNet_balanced.h5',
    'GoogleLeNet_imbalanced': 'models/GoogleLeNet_imbalanced.h5',
    'GoogleLeNet_balanced': 'models/GoogleLeNet_balanced.h5',
}

LABELS = ['Glioma', 'Meningioma', 'Notumor', 'Pituitary']
IMAGE_SIZE = 224

def load_model(model_name):
    model_path = MODELS.get(model_name)
    if model_path:
        return tf.keras.models.load_model(model_path)
    return None