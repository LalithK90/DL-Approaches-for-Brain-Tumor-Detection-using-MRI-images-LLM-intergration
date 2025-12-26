import tensorflow as tf

MODELS = {
    'vgg19_imbalanced': 'models/vgg19_imbalanced.h5',
    'vgg19_balance': 'models/vgg19_balance.h5',
    'vgg16_imbalanced': 'models/vgg16_imbalanced.h5',
    'vgg16_balance': 'models/vgg16_balance.h5',
    'propose_imbalanced': 'models/propose_imbalanced.h5',
    'propose_balance': 'models/propose_balance.h5',
    'ResNet50_imbalanced': 'models/ResNet50_imbalanced.h5',
    'ResNet50_balance': 'models/ResNet50_balance.h5',
    'MobileVNet_imbalanced': 'models/MobileVNet_imbalanced.h5',
    'MobileVNet_balance': 'models/MobileVNet_balance.h5',
    'GoogleLeNet_imbalanced': 'models/GoogleLeNet_imbalanced.h5',
    'GoogleLeNet_balance': 'models/GoogleLeNet_balance.h5',
}

LABELS = ['Glioma', 'Meningioma', 'Notumor', 'Pituitary']
IMAGE_SIZE = 224

def load_model(model_name):
    model_path = MODELS.get(model_name)
    if model_path:
        return tf.keras.models.load_model(model_path)
    return None