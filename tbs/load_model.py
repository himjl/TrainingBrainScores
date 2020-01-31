import numpy as np
import tbs.resnet_model as resnet_model
import tbs.download_weights as download_weights

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]


def imagenet_preprocessing(img:np.ndarray):
    # img: np.ndarray [batch, h, w, 3]
    img = img - np.array(CHANNEL_MEANS)[None, ...]
    return img


def ResNet50(
        weights='MutatorB1000',
        epoch=None,
        batch_size=None,
        trainable = False):

    model = resnet_model.resnet50(num_classes=1000,
                                  batch_size=batch_size,
                                  )
    model.trainable = trainable

    if weights == 'random_weights':
        return model

    weights_path = download_weights.download_weights(weights, epoch=epoch)
    model.load_weights(weights_path)
    return model
