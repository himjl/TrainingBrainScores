import os
import tensorflow as tf

def download_weights(weights_name, epoch=None):

    if epoch is None:
        epoch = 90

    if epoch > 90 or epoch < 1:
        raise ValueError('Epoch must be between 1 and 90 (inclusive)')

    epoch_string = str(epoch).zfill(4)

    if weights_name == 'MutatorB1000':
        # Synthetic training with no image augmentation
        weight_urls = [
            ' https://milresources.s3.amazonaws.com/pretrained_weights/resnet_v1_50/MutatorB1000/model.ckpt-%s.data-00000-of-00002'%epoch_string,
            'https://milresources.s3.amazonaws.com/pretrained_weights/resnet_v1_50/MutatorB1000/model.ckpt-%s.data-00001-of-00002'%epoch_string,
            ' https://milresources.s3.amazonaws.com/pretrained_weights/resnet_v1_50/MutatorB1000/model.ckpt-%s.index'%epoch_string,
            ]

        weight_names = [s.split('MutatorB1000/')[-1] for s in weight_urls]
        cache_subdir = 'ResNet_v1_50_MutatorB1000'
        fpath = os.path.join(os.path.expanduser('~'), '.keras', cache_subdir, 'model.ckpt-%s'%epoch_string)

    elif weights_name == 'MutatorB1000_aug':
        # Synthetic training with heavy image augmentation
        weight_urls = [
            ' https://milresources.s3.amazonaws.com/pretrained_weights/resnet_v1_50/MutatorB1000_aug/model.ckpt-%s.data-00000-of-00002' % epoch_string,
            'https://milresources.s3.amazonaws.com/pretrained_weights/resnet_v1_50/MutatorB1000_aug/model.ckpt-%s.data-00001-of-00002' % epoch_string,
            ' https://milresources.s3.amazonaws.com/pretrained_weights/resnet_v1_50/MutatorB1000_aug/model.ckpt-%s.index' % epoch_string,
        ]

        weight_names = [s.split('MutatorB1000_aug/')[-1] for s in weight_urls]
        cache_subdir = 'ResNet_v1_50_MutatorB1000_aug'
        fpath = os.path.join(os.path.expanduser('~'), '.keras', cache_subdir, 'model.ckpt-%s' % epoch_string)

    elif weights_name == 'imagenet':
        # Standard ResNet imagenet training (some cropping-based augmentation)
        weight_urls = [' https://milresources.s3.amazonaws.com/pretrained_weights/resnet_v1_50/imagenet/model.ckpt-%s.data-00000-of-00002'%epoch_string,
                       'https://milresources.s3.amazonaws.com/pretrained_weights/resnet_v1_50/imagenet/model.ckpt-%s.data-00001-of-00002'%epoch_string,
                       ' https://milresources.s3.amazonaws.com/pretrained_weights/resnet_v1_50/imagenet/model.ckpt-%s.index'%epoch_string,
                       ]

        weight_names = [s.split('imagenet/')[-1] for s in weight_urls]
        cache_subdir = 'ResNet_v1_50_imagenet'

        fpath = os.path.join(os.path.expanduser('~'), '.keras', cache_subdir, 'model.ckpt-%s'%epoch_string)
    else:
        raise ValueError('Could not identify requested weight set %s' % weights_name)

    for weights_name, weights_location in zip(weight_names, weight_urls):
        tf.keras.utils.get_file(
            weights_name,
            weights_location,
            md5_hash=None,
            file_hash=None,
            cache_subdir=cache_subdir,
            hash_algorithm='auto',
            extract=True,
            archive_format='auto',
            cache_dir=None)

    return fpath
