import tbs.load_model as load_model
import tensorflow as tf

model = load_model.ResNet50(weights = 'imagenet', epoch=10, batch_size=None, trainable=False)

# To get intermediate layers:
layer_name = 'activation_48'

inputs = model.inputs
ambient_outputs = model.get_layer(name=layer_name).output
model = tf.keras.Model(inputs = inputs, outputs = ambient_outputs)

"""
img_disk = np.array([PIL.Image.open(path) for path in fpaths] )
img_input = load_model.imagenet_preprocessing(img = img_disk)
logits = model(img_input)
"""
