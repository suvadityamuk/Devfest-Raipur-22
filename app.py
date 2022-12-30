import tensorflow as tf
from tensorflow import keras
import os
tf.config.run_functions_eagerly(True)

# To force inference using CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Model definition
image_input = keras.Input(shape=(None, None, 3))

x = keras.layers.Resizing(
    height=224, width=224, interpolation="lanczos3", crop_to_aspect_ratio=False
)(image_input)

x = keras.layers.Rescaling(scale=1.0 / 255, offset=0.0)(x)

mobilenet = keras.applications.MobileNetV2(
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    classes=1000,
    classifier_activation="softmax",
)

model_output = mobilenet(x)

model = keras.Model(inputs=image_input, outputs=model_output)

version_number = 1

# Internal inference function
def inference(image: tf.Tensor):
    y = model(image)
    preds = keras.applications.imagenet_utils.decode_predictions(y, top=5)
    result = {i[1]: str(i[2]) for i in preds[0]}
    sorted_result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1])}
    return sorted_result

# Output Signature function for TensorFlow Serving
@tf.function(input_signature=[tf.TensorSpec(name="image_bytes_string", shape=None, dtype=tf.string)])
def predict_b64_string(b64str_tensor):
    img = tf.reshape(b64str_tensor, [])
    img = tf.io.decode_image(img, channels=3, dtype=tf.float16, expand_animations=False)
    tensor = tf.expand_dims(img, axis=0)
    res = inference(tensor)
    return res

model.save(
    './mobilenetv2-imagenet-devfest/{version_number}',
    save_format='tf',
    include_optimizer=True,
    overwrite=True,
    signatures={
        "image_b64string_signature":predict_b64_string
    }
)