from tensorflow import keras
import tensorflow_hub as hub

num_classes = 1000

input_layer = keras.Input(shape=(224, 224, 3))
hub_layer = hub.KerasLayer("https://tfhub.dev/sayakpaul/mixer_b16_i21k_fe/1",  trainable=True)
x = hub_layer(input_layer)
output_layer = keras.layers.Dense(num_classes, activation='softmax')
model = keras.Model(inputs=input_layer, outputs=output_layer)