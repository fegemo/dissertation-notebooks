import tensorflow as tf

# executed in tensorflow 2.10.1 (keras 2)
def save_model_in_keras2():
    input_layer_0 = tf.keras.layers.Input(shape=(10,), name="image_input")
    hidden_layer_1 = tf.keras.layers.Dense(10, activation='relu')(input_layer_0)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer_1)
    single_input_model = tf.keras.models.Model(inputs=input_layer_0, outputs=output_layer)
    single_input_model.save('temp/single_input_keras2')

    input_layer_1 = tf.keras.layers.Input(shape=(2,), name="label_input")
    hidden_layer_1 = tf.keras.layers.Concatenate()([input_layer_0, input_layer_1])
    hidden_layer_2 = tf.keras.layers.Dense(2, activation='relu')(hidden_layer_1)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer_2)
    multi_input_model = tf.keras.models.Model(inputs=(input_layer_0, input_layer_1), outputs=output_layer)
    multi_input_model.save('temp/multi_input_keras2')

# executed in tensorflow 2.18.0 (keras 3)
def save_model_in_keras3():
    input_layer_0 = tf.keras.layers.Input(shape=(10,), name="image_input")
    hidden_layer_1 = tf.keras.layers.Dense(10, activation='relu')(input_layer_0)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer_1)
    single_input_model = tf.keras.models.Model(inputs=input_layer_0, outputs=output_layer)
    single_input_model.export('temp/single_input_keras3')

    input_layer_1 = tf.keras.layers.Input(shape=(2,), name="label_input")
    hidden_layer_1 = tf.keras.layers.Concatenate()([input_layer_0, input_layer_1])
    hidden_layer_2 = tf.keras.layers.Dense(2, activation='relu')(hidden_layer_1)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer_2)
    multi_input_model = tf.keras.models.Model(inputs=(input_layer_0, input_layer_1), outputs=output_layer)
    multi_input_model.export('temp/multi_input_keras3')


# execute in tensorflow 2.18.0 (keras 3)
image_input = tf.random.normal((1, 10))
label_input = tf.random.normal((1, 2))

# load keras3 model in keras3 (works)
model = tf.keras.layers.TFSMLayer('temp/multi_input_keras3')
result = model([image_input, label_input])
print("result:", result)

# load legacy model in keras 3 (calling it fails)
legacy_model = tf.keras.layers.TFSMLayer('temp/multi_input_keras2', call_endpoint='serving_default')
result = legacy_model([image_input, label_input])
print("result:", result)
