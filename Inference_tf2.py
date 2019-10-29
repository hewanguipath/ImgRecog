# from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os

def Inference (file, model_path, labels_path=''):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])

    x = tf.keras.preprocessing.image.img_to_array(img)
    x = tf.keras.applications.mobilenet.preprocess_input(x[tf.newaxis,...])

    if not labels_path:
        labels_path = os.path.join(model_path, "label.txt")
        
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    loaded = tf.saved_model.load(model_path)
    #print(list(loaded.signatures.keys()))  # ["serving_default"]

    """Imported signatures always return dictionaries."""
    infer = loaded.signatures[list(loaded.signatures.keys())[0]]
    #print(infer.structured_outputs)

    """The 1st Key of structured output is the output layer's name"""
    result = infer(tf.constant(x))[list(infer.structured_outputs.keys())[0]]
    confidence = np.amax(result[0], axis=-1)
    print("labeling after saving and loading: ", np.asarray(result[0]))

    decoded = imagenet_labels[np.argmax(result[0], axis=-1)]
    #print("Result after saving and loading: ", decoded)
    
    return (decoded + ": " + str(confidence))


if __name__ == "__main__":
    print( Inference(
        file = "C:\\Users\\He.Wang\\Pictures\\DataSet\\Luggage_old\\BrokenLuggage\\download (13).jpg",
        model_path = "C:\\tmp\\saved_models"
        ))