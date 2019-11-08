# Made by he.wang@uipath.com for UiPath process adaptation
#
import tensorflow as tf
import numpy as np
import os
import json
import argparse
from collections import Counter 

def load(model_path, labels_path=''):
    global loaded_model, imagenet_labels
    
    if not labels_path:
        labels_path = os.path.join(model_path, "label.txt")
    imagenet_labels = np.array(open(labels_path).read().splitlines())
    loaded_model = tf.saved_model.load(model_path)   

def inference (file):
    global loaded_model, imagenet_labels
    
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    """Imported signatures always return dictionaries."""
    infer = loaded_model.signatures[list(loaded_model.signatures.keys())[0]]
    #print(list(loaded.signatures.keys()))  # ["serving_default"] 
    
    none, img_height, img_width, num_of_classes = infer.inputs[0].shape 
    #print (img_height, img_width)
    
    img = tf.keras.preprocessing.image.load_img(file, target_size=[img_height, img_width])
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = tf.keras.applications.mobilenet.preprocess_input(x[tf.newaxis,...])

    """The 1st Key of structured output is the output layer's name"""
    result = infer(tf.constant(x))[list(infer.structured_outputs.keys())[0]]
    
    TopX = num_of_classes if num_of_classes<5 else 5
    dictOfRes = dict(zip(imagenet_labels, np.asarray(result[0])))
    TopXdictOfRes = Counter(dictOfRes).most_common(TopX) 
    #print (TopXdictOfRes)
    
    return json.dumps({
            k : str(round(v*100)) for k, v in TopXdictOfRes
            })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_help = True
    parser.add_argument("image", help="image to be processed")
    parser.add_argument("--model_path", help="graph/model to be executed")
    parser.add_argument("--labels_path", help="name of file containing labels, defualt would be the model_path/label.txt")
    args = parser.parse_args()

    assert args.image, "image is manditory"
        
    if args.model_path:
      model_path = args.model_path
    else:
      model_path = os.path.join(os.path.join(os.path.abspath(os.sep), 'tmp'), 'saved_models') 
    if args.labels_path:
        label_file = args.labels_path
    else:
        label_file = os.path.join(model_path, "label.txt")

    load ( model_path = model_path, labels_path = label_file )
    print( inference( file = args.image))
    
