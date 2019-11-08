# Copyright 2019 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# most of the code get from tensorflow examples
# he.wang@uipath.com Modified for command line usage
# you are welcome to modify and add your functions

import time
import os
import datetime
from packaging import version
import argparse

import tensorflow as tf
import tensorflow_hub as hub

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This process requires TensorFlow 2.0 or above."

print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

def Retrain(data_dir, saved_model_path, epochs = 5, 
            do_data_augmentation = False, do_fine_tuning = False, 
            image_size=(224, 224), batch_size=32, saved_label_path='',
            module_handle = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"):
  
    start = time.time()
    print("Using {} with input size {}".format(module_handle, image_size))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    """
    Inputs are suitably resized for the selected module. Dataset augmentation (i.e., random distortions of an image each time it is read) improves training, esp. when fine-tuning.
    """
    datagen_kwargs = dict(rescale=1./255, validation_split=.20)
    dataflow_kwargs = dict(target_size=image_size, batch_size=batch_size, interpolation="bilinear")

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

    if do_data_augmentation:
      train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rotation_range=40,
          horizontal_flip=True,
          width_shift_range=0.2, height_shift_range=0.2,
          shear_range=0.2, zoom_range=0.2,
          **datagen_kwargs)
    else:
      train_datagen = valid_datagen
      
    train_generator = train_datagen.flow_from_directory(data_dir, subset="training", shuffle=True, **dataflow_kwargs)

    class_names = sorted(train_generator.class_indices.items(), key=lambda pair:pair[1])
    class_names = [key.title() for key, value in class_names]
    print ("Labels: ", class_names)
    
    """## Defining the model
    All it takes is to put a linear classifier on top of the `feature_extractor_layer` with the Hub module.
    For speed, we start out with a non-trainable `feature_extractor_layer`, but you can also enable fine-tuning for greater accuracy.
    """
    model = tf.keras.Sequential([
        hub.KerasLayer(module_handle, trainable=do_fine_tuning),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(train_generator.num_classes, activation='softmax',
                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,)+image_size+(3,))
    print(model.summary())

    """## compile the model"""
    model.compile(
      optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), 
      loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
      metrics=['accuracy'])

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size
    
    """ Define the Keras TensorBoard callback. """
    logdir = os.path.join(saved_model_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    """## Training the model"""
    hist = model.fit_generator(
        train_generator,
        epochs=epochs, steps_per_epoch=steps_per_epoch,
        callbacks = [tensorboard_callback],
        validation_data=valid_generator,
        validation_steps=validation_steps).history
    print(hist)

    """Finally, the trained model can be saved for deployment to TF Serving or TF Lite (on mobile) as follows."""
    tf.saved_model.save(model, saved_model_path)
    
    if not saved_label_path:
      saved_label_path = os.path.join(saved_model_path, 'label.txt')
    with open(saved_label_path, 'w') as f:
      for item in class_names:
          f.write("%s\n" % item)
          
    end = time.time()
    duration = end-start
    print ("time consumed: " + str(datetime.timedelta(seconds=int(duration))) + "s")
    return str(hist['val_accuracy'][-1])
    
""" This Wrapper function is only for old models, eg. efficent Net """
class Wrapper(tf.train.Checkpoint):
  def __init__(self, spec):
    super(Wrapper, self).__init__()
    self.module = hub.load(spec, tags=[])
    self.variables = self.module.variables
    self.trainable_variables = []
  def __call__(self, x):
    return self.module.signatures["default"](x)["default"]
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_help = True
    parser.add_argument("image_dir", help="folder of training images")
    parser.add_argument("--saved_model_dir", help="graph/model to be output")
    parser.add_argument("--output_labels", help="Label file to be output")
    parser.add_argument("--tfhub_module", default="https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
                        help="Which TensorFlow Hub module to use. For more options, search https://tfhub.dev/s?module-type=image-classification&q=tf2 for image feature vector modules.")
    parser.add_argument("--epochs", default=5, type=int, help="how many epochs to run")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--image_size", type=int, help="the input image size of the module")
    parser.add_argument("--do_data_augmentation", type=bool, default=False, help="whether or not do data augmentation")
    parser.add_argument("--do_fine_tuning", type=bool, default=False, help="whether or not do fine tuning")
    args = parser.parse_args()

    assert args.image_dir, "Please give the training images by --image_dir"

    if args.saved_model_dir:
      saved_model_dir = args.saved_model_dir
    else:
      saved_model_dir = os.path.join(os.path.join(os.path.abspath(os.sep), 'tmp'), 'saved_model')
    if args.output_labels:
      output_labels = args.output_labels
    else:
      output_labels = os.path.join(saved_model_dir, 'label.txt')
    
    image_size = None
    if args.image_size:
      image_size = (args.image_size, args.image_size)
    else:
      try:
        module_spec = hub.load_module_spec(args.tfhub_module)
        image_size = tuple(hub.get_expected_image_size(module_spec))
        print ("get model spec", image_size)
      except:
        if "mobilenet" in args.tfhub_module:
          print ("get model spec failed, use default spec as 224 x 224,")
          image_size = (224, 224)

    if image_size: 
      Retrain(
      data_dir = args.image_dir,
      saved_model_path = saved_model_dir,
      saved_label_path = output_labels,
      epochs = args.epochs,
      batch_size = args.batch_size,
      module_handle = args.tfhub_module,
      image_size = image_size,
      do_data_augmentation = args.do_data_augmentation,
      do_fine_tuning = args.do_fine_tuning
      )
    else:
      print("Due to fail to get spec from TF2 modules, you have to setup the image size with --image_size, eg. \"--image_size 299\" for inception_v3")