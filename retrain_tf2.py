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

import time
import os
import datetime
from packaging import version

import tensorflow as tf
import tensorflow_hub as hub

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This process requires TensorFlow 2.0 or above."

print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

def Retrain(data_dir, saved_model_path, epochs = 5, do_data_augmentation = False, do_fine_tuning = False):
    start = time.time()
    #MODULE_HANDLE = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
    MODULE_HANDLE = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"
    IMAGE_SIZE = (224, 224)
    print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

    BATCH_SIZE = 32 #@param {type:"integer"}

    """
    Inputs are suitably resized for the selected module. Dataset augmentation (i.e., random distortions of an image each time it is read) improves training, esp. when fine-tuning.
    """

    datagen_kwargs = dict(rescale=1./255, validation_split=.20)
    dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, interpolation="bilinear")

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
    print ("Classes: ", class_names)

    """## Defining the model
    All it takes is to put a linear classifier on top of the `feature_extractor_layer` with the Hub module.
    For speed, we start out with a non-trainable `feature_extractor_layer`, but you can also enable fine-tuning for greater accuracy.
    """
    print("Building model with", MODULE_HANDLE)

    """ This Wrapper function is only for old models, eg. efficent Net """
    class Wrapper(tf.train.Checkpoint):
      def __init__(self, spec):
        super(Wrapper, self).__init__()
        self.module = hub.load(spec, tags=[])
        self.variables = self.module.variables
        self.trainable_variables = []
      def __call__(self, x):
        return self.module.signatures["default"](x)["default"]

    if 'efficientnet' in MODULE_HANDLE:
      MODULE_HANDLE = Wrapper(MODULE_HANDLE)
      
    model = tf.keras.Sequential([
        hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(train_generator.num_classes, activation='softmax',
                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,)+IMAGE_SIZE+(3,))
    model.summary()

    """## Training the model"""

    model.compile(
      optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), 
      loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
      metrics=['accuracy'])

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size
    
    """ Define the Keras TensorBoard callback. """
    logdir = os.path.join(saved_model_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    hist = model.fit_generator(
        train_generator,
        epochs=epochs, steps_per_epoch=steps_per_epoch,
        callbacks = [tensorboard_callback],
        validation_data=valid_generator,
        validation_steps=validation_steps).history
    print(hist)

    """Finally, the trained model can be saved for deployment to TF Serving or TF Lite (on mobile) as follows."""
    tf.saved_model.save(model, saved_model_path)
    
    with open(os.path.join(saved_model_path, 'label.txt'), 'w') as f:
      for item in class_names:
          f.write("%s\n" % item)
          
    end = time.time()
    duration = end-start
    print ("time consumed: " + str(datetime.timedelta(seconds=int(duration))) + "s")
    return str(hist['val_accuracy'][-1])
    

if __name__ == "__main__":
    print( Retrain(
    data_dir = 'C:\\Users\\He.Wang\\Pictures\\DataSet\\Luggage_old',
    saved_model_path = "C:\\tmp\\saved_models",
    epochs = 2,
    do_data_augmentation = False
    ))