# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# Wang He modified for uipath python invoke activity
# optimized for better performance, load model only once

import numpy as np
import tensorflow as tf
# Just disables the warning, doesn't enable AVX/FMA
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

#Load model and Lable
def load():
  global graph, input_operation, output_operation, labels, ts
  ts = time.time()
  print ("Start: ", time.time()-ts)

  model_file = "/tmp/output_graph.pb"
  label_file = "/tmp/output_labels.txt"

  input_layer = "Placeholder"
  output_layer = "final_result"

  graph = load_graph(model_file)
  labels = load_labels(label_file)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)
  print ("load model: ", time.time()-ts)

#Process image recog
def imageProcess(file_name):
  global graph, input_operation, output_operation, labels,ts
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255

  t = read_tensor_from_image_file(
      file_name,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)
  print ("load img: ", time.time()-ts)

  with tf.Session(graph=graph) as sess:
    results = sess.run(
      output_operation.outputs[0], {
      input_operation.outputs[0]: t
    })
  print ("Classify: ", time.time()-ts)
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  if results[top_k[0]] > 0.8:  #the confidence threshold to be considered as valid prediction
    return (labels[top_k[0]]+": "+str(round(float(results[top_k[0]])*100,2)) + "% \n" + labels[top_k[1]]+": "+str(round(float(results[top_k[1]])*100,2)) + "% \n" + labels[top_k[2]]+": "+str(round(float(results[top_k[2]])*100,2)) + "%")
  else:
    return ("To be verify: "+str(round(float(results[top_k[0]])*100,2)) + "%")


load()
print(imageProcess("c:\\tmp\\me.jpg"))