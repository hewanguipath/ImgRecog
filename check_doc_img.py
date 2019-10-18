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
# Optimized for better performance, load model only once
# Add function for going through folders

import numpy as np
import tensorflow as tf
# Just disables the warning, doesn't enable AVX/FMA
import os
import sys
import time
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
fileCount = 0

def variance_of_laplacian(imagePath):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
  image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
  return cv2.Laplacian(image, cv2.CV_64F).var()

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.compat.v1.GraphDef()

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
  file_reader = tf.io.read_file(file_name, input_name)
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
  resized = tf.compat.v1.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.compat.v1.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

#Load model and Lable
def load():
  global graph, input_operation, output_operation, labels, ts
  ts = time.time()
  #print ("Start: ", time.time()-ts)

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
  print ("load model takes: ", str(round(time.time()-ts,3)) + "s")
  ts = time.time()

#Process image recog
def imageProcess(file_name):
  global graph, input_operation, output_operation, labels,ts
  PERFORMANCE_TESTING = 0

  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  print(os.path.basename(file_name), end="\t")

  t = read_tensor_from_image_file(
      file_name,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)
  if PERFORMANCE_TESTING:
    print ("\t"+"load img: ", round(time.time()-ts, 3))
    ts = time.time()

  with tf.compat.v1.Session(graph=graph) as sess:
    results = sess.run(
      output_operation.outputs[0], {
      input_operation.outputs[0]: t
    })
  if PERFORMANCE_TESTING:
    print ("\t"+"Classify: ", round(time.time()-ts, 3))
    ts = time.time()
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  if results[top_k[0]] > 0.8:  #the confidence threshold to be considered as valid prediction
    print ("is "+labels[top_k[0]]+": "+str(round(float(results[top_k[0]])*100,2)) + "% ")
    return ("{ \""+labels[top_k[0]]+"\": \""+str(round(float(results[top_k[0]])*100,2)) + "%\" }")
  else:
    print ("is not sure " + str(round(float(results[top_k[0]])*100,2)) + "%")
    return ("{ \"ToBeVerify\": \""+str(round(float(results[top_k[0]])*100,2)) + "%\" }")

# go through all the files in the folder
def fileGothrough(inputPath, recursive = False):
  global fileCount
  if os.path.isfile(inputPath):
    print(imageProcess(inputPath))
  elif os.path.isdir(inputPath):
    for imgfile in os.listdir(inputPath):
      try:
        if os.path.isfile(inputPath+"\\"+imgfile):
          imageProcess(inputPath+"\\"+imgfile)
          fileCount += 1
        elif recursive:
          fileGothrough(inputPath+"\\"+imgfile, True)
      except:
        print("is not a image file")
  else:
    print("File or Folder doesn't exist")

#Test run
if __name__ == "__main__": 
  if len(sys.argv) == 3 and sys.argv[2] in ("-s","/s"):
    load()
    fileGothrough(sys.argv[1], True)
    print(str(fileCount) + " image files, taks " + str(round(time.time()-ts,3)) + "s total")
  elif len(sys.argv) == 2:
    if sys.argv[1] in ("-h","/h","--h","--help"):
      print ("\npython "+__file__+ " <File or Folder Path> [-s|/s]\n")
      print ("    -h or /h\tfor help")
      print ("    -s or /s\twill scan files in folder and subfolders\n")
      print ("eg.\tpython "+__file__+ " c:\\tmp\\me.jpg\t for single picture")
      print ("\tpython "+__file__+ " c:\\tmp\t\t for the files in folder")
      print ("\tpython "+__file__+ " c:\\tmp -s\t for recursive")
    else:
      load()
      fileGothrough(sys.argv[1])
      print(str(fileCount) + " image files, takes " + str(round(time.time()-ts,3)) + "s total")
  else:
    print ("Please try python "+__file__+ " -h for help")

