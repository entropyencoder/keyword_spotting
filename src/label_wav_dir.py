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
r"""Runs a trained audio graph against a WAVE file and reports the results.

The model, labels and .wav file specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

import os
import csv

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

FLAGS = None


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  """Runs the audio data through the graph and prints predictions."""
  with tf.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    if num_top_predictions > 0:
      # Sort to show labels in order of confidence
      top_k = predictions.argsort()[-num_top_predictions:][::-1]
      for node_id in top_k:
        human_string = labels[node_id]
        score = predictions[node_id]
        print('%s (score = %.5f)' % (human_string, score))
    else: # Batch processing mode
      if num_top_predictions == 0:
        top_k = predictions.argsort()[-1:][::-1]
        print('%s' % labels[top_k[0]])
      else: # num_top_predictions < 0
        top_k = predictions.argsort()[-1:][::-1]
        return labels[top_k[0]]

    return 0


# def label_wav(wav, labels, graph, input_name, output_name, how_many_labels):
def label_wav(wav, wav_dir, labels, graph, input_name, output_name, how_many_labels, out_csv):
  """Loads the model and labels, and runs the inference to print predictions."""
  # if not wav or not tf.gfile.Exists(wav):
  if (not wav or not tf.gfile.Exists(wav)) and (not wav_dir or not tf.gfile.Exists(wav_dir)):
      tf.logging.fatal('Audio file does not exist %s', wav)

  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  labels_list = load_labels(labels)

  # load graph, which is stored in the default session
  load_graph(graph)

  if (wav and tf.gfile.Exists(wav)):
    with open(wav, 'rb') as wav_file:
      wav_data = wav_file.read()

    run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)

  if (wav_dir and tf.gfile.Exists(wav_dir)):
    directory = os.fsencode(wav_dir)

    if out_csv == None:
      if wav_dir[-1]=="/":
        csv_file_name = wav_dir[:-1] + ".csv"
      else:
        csv_file_name = wav_dir + ".csv"
    else:
      csv_file_name = out_csv

    # with open('pred_all.csv', 'w') as csvfile:
    with open(csv_file_name, 'w') as csvfile:
      pred_writer = csv.writer(csvfile, delimiter=',',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
      pred_writer.writerow(['fname','label'])

      for file in os.listdir(directory):
        cur_file = os.fsdecode(file)
        if cur_file.endswith(".wav"):
          with open(wav_dir+"/"+cur_file, 'rb') as wav_file:
            wav_data = wav_file.read()
          pred = run_graph(wav_data, labels_list, input_name, output_name, -1) #how_many_labels)
          pred_writer.writerow([cur_file, pred])
        else:
          continue


def main(_):
  """Entry point for script, converts flags to arguments."""
  label_wav(FLAGS.wav, FLAGS.wav_dir, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
            FLAGS.output_name, FLAGS.how_many_labels, FLAGS.out_csv)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--wav', type=str, default='', help='Audio file to be identified.')
  parser.add_argument(
      '--wav_dir', type=str, default='', help='Folder having audio files to be identified.')
  parser.add_argument(
      '--out_csv', type=str, default='', help='CSV file to have results from folder-wise execution.')
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument(
      '--input_name',
      type=str,
      default='wav_data:0',
      help='Name of WAVE data input node in model.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of node outputting a prediction in the model.')
  parser.add_argument(
      '--how_many_labels',
      type=int,
      default=3,
      help='Number of results to show.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
