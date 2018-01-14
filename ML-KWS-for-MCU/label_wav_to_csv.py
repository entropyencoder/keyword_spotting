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
#
# Modifications Copyright 2017 Arm Inc. All Rights Reserved. 
# Adapted from freeze.py to run inference on train/val/test dataset on the 
# trained model in the form of checkpoint
#          
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import tensorflow as tf
import input_data
import models

from six.moves import xrange

import numpy as np
import csv

#########################################
# Temporarily copied                    #
#########################################
import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
# from tensorflow.python.util import compat

import random
# import re
# import sys
# import tarfile

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
# SILENCE_LABEL = '_silence_'
SILENCE_LABEL = 'silence'
SILENCE_INDEX = 0
# UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_LABEL = 'unknown'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185

class_labels = ('silence','unknown','yes','no','up','down','left','right','on','off','stop','go')

class AudioProcessor(object):
  """Handles loading, partitioning, and preparing audio training data."""

  def __init__(self, data_url, data_dir, silence_percentage, unknown_percentage,
               wanted_words, validation_percentage, testing_percentage,
               model_settings):
    self.data_dir = data_dir
    # self.maybe_download_and_extract_dataset(data_url, data_dir)
    self.prepare_data_index(silence_percentage, unknown_percentage,
                            wanted_words, validation_percentage,
                            testing_percentage)
    # self.prepare_background_data()
    self.prepare_processing_graph(model_settings)

  # def maybe_download_and_extract_dataset(self, data_url, dest_directory):
  #   """Download and extract data set tar file.
  #
  #   If the data set we're using doesn't already exist, this function
  #   downloads it from the TensorFlow.org website and unpacks it into a
  #   directory.
  #   If the data_url is none, don't download anything and expect the data
  #   directory to contain the correct files already.
  #
  #   Args:
  #     data_url: Web location of the tar file containing the data set.
  #     dest_directory: File path to extract data to.
  #   """
  #   if not data_url:
  #     return
  #   if not os.path.exists(dest_directory):
  #     os.makedirs(dest_directory)
  #   filename = data_url.split('/')[-1]
  #   filepath = os.path.join(dest_directory, filename)
  #   if not os.path.exists(filepath):
  #
  #     def _progress(count, block_size, total_size):
  #       sys.stdout.write(
  #           '\r>> Downloading %s %.1f%%' %
  #           (filename, float(count * block_size) / float(total_size) * 100.0))
  #       sys.stdout.flush()
  #
  #     try:
  #       filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
  #     except:
  #       tf.logging.error('Failed to download URL: %s to folder: %s', data_url,
  #                        filepath)
  #       tf.logging.error('Please make sure you have enough free space and'
  #                        ' an internet connection')
  #       raise
  #     print()
  #     statinfo = os.stat(filepath)
  #     tf.logging.info('Successfully downloaded %s (%d bytes)', filename,
  #                     statinfo.st_size)
  #   tarfile.open(filepath, 'r:gz').extractall(dest_directory)

  def prepare_data_index(self, silence_percentage, unknown_percentage,
                         wanted_words, validation_percentage,
                         testing_percentage):
    """Prepares a list of the samples organized by set and label.

    The training loop needs a list of all the available data, organized by
    which partition it should belong to, and with ground truth labels attached.
    This function analyzes the folders below the `data_dir`, figures out the
    right
    labels for each file based on the name of the subdirectory it belongs to,
    and uses a stable hash to assign it to a data set partition.

    Args:
      silence_percentage: How much of the resulting data should be background.
      unknown_percentage: How much should be audio outside the wanted classes.
      wanted_words: Labels of the classes we want to be able to recognize.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      Dictionary containing a list of file information for each set partition,
      and a lookup map for each class to determine its numeric index.

    Raises:
      Exception: If expected files are not found.
    """
    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)
    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
      wanted_words_index[wanted_word] = index + 2
    # self.data_index = {'validation': [], 'testing': [], 'training': []}
    # unknown_index = {'validation': [], 'testing': [], 'training': []}
    self.data_index = {'testing': []}
    # unknown_index = {'testing': []}
    all_words = {}
    # Look through all the subfolders to find audio samples
    # search_path = os.path.join(self.data_dir, '*', '*.wav')
    search_path = os.path.join(self.data_dir, '*.wav')
    for wav_path in gfile.Glob(search_path):
      # _, word = os.path.split(os.path.dirname(wav_path))
      # word = word.lower()
      # Treat the '_background_noise_' folder as a special case, since we expect
      # it to contain long audio samples we mix in to improve training.
      # if word == BACKGROUND_NOISE_DIR_NAME:
      #   continue
      # all_words[word] = True
      # set_index = which_set(wav_path, validation_percentage, testing_percentage)
      # set_index = 'testing'
      # If it's a known class, store its detail, otherwise add it to the list
      # we'll use to train the unknown label.
      # if word in wanted_words_index:
      #   self.data_index[set_index].append({'label': word, 'file': wav_path})
      # else:
      #   unknown_index[set_index].append({'label': word, 'file': wav_path})
      self.data_index['testing'].append({'file':wav_path})

    # if not all_words:
    #   raise Exception('No .wavs found at ' + search_path)
    if not self.data_index['testing']:
      raise Exception('No .wavs found at ' + search_path)
    # for index, wanted_word in enumerate(wanted_words):
    #   if wanted_word not in all_words:
    #     raise Exception('Expected to find ' + wanted_word +
    #                     ' in labels but only found ' +
    #                     ', '.join(all_words.keys()))
    # # We need an arbitrary file to load as the input for the silence samples.
    # # It's multiplied by zero later, so the content doesn't matter.
    # silence_wav_path = self.data_index['training'][0]['file']
    # for set_index in ['validation', 'testing', 'training']:
    #   set_size = len(self.data_index[set_index])
    #   silence_size = int(math.ceil(set_size * silence_percentage / 100))
    #   for _ in range(silence_size):
    #     self.data_index[set_index].append({
    #         'label': SILENCE_LABEL,
    #         'file': silence_wav_path
    #     })
    #   # Pick some unknowns to add to each partition of the data set.
    #   random.shuffle(unknown_index[set_index])
    #   unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
    #   self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
    set_size = len(self.data_index['testing'])

    # # Make sure the ordering is random.
    # for set_index in ['validation', 'testing', 'training']:
    #   random.shuffle(self.data_index[set_index])
    # Prepare the rest of the result data structure.
    # self.words_list = prepare_words_list(wanted_words)
    self.words_list = [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words
    self.word_to_index = {}
    # for word in all_words:
    #   if word in wanted_words_index:
    #     self.word_to_index[word] = wanted_words_index[word]
    #   else:
    #     self.word_to_index[word] = UNKNOWN_WORD_INDEX
    self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

  # def prepare_background_data(self):
  #   """Searches a folder for background noise audio, and loads it into memory.
  #
  #   It's expected that the background audio samples will be in a subdirectory
  #   named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
  #   the sample rate of the training data, but can be much longer in duration.
  #
  #   If the '_background_noise_' folder doesn't exist at all, this isn't an
  #   error, it's just taken to mean that no background noise augmentation should
  #   be used. If the folder does exist, but it's empty, that's treated as an
  #   error.
  #
  #   Returns:
  #     List of raw PCM-encoded audio samples of background noise.
  #
  #   Raises:
  #     Exception: If files aren't found in the folder.
  #   """
  #   self.background_data = []
  #   background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
  #   if not os.path.exists(background_dir):
  #     return self.background_data
  #   with tf.Session(graph=tf.Graph()) as sess:
  #     wav_filename_placeholder = tf.placeholder(tf.string, [])
  #     wav_loader = io_ops.read_file(wav_filename_placeholder)
  #     wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
  #     search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME,
  #                                '*.wav')
  #     for wav_path in gfile.Glob(search_path):
  #       wav_data = sess.run(
  #           wav_decoder,
  #           feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
  #       self.background_data.append(wav_data)
  #     if not self.background_data:
  #       raise Exception('No background wav files were found in ' + search_path)

  def prepare_processing_graph(self, model_settings):
    """Builds a TensorFlow graph to apply the input distortions.

    Creates a graph that loads a WAVE file, decodes it, scales the volume,
    shifts it in time, adds in background noise, calculates a spectrogram, and
    then builds an MFCC fingerprint from that.

    This must be called with an active TensorFlow session running, and it
    creates multiple placeholder inputs, and one output:

      - wav_filename_placeholder_: Filename of the WAV to load.
      - foreground_volume_placeholder_: How loud the main clip should be.
      - time_shift_padding_placeholder_: Where to pad the clip.
      - time_shift_offset_placeholder_: How much to move the clip in time.
      - background_data_placeholder_: PCM sample data for background noise.
      - background_volume_placeholder_: Loudness of mixed-in background.
      - mfcc_: Output 2D fingerprint of processed audio.

    Args:
      model_settings: Information about the current model being trained.
    """
    desired_samples = model_settings['desired_samples']
    self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
    wav_decoder = contrib_audio.decode_wav(
        wav_loader, desired_channels=1, desired_samples=desired_samples)
    # Allow the audio sample's volume to be adjusted.
    self.foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
    scaled_foreground = tf.multiply(wav_decoder.audio,
                                    self.foreground_volume_placeholder_)
    # Shift the sample's start position, and pad any gaps with zeros.
    self.time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
    self.time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
    padded_foreground = tf.pad(
        scaled_foreground,
        self.time_shift_padding_placeholder_,
        mode='CONSTANT')
    sliced_foreground = tf.slice(padded_foreground,
                                 self.time_shift_offset_placeholder_,
                                 [desired_samples, -1])
    # Mix in background noise.
    self.background_data_placeholder_ = tf.placeholder(tf.float32,
                                                       [desired_samples, 1])
    self.background_volume_placeholder_ = tf.placeholder(tf.float32, [])
    background_mul = tf.multiply(self.background_data_placeholder_,
                                 self.background_volume_placeholder_)
    background_add = tf.add(background_mul, sliced_foreground)
    background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
    # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
    spectrogram = contrib_audio.audio_spectrogram(
        background_clamp,
        window_size=model_settings['window_size_samples'],
        stride=model_settings['window_stride_samples'],
        magnitude_squared=True)
    if model_settings['use_mfcc'] == True:
      self.mfcc_ = contrib_audio.mfcc(
        spectrogram,
        wav_decoder.sample_rate,
        dct_coefficient_count=model_settings['dct_coefficient_count'])
    else:
      linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=model_settings['dct_coefficient_count'], num_spectrogram_bins=spectrogram.shape[-1].value,
        sample_rate=model_settings['sample_rate'], upper_edge_hertz=7600.0, lower_edge_hertz=80.0)
      self.mfcc_ = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
      self.mfcc_.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

  def set_size(self, mode):
    """Calculates the number of samples in the dataset partition.

    Args:
      mode: Which partition, must be 'training', 'validation', or 'testing'.

    Returns:
      Number of samples in the partition.
    """
    return len(self.data_index[mode])

  def get_data(self, how_many, offset, model_settings, background_frequency,
               background_volume_range, time_shift, mode, sess):
    """Gather samples from the data set, applying transformations as needed.

    When the mode is 'training', a random selection of samples will be returned,
    otherwise the first N clips in the partition will be used. This ensures that
    validation always uses the same samples, reducing noise in the metrics.

    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      offset: Where to start when fetching deterministically.
      model_settings: Information about the current model being trained.
      background_frequency: How many clips will have background noise, 0.0 to
        1.0.
      background_volume_range: How loud the background noise will be.
      time_shift: How much to randomly shift the clips by in time.
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'.
      sess: TensorFlow session that was active when processor was created.

    Returns:
      List of sample data for the transformed samples, and list of labels in
      one-hot form.
    """
    # Pick one of the partitions to choose samples from.
    candidates = self.data_index[mode]
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = max(0, min(how_many, len(candidates) - offset))
    # Data and labels will be populated and returned.
    data = np.zeros((sample_count, model_settings['fingerprint_size']))
    # labels = np.zeros((sample_count, model_settings['label_count']))
    fnames = []
    desired_samples = model_settings['desired_samples']
    # use_background = self.background_data and (mode == 'training')
    # pick_deterministically = (mode != 'training')
    # Use the processing graph we created earlier to repeatedly to generate the
    # final output sample data we'll use in training.
    for i in xrange(offset, offset + sample_count):
      # Pick which audio sample to use.
      # if how_many == -1 or pick_deterministically:
      #   sample_index = i
      # else:
      #   sample_index = np.random.randint(len(candidates))
      sample_index = i
      sample = candidates[sample_index]
      # # If we're time shifting, set up the offset for this sample.
      # if time_shift > 0:
      #   time_shift_amount = np.random.randint(-time_shift, time_shift)
      # else:
      #   time_shift_amount = 0
      # if time_shift_amount > 0:
      #   time_shift_padding = [[time_shift_amount, 0], [0, 0]]
      #   time_shift_offset = [0, 0]
      # else:
      #   time_shift_padding = [[0, -time_shift_amount], [0, 0]]
      #   time_shift_offset = [-time_shift_amount, 0]
      time_shift_amount = 0
      time_shift_padding = [[0, -time_shift_amount], [0, 0]]
      time_shift_offset = [-time_shift_amount, 0]
      input_dict = {
          self.wav_filename_placeholder_: sample['file'],
          self.time_shift_padding_placeholder_: time_shift_padding,
          self.time_shift_offset_placeholder_: time_shift_offset,
      }
      # # Choose a section of background noise to mix in.
      # if use_background:
      #   background_index = np.random.randint(len(self.background_data))
      #   background_samples = self.background_data[background_index]
      #   background_offset = np.random.randint(
      #       0, len(background_samples) - model_settings['desired_samples'])
      #   background_clipped = background_samples[background_offset:(
      #       background_offset + desired_samples)]
      #   background_reshaped = background_clipped.reshape([desired_samples, 1])
      #   if np.random.uniform(0, 1) < background_frequency:
      #     background_volume = np.random.uniform(0, background_volume_range)
      #   else:
      #     background_volume = 0
      # else:
      #   background_reshaped = np.zeros([desired_samples, 1])
      #   background_volume = 0
      background_reshaped = np.zeros([desired_samples, 1])
      background_volume = 0
      input_dict[self.background_data_placeholder_] = background_reshaped
      input_dict[self.background_volume_placeholder_] = background_volume
      # If we want silence, mute out the main sample but leave the background.
      # if sample['label'] == SILENCE_LABEL:
      #   input_dict[self.foreground_volume_placeholder_] = 0
      # else:
      #   input_dict[self.foreground_volume_placeholder_] = 1
      input_dict[self.foreground_volume_placeholder_] = 1
      # Run the graph to produce the output audio.
      data[i - offset, :] = sess.run(self.mfcc_, feed_dict=input_dict).flatten()
      # label_index = self.word_to_index[sample['label']]
      # labels[i - offset, label_index] = 1
      fnames.append(os.path.split(sample['file'])[1])
    # return data, labels
    return data, fnames

  def get_unprocessed_data(self, how_many, model_settings, mode):
    """Retrieve sample data for the given partition, with no transformations.

    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      model_settings: Information about the current model being trained.
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'.

    Returns:
      List of sample data for the samples, and list of labels in one-hot form.
    """
    candidates = self.data_index[mode]
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = how_many
    desired_samples = model_settings['desired_samples']
    words_list = self.words_list
    data = np.zeros((sample_count, desired_samples))
    labels = []
    with tf.Session(graph=tf.Graph()) as sess:
      wav_filename_placeholder = tf.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = contrib_audio.decode_wav(
          wav_loader, desired_channels=1, desired_samples=desired_samples)
      foreground_volume_placeholder = tf.placeholder(tf.float32, [])
      scaled_foreground = tf.multiply(wav_decoder.audio,
                                      foreground_volume_placeholder)
      for i in range(sample_count):
        if how_many == -1:
          sample_index = i
        else:
          sample_index = np.random.randint(len(candidates))
        sample = candidates[sample_index]
        input_dict = {wav_filename_placeholder: sample['file']}
        if sample['label'] == SILENCE_LABEL:
          input_dict[foreground_volume_placeholder] = 0
        else:
          input_dict[foreground_volume_placeholder] = 1
        data[i, :] = sess.run(scaled_foreground, feed_dict=input_dict).flatten()
        label_index = self.word_to_index[sample['label']]
        labels.append(words_list[label_index])
    return data, labels
#################################################################3

def run_inference(wanted_words, sample_rate, clip_duration_ms,
                  window_size_ms, window_stride_ms, dct_coefficient_count,
                  model_architecture, model_size_info, use_mfcc,
                  csv_writer):
  """Creates an audio model with the nodes needed for inference.

  Uses the supplied arguments to create a model, and inserts the input and
  output nodes that are needed to use the graph for inference.

  Args:
    wanted_words: Comma-separated list of the words we're trying to recognize.
    sample_rate: How many samples per second are in the input audio files.
    clip_duration_ms: How many samples to analyze for the audio pattern.
    window_size_ms: Time slice duration to estimate frequencies from.
    window_stride_ms: How far apart time slices should be.
    dct_coefficient_count: Number of frequency bands to analyze.
    model_architecture: Name of the kind of model to generate.
    model_size_info: Model dimensions : different lengths for different models
  """
  
  tf.logging.set_verbosity(tf.logging.INFO)
  sess = tf.InteractiveSession()
  words_list = input_data.prepare_words_list(wanted_words.split(','))
  model_settings = models.prepare_model_settings(
      len(words_list), sample_rate, clip_duration_ms, window_size_ms,
      window_stride_ms, dct_coefficient_count, use_mfcc)

  # audio_processor = input_data.AudioProcessor(
  audio_processor = AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
      FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings)
  
  label_count = model_settings['label_count']
  fingerprint_size = model_settings['fingerprint_size']

  fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')

  logits = models.create_model(
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      FLAGS.model_size_info,
      is_training=False)

  # ground_truth_input = tf.placeholder(
  #     tf.float32, [None, label_count], name='groundtruth_input')

  predicted_indices = tf.argmax(logits, 1)
  # expected_indices = tf.argmax(ground_truth_input, 1)
  # correct_prediction = tf.equal(predicted_indices, expected_indices)
  # confusion_matrix = tf.confusion_matrix(
  #     expected_indices, predicted_indices, num_classes=label_count)
  # evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  models.load_variables_from_checkpoint(sess, FLAGS.checkpoint)

  # # training set
  # set_size = audio_processor.set_size('training')
  # tf.logging.info('set_size=%d', set_size)
  # total_accuracy = 0
  # total_conf_matrix = None
  # for i in xrange(0, set_size, FLAGS.batch_size):
  #   training_fingerprints, training_ground_truth = (
  #       audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
  #                                0.0, 0, 'training', sess))
  #   training_accuracy, conf_matrix = sess.run(
  #       [evaluation_step, confusion_matrix],
  #       feed_dict={
  #           fingerprint_input: training_fingerprints,
  #           ground_truth_input: training_ground_truth,
  #       })
  #   batch_size = min(FLAGS.batch_size, set_size - i)
  #   total_accuracy += (training_accuracy * batch_size) / set_size
  #   if total_conf_matrix is None:
  #     total_conf_matrix = conf_matrix
  #   else:
  #     total_conf_matrix += conf_matrix
  # tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  # tf.logging.info('Training accuracy = %.2f%% (N=%d)' %
  #                 (total_accuracy * 100, set_size))
  #
  #
  # # validation set
  # set_size = audio_processor.set_size('validation')
  # tf.logging.info('set_size=%d', set_size)
  # total_accuracy = 0
  # total_conf_matrix = None
  # for i in xrange(0, set_size, FLAGS.batch_size):
  #   validation_fingerprints, validation_ground_truth = (
  #       audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
  #                                0.0, 0, 'validation', sess))
  #   validation_accuracy, conf_matrix = sess.run(
  #       [evaluation_step, confusion_matrix],
  #       feed_dict={
  #           fingerprint_input: validation_fingerprints,
  #           ground_truth_input: validation_ground_truth,
  #       })
  #   batch_size = min(FLAGS.batch_size, set_size - i)
  #   total_accuracy += (validation_accuracy * batch_size) / set_size
  #   if total_conf_matrix is None:
  #     total_conf_matrix = conf_matrix
  #   else:
  #     total_conf_matrix += conf_matrix
  # tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  # tf.logging.info('Validation accuracy = %.2f%% (N=%d)' %
  #                 (total_accuracy * 100, set_size))
  
  # test set
  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)
  # total_accuracy = 0
  # total_conf_matrix = None
  expected_classes = []
  for i in xrange(0, set_size, FLAGS.batch_size):
    # test_fingerprints, test_ground_truth = audio_processor.get_data(
    #     FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
    # test_accuracy, conf_matrix = sess.run(
    #     [evaluation_step, confusion_matrix],
    #     feed_dict={
    #         fingerprint_input: test_fingerprints,
    #         ground_truth_input: test_ground_truth,
    #     })
    test_fingerprints, test_fnames = audio_processor.get_data(
        FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
    expected_classes = sess.run(
        predicted_indices,
        feed_dict={
            fingerprint_input: test_fingerprints,
        })
    # batch_size = min(FLAGS.batch_size, set_size - i)
    # print ("i, len(expeceted_classes), len(test_fnames)=", i, len(expected_classes), len(test_fnames))
    for j in range(min(FLAGS.batch_size, set_size - i)):
      csv_writer.writerow([test_fnames[j], class_labels[expected_classes[j]]])
    # total_accuracy += (test_accuracy * batch_size) / set_size
    # if total_conf_matrix is None:
    #   total_conf_matrix = conf_matrix
    # else:
    #   total_conf_matrix += conf_matrix
  # tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  # tf.logging.info('Test accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
  #                                                          set_size))

def main(_):

  with open(FLAGS.out_csv, 'w') as csvfile:
    pred_writer = csv.writer(csvfile, delimiter=',',
                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    pred_writer.writerow(['fname', 'label'])

    # Create the model, load weights from checkpoint and run on train/val/test
    run_inference(FLAGS.wanted_words, FLAGS.sample_rate,
                  FLAGS.clip_duration_ms, FLAGS.window_size_ms,
                  FLAGS.window_stride_ms, FLAGS.dct_coefficient_count,
                  FLAGS.model_architecture, FLAGS.model_size_info, FLAGS.use_mfcc,
                  pred_writer)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      # default='/tmp/speech_dataset/',
      default='/home/yschoi/work/speech_recognition_challenge/data/test/audio/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--checkpoint',
      type=str,
      default='',
      help='Checkpoint to load the weights from.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='dnn',
      help='What model architecture to use')
  parser.add_argument(
      '--model_size_info',
      type=int,
      nargs="+",
      default=[128,128,128],
      help='Model dimensions - different for various models')
  # Use MFCC as input feature if True, otherwise, use log mel spectrogram
  parser.add_argument('--use_mfcc', dest='use_mfcc', action='store_true')
  parser.add_argument('--no_use_mfcc', dest='use_mfcc', action='store_false')
  parser.set_defaults(use_mfcc=True)
  parser.add_argument(
      '--out_csv', type=str, default='', help='CSV file to have results from folder-wise execution.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
