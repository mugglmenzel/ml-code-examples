from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import textwrap
import urllib.parse

import pandas as pd
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow_datasets import features as tfds_features

from google.cloud import bigquery


_CITATION = "Michael Menzel, Google"

_DESCRIPTION = "This is an example of a BigQuery-based image autoencoder dataset."

class VertexAIImageAutoencoderDatasetConfig(tfds.core.BuilderConfig):

    def __init__(self, 
                 filter_label,
                 name='vertexai_image_ae_ds'                 
                ):

        self.name = f'{name}_filter_{filter_label}'
        self.description = _DESCRIPTION
        self.version = tfds.core.Version('1.0.0')
        self.supported_versions = [tfds.core.Version('1.0.0')]
        
        self.filter_label = filter_label

    def info(self, dataset_builder):
        return tfds.core.DatasetInfo(
            builder=dataset_builder,
            description=_DESCRIPTION,
            features=tfds_features.FeaturesDict({
                'original': tfds.features.Image(shape=(480, 640, 3), encoding_format='jpeg'), 
                'encoded': tfds.features.Image(shape=(480, 640, 3), encoding_format='jpeg'),
                             
            }),
            supervised_keys=('original', 'encoded'),
            citation=_CITATION,
        )

    def split_generators(self, dl_manager, dataset_builder):
        return {
            tfds.Split.TRAIN: self.generate_examples(split=tfds.Split.TRAIN),
            tfds.Split.VALIDATION: self.generate_examples(split=tfds.Split.VALIDATION),
            tfds.Split.TEST: self.generate_examples(split=tfds.Split.TEST),
        }
    
    
    def generate_examples(self, split):
        data = {
            tfds.Split.VALIDATION: os.environ.get('AIP_VALIDATION_DATA_URI'),
            tfds.Split.TRAIN: os.environ.get('AIP_TRAINING_DATA_URI'),
            tfds.Split.TEST: os.environ.get('AIP_TEST_DATA_URI')
        }
        
        dataset_json = [json.loads(js) 
                for data_file in tf.io.gfile.glob(data[split]) 
                for js in tf.io.read_file(data_file).numpy().decode('UTF-8').split('\n')]
        dataset_images = {'image': [el['imageGcsUri'] for el in dataset_json if el['classificationAnnotations'][0]['displayName'] == self.filter_label]}
        
        @tf.function
        def load_image(feat):
            return tf.io.decode_jpeg(tf.io.read_file(feat['image']))
            
        split_ds = (tf.data.Dataset
                    .from_tensor_slices(dataset_images)
                    .map(load_image))

        for i, ex in split_ds.enumerate().as_numpy_iterator():
            yield f'{split}_{i}', {'original': ex, 'encoded': ex}


class VertexAIImageAutoencoderDataset(tfds.core.GeneratorBasedBuilder):
    """BigQuery Time Series Dataset
    """

    BUILDER_CONFIGS = []

    VERSION = tfds.core.Version('1.0.0')

    @staticmethod
    def load(*args, **kwargs):
        return tfds.load('VertexAIImageAutoencoderDataset', *args, **kwargs)  # pytype: disable=wrong-arg-count

    MANUAL_DOWNLOAD_INSTRUCTIONS = textwrap.dedent("""\
      """)

    def _info(self):
        return self.builder_config.info(self)

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return self.builder_config.split_generators(dl_manager, self)

    def _generate_examples(self, **kwargs):
        """Yields examples."""
        return self.builder_config.generate_examples(**kwargs)
