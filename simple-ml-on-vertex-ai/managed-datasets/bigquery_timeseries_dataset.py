from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import textwrap

import pandas as pd
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow_datasets import features as tfds_features

from google.cloud import bigquery


_CITATION = "Michael Menzel, Google"

_DESCRIPTION = "This is an example of a BigQuery time series dataset."

class BQTimeSeriesDatasetConfig(tfds.core.BuilderConfig):

    def __init__(self, 
                 gcp_project, bq_table, features, targets,
                 name='bq_ts_ds', 
                 sequence_len=50, targets_len=5, stride_len=1,
                 split_train=.8, split_val=.1, split_test=.1
                ):

        self.name = f"{name}_{bq_table}_{'_'.join(features)}+{'_'.join(targets)}"
        self.description = _DESCRIPTION
        self.version = tfds.core.Version('1.0.0')
        self.supported_versions = [tfds.core.Version('1.0.0')]
        self.gcp_project = gcp_project
        self.bq_table = bq_table
        self.features = features
        self.targets = targets
        self.sequence_len = sequence_len
        self.targets_len = targets_len
        self.stride_len = stride_len
        self.split_train = split_train
        self.split_val = split_val
        self.split_test = split_test
        
        self._prepare_timeseries()
        
    def _prepare_timeseries(self):
        bqclient = bigquery.Client(project=self.gcp_project)
        query = bqclient.query(f"SELECT {','.join(set(['Timestamp']+self.features+self.targets))} FROM {self.bq_table}")
        self.df = query.to_dataframe().set_index('Timestamp')
        self.df['timeblock'] = (~(self.df.index.to_series().diff() == pd.Timedelta('1m'))).cumsum()
        timeblock_sizes = self.df.groupby('timeblock').size()
        
        relevant_timeblocks = timeblock_sizes[timeblock_sizes > (self.sequence_len*self.stride_len+1)].index
        datasets_timeblocks = []

        for timeblock in relevant_timeblocks:
            ds_timeblock = self.df[self.df['timeblock'] == timeblock]

            datasets_timeblocks.append((
                tf.keras.preprocessing.timeseries_dataset_from_array(
                    ds_timeblock[self.features][:-self.sequence_len], 
                    ds_timeblock[self.targets][self.sequence_len:],
                    batch_size=None,
                    sequence_length=self.sequence_len,
                    sequence_stride=self.stride_len,
                    shuffle=False)))


        self.ds = tf.data.Dataset.from_tensor_slices(datasets_timeblocks).flat_map(lambda x: x).cache()

    def info(self, dataset_builder):
        return tfds.core.DatasetInfo(
            builder=dataset_builder,
            description=_DESCRIPTION,
            features=tfds_features.FeaturesDict({
                'historical': tfds_features.Tensor(shape=(self.sequence_len, len(self.features),), dtype=tf.float64),
                'targets': tfds_features.Tensor(shape=(len(self.targets),), dtype=tf.float64),
            }),
            supervised_keys=('historical', 'targets'),
            citation=_CITATION,
        )

    def split_generators(self, dl_manager, dataset_builder):
        return {
            tfds.Split.TRAIN: self.generate_examples(split=tfds.Split.TRAIN),
            tfds.Split.VALIDATION: self.generate_examples(split=tfds.Split.VALIDATION),
            tfds.Split.TEST: self.generate_examples(split=tfds.Split.TEST),
        }
    
    
    def generate_examples(self, split):
        ds_len = len(list(self.ds))
        
        if split == tfds.Split.TRAIN:
            split_ds = (self.ds
                    .take(math.floor(self.split_train*ds_len))
                    .prefetch(tf.data.AUTOTUNE))
        elif split == tfds.Split.VALIDATION:
            split_ds = (self.ds
                        .skip(math.ceil(self.split_train*ds_len))
                        .take(math.ceil(self.split_val*ds_len))
                        .prefetch(tf.data.AUTOTUNE))
        elif split == tfds.Split.TEST:
            split_ds = (self.ds
                        .skip(math.ceil((self.split_train+self.split_val)*ds_len))
                        .take(math.ceil(self.split_test*ds_len))
                        .prefetch(tf.data.AUTOTUNE))
        
        for i, (x, y) in enumerate(split_ds.as_numpy_iterator()):
            yield f'{split}_{i}', {'historical': x.astype(np.float64), 
                                   'targets': y.astype(np.float64)}


class BQTimeSeriesDataset(tfds.core.GeneratorBasedBuilder):
    """BigQuery Time Series Dataset
    """

    BUILDER_CONFIGS = []

    VERSION = tfds.core.Version('1.0.0')

    @staticmethod
    def load(*args, **kwargs):
        return tfds.load('BQTimeSeriesDataset', *args, **kwargs)  # pytype: disable=wrong-arg-count

    MANUAL_DOWNLOAD_INSTRUCTIONS = textwrap.dedent("""\
      manual_dir should point to the files in Aeronet.
      """)

    def _info(self):
        return self.builder_config.info(self)

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return self.builder_config.split_generators(dl_manager, self)

    def _generate_examples(self, **kwargs):
        """Yields examples."""
        return self.builder_config.generate_examples(**kwargs)

#calculate fibonancci numbers
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a