"""
Copyright 2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

logging.info(f"Using Tensorflow version {tf.__version__}")

import hypertune

hpt = hypertune.HyperTune()
recorder = {'previous': 0, 'steps': []}

def record(step, writer):
    previous = recorder['steps'][recorder['previous']]['time'] if recorder['previous'] < len(recorder['steps']) else time.time()
    current = time.time()
    logging.info(f"[{step}]: +{current - previous} sec ({current} UNIX)")
    with writer.as_default():
        tf.summary.scalar(step, current, step=0)
    hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag=step,
                metric_value=current)
    recorder['previous'] = len(recorder['steps']) - 1
    recorder['steps'].append({'name': step, 'time': current})
    
    
def summarize_recorder():
    logging.info("Summary of processing steps (in seconds):")
    previous = 0
    for step in recorder['steps']:
        logging.info(f"  Step: {step['name']}, Time: {step['time']}, Duration: {step['time'] - previous}")
        previous = step['time']


class LossReporterCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            print(f"loss: {logs['loss']} in epoch: {epoch}")
            tf.summary.scalar('loss', logs['loss'], step=epoch)
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='loss',
                metric_value=logs['loss'],
                global_step=epoch)

            
def _is_chief(strategy):
    task_type = strategy.cluster_resolver.task_type
    return task_type == 'chief' or task_type is None


def _model_save_path(filename, strategy):
    if strategy.cluster_resolver:
        task_type = strategy.cluster_resolver.task_type
        task_id = strategy.cluster_resolver.task_id
        subfolder = () if _is_chief(strategy) else (str(task_type), str(task_id))
    else:
        subfolder = ()
    return os.path.join(os.environ['AIP_MODEL_DIR'], *subfolder, filename)


def _compile_model(strategy):
    model = keras.Sequential([
        keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)),
        keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='elu'),
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='elu'),
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='elu'),
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='elu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    optimizer_config = {
        'class_name': 'adam',
        'config': {
            'learning_rate': params.learning_rate
        }
    }
    optimizer = tf.keras.optimizers.get(optimizer_config)

    
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model
    
def _train(params, strategy, writer):
    num_workers = strategy.num_replicas_in_sync or 1
    
    TRAIN_BATCH_SIZE = params.batch_size * num_workers
    VAL_BATCH_SIZE = params.batch_size * num_workers
    logging.info(f"Running with {TRAIN_BATCH_SIZE} train batch size and {VAL_BATCH_SIZE} validation batch size.")
        
    (train_data, val_data), mnist_info = tfds.load("mnist", 
                                                   try_gcs=True,
                                                   with_info=True,
                                                   split=['train', 'test'], 
                                                   as_supervised=True)

    @tf.function
    def norm_data(image, label): 
        return tf.cast(image, tf.float32) / 255., label
    
    TRAIN_STEPS_EPOCH = int(mnist_info.splits['train'].num_examples // TRAIN_BATCH_SIZE)
    VAL_STEPS_EPOCH = int(mnist_info.splits['test'].num_examples // VAL_BATCH_SIZE)
    logging.info(f"Running with {TRAIN_STEPS_EPOCH} train steps and {VAL_STEPS_EPOCH} validation steps.")
    
    ds_options = tf.data.Options()
    ds_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    train_ds = (train_data
                .with_options(ds_options)
                .map(norm_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .batch(TRAIN_BATCH_SIZE, drop_remainder=True)
                .cache()
                .repeat(params.num_epochs)
                .prefetch(tf.data.experimental.AUTOTUNE))
    val_ds = (val_data
              .with_options(ds_options)
              .map(norm_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
              .batch(VAL_BATCH_SIZE, drop_remainder=True)
              .cache()
              .repeat(params.num_epochs)
              .prefetch(tf.data.experimental.AUTOTUNE))
    record('dataset_ready', writer)
    
    with strategy.scope():
        model = _compile_model(strategy)

    model.summary()
    record('model_ready', writer)

    model.fit(train_ds, validation_data=val_ds, 
              steps_per_epoch=TRAIN_STEPS_EPOCH, validation_steps=VAL_STEPS_EPOCH, 
              epochs=params.num_epochs, 
              callbacks=[
                  LossReporterCallback(),
                  tf.keras.callbacks.TensorBoard(os.environ['AIP_TENSORBOARD_LOG_DIR'], profile_batch=0)
              ])
#tf.keras.callbacks.experimental.BackupAndRestore(os.path.join(params.job_dir, 'backups'))
    record('model_trained', writer)

    model_save_path = _model_save_path('mnist-cnn.model', strategy)
    logging.info(f'Saving model to {model_save_path}.')
    model.save(model_save_path)
    record('model_saved', writer)

    logging.info('Model training complete.')
    record('done', writer)
    
    logging.info(params)
    summarize_recorder()
    
    
def _get_args():
    """Argument parser.
    Returns:
    Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='number of times to go through the data, default=5')
    parser.add_argument(
        '--batch-size',
        default=100,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float,
        help='learning rate for optimizer, default=.01')
    parser.add_argument(
        '--long-runner',
        default='False',
        type=str,
        help='long running job indicator, default=False')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='DEBUG')
    return parser.parse_args()

def _detect_strategy():
    strategy = None
    try:
        logging.info('TPU_CONFIG:' + str(os.environ.get('TPU_CONFIG')))
        logging.info('TF_CONFIG:' + str(os.environ.get('TF_CONFIG')))
        tf_config = json.loads(os.environ.get('TF_CONFIG')) if os.environ.get('TF_CONFIG') else None
        tpu_config = json.loads(os.environ.get('TPU_CONFIG')) if os.environ.get('TPU_CONFIG') else None
        tf_cluster = tf_config['cluster'] if tf_config and 'cluster' in tf_config else {}
        worker_count = len(tf_cluster['worker']) if tf_cluster and 'worker' in tf_cluster else 0

        if tpu_config:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
        elif worker_count > 0:
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
        else:
            strategy = tf.distribute.MirroredStrategy()
    except Exception as e:
        logging.error('Could not detect TF and TPU configuration.' + str(e))
        
    return strategy


def _fix_os_vars():
    if not 'AIP_TENSORBOARD_LOG_DIR' in os.environ:
        os.environ['AIP_TENSORBOARD_LOG_DIR'] = os.environ['AIP_MODEL_DIR']

if __name__ == "__main__":   
    params = _get_args()
    if params:
        tf.get_logger().setLevel(logging.getLevelName(params.verbosity))
        logging.basicConfig(level=logging.getLevelName(params.verbosity))

    strategy = _detect_strategy()
    _fix_os_vars()

    if params and strategy:
        writer = tf.summary.create_file_writer(os.environ['AIP_TENSORBOARD_LOG_DIR'])
        record('program_start', writer)
        logging.info(f'Running training program with strategy:{strategy}')
        _train(params, strategy, writer)
    else:
        logging.error('Could not parse parameters and configuration.')