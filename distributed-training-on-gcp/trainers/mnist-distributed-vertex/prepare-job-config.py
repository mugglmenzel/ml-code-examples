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
from collections import abc
from copy import deepcopy
import yaml

def modifyConfigYaml(config_yaml, params):
    out_yaml = merge(config_yaml, {
        "baseOutputDirectory": {
            "outputUriPrefix": params.job_dir
        }
    })
    
    if params.tensorboard:
        out_yaml = merge(out_yaml, {
            "tensorboard": params.tensorboard
        })
        
    args = []
    if params.batch_size:
        args.append(f"--batch-size={params.batch_size}")
    if params.num_epochs:
        args.append(f"--num-epochs={params.num_epochs}")
        
    pool_specs = []
    for pool_spec in out_yaml["workerPoolSpecs"]:
        pool_specs.append(merge(pool_spec, {
            "containerSpec": {
                "imageUri": params.container_image_uri,
                "args": args
            }
        }))

    out_yaml["workerPoolSpecs"] = pool_specs
    
    return out_yaml

def merge(dict1, dict2):
    ''' Return a new dictionary by merging two dictionaries recursively. '''

    result = deepcopy(dict1)

    for key, value in dict2.items():
        if isinstance(value, abc.Mapping):
            result[key] = merge(result.get(key, {}), value)
        else:
            result[key] = deepcopy(dict2[key])

    return result


def _get_args():
    """Argument parser.
    Returns:
    Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='YAML config to load')
    parser.add_argument(
        '--container-image-uri',
        type=str,
        required=True,
        help='Container image URI for the job')
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='GCS Job directory')
    parser.add_argument(
        '--batch-size',
        type=str,
        required=False,
        help='Batch size')
    parser.add_argument(
        '--num-epochs',
        type=str,
        required=False,
        help='Epochs')
    parser.add_argument(
        '--tensorboard',
        type=str,
        required=False,
        help='Tensorboard instance')
    parser.add_argument(
        '--out-config',
        type=str,
        required=True,
        help='YAML config to write')
    return parser.parse_args()


if __name__ == "__main__":   
    params = _get_args()
    
    out_yaml = {}
    
    with open(params.config) as yf:
        config_yaml = yaml.load(yf, Loader=yaml.FullLoader)
        
        if 'trialJobSpec' in config_yaml:
            config_yaml['trialJobSpec'] = modifyConfigYaml(config_yaml['trialJobSpec'], params)
            out_yaml = config_yaml
        else:
            out_yaml = modifyConfigYaml(config_yaml, params)
        
    with open(params.out_config, 'w') as yo:
        yaml.dump(out_yaml, yo)

    print(f"Wrote to {params.out_config}:\n{yaml.dump(out_yaml)}")
    

