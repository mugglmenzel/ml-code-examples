#!/bin/bash
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -e

cd "$(dirname "$0")"

usage() { 
cat << EOF
Usage: $0 -e <experiment> [-E <epochs>] -c <config> -j <jobdir> -r <region> [-p <project>] [-d <Dockerfile>] [-R <registry>] [-t] -s

Bundles code in trainer/ folder as a container image and launches a Vertex AI hyperparameter tuning job with the container.
For the training job, the configuration is read from the provided config files.
        
-e | --experiment       Experiment name which is used as container image name and training job name (Note: - is replaced by _).
-c | --config           Configuration file for the Cloud AI Platform Training job.
-j | --jobdir           GCP bucket to exptect outputs & results from the training program.
-r | --region           GCP region to launch training job in (Note: choose depending on which resources used in the config file).
-p | --project          GCP project id to use for build and training job.
-d | --dockerfile       Dockerfile to build the container image.
-R | --registry         Container registry to push the container image to (Note: Cloud Build needs permissions).
-t | --tensorboard      Indicate if you want metrics exported to a Tensorboard instance.
-s | --service-account  Service account to launch workers with.
EOF
exit 1
}

main() {
    echo "Launching experiment" $EXPERIMENT_NAME "..."

    TIMESTAMP=$(date +%s)
    IMAGE_URI=$CONTAINER_REGISTRY/$PROJECT/$EXPERIMENT_NAME:$TIMESTAMP
    JOB_NAME="${EXPERIMENT_NAME//-/_}_$TIMESTAMP"
    
    echo "Building trainer code" $EXPERIMENT_NAME:$TIMESTAMP "..."
    gcloud builds submit \
        --substitutions=_TRAINER_NAME=$EXPERIMENT_NAME,_REGISTRY=$CONTAINER_REGISTRY,_DOCKERFILE=$DOCKERFILE,TAG_NAME=$TIMESTAMP \
        --project=$PROJECT --gcs-log-dir=$JOB_DIR/$JOB_NAME/build/
        

    if [ "$WITH_TENSORBOARD" = true ] ; then
        echo "Creating tensorboard instance" $JOB_NAME "..."
        gcloud ai tensorboards create --display-name=$JOB_NAME --project=$PROJECT --region=$REGION
        TENSORBOARD=$(gcloud ai tensorboards list --region $REGION --filter=displayName=$JOB_NAME --format="value(name)")
        python3 prepare-job-config.py --config $CONFIG \
            --container-image-uri $IMAGE_URI --job-dir $JOB_DIR/$JOB_NAME/ \
            --tensorboard $TENSORBOARD --out-config /tmp/$JOB_NAME.yaml
    else
        python3 prepare-job-config.py --config $CONFIG \
            --container-image-uri $IMAGE_URI --job-dir $JOB_DIR/$JOB_NAME/ \
            --out-config /tmp/$JOB_NAME.yaml
    fi


    echo "Launching job" $JOB_NAME "with output folder" $JOB_DIR "/" $JOB_NAME "/..."
    gcloud ai hp-tuning-jobs create --display-name $JOB_NAME \
        --project=$PROJECT --region=$REGION \
        --config=/tmp/$JOB_NAME.yaml --service-account=$SERVICE_ACCOUNT \
        --max-trial-count=30 --parallel-trial-count=3

}

PROJECT=$(gcloud config get-value project 2> /dev/null)
DOCKERFILE="Dockerfile"
CONTAINER_REGISTRY="eu.gcr.io"
WITH_TENSORBOARD=false
SERVICE_ACCOUNT=$(gcloud config get-value account 2> /dev/null)

options=$(getopt e:B:E:c:j:r:p:d:R:ts: "$@")
eval set -- "$options"

while true
do
    case $1 in
    -e|--experiment)
        shift
        EXPERIMENT_NAME=$1
        ;;
    -c|--config)
        shift
        CONFIG=$1
        ;;
    -j|--jobdir) 
        shift
        JOB_DIR=$1
        ;;
    -p|--project) 
        shift
        PROJECT=$1
        ;;
    -r|--region) 
        shift
        REGION=$1
        ;;
    -d|--dockerfile) 
        shift
        DOCKERFILE=$1
        ;;
    -R|--registry) 
        shift
        CONTAINER_REGISTRY=$1
        ;;
    -t|--tensorboard) 
        WITH_TENSORBOARD=true
        ;;
    -s|--service-account) 
        shift
        SERVICE_ACCOUNT=$1
        ;;
    --)
        shift
        break;;
    esac
    shift
done

if [ -z "${EXPERIMENT_NAME}" ] || [ -z "${CONFIG}" ] || [ -z "${JOB_DIR}" ] || [ -z "${REGION}" ]; then
    usage
fi

main