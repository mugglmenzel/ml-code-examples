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
Usage: $0 -e <experiment> [-p <project>] [-w]

Watches a Vertex AI training job.
        
-e      Experiment name which is used as container image name and training job name (Note: - is replaced by _).
-p      GCP project id to use for build and training job.
-w      Indicate if you want to refresh results every 5 sec.
EOF
exit 1
}

fetch() {
gcloud ai-platform jobs describe $EXPERIMENT_NAME --summarize
}

main() {
    echo "Watching experiment" $EXPERIMENT_NAME "..."
    if $WATCH; then 
        while sleep 5
        do
            tput clear
            fetch
        done
    else fetch
    fi
}


PROJECT=$(gcloud config get-value project 2> /dev/null)
WATCH=false

options=$(getopt e:p:w "$@")
eval set -- "$options"

while true
do
case $1 in
-e|--experiment)
    shift
    EXPERIMENT_NAME=$1
    ;;
-p|--project) 
    shift
    PROJECT=$1
    ;;
-w|--watch) 
    WATCH=true
    ;;
--)
    shift
    break;;
esac
shift
done

if [ -z "${EXPERIMENT_NAME}" ]; then
    usage
fi

main