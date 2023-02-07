#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname string "$0")" ; pwd -P )"

docker build -t airogs_algorithm "$SCRIPTPATH"

# docker build -t airogs_algorithm /Users/monairsfeld/ProjectsAachen/airogs_example/airogs-example-algorithm
