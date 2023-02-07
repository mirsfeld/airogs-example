#!/usr/bin/env bash

./build.sh

echo 'finished building, start saving'

docker save airogs_algorithm | xz -c > airogs_algorithm.tar.xz
# docker save airogs_algorithm > airogs_algorithm.tar.xz
# docker save --output airogs_algorithm.tar airogs_algorithm
