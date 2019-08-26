#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/mynet_solver.prototxt $@
