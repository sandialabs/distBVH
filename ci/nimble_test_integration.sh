#!/bin/bash

set -e

if [ $# -lt 1 ]; then
  echo "Required arguments: <test_name>"
  exit 1
fi

export PATH=/opt/view/bin:$PATH

test_name=$1
trace_out_dir=$TRACE_OUTPUT_DIR/$test_name

pushd /opt/builds/NimbleSM/test/contact/$test_name
mpirun -np 2 /opt/builds/NimbleSM/src/NimbleSM \
  $test_name.in \
  --use_vt \
  --vt_trace --vt_trace_dir=$trace_out_dir \
  --vt_lb --vt_lb_name=HierarchicalLB --vt_lb_interval=10 --vt_lb_keep_last_elm

popd  # /opt/builds/NimbleSM/test/contact/$test_name
