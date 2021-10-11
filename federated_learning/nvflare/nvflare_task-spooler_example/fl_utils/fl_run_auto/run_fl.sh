#!/usr/bin/env bash

n_clients=$1
config=$2
run_number=$3

if test -z "${n_clients}"
then
      echo "Please provide the number of clients, e.g. ./run_fl.sh 2 config_name 1"
      exit 1
fi

if test -z "${config}"
then
      echo "Please provide the configuration folder name, e.g. ./run_fl.sh 2 config_name 1"
      exit 1
fi

if test -z "${run_number}"
then
      echo "Please provide the run number, e.g. ./run_fl.sh 2 config_name 1"
      exit 1
fi

## Run FL training ##
echo "Run FL training"
python3 ${projectpath}/fl_utils/fl_run_auto/run_fl.py --nr_clients ${n_clients} \
  --config "../../../${config}" \
  --run_number ${run_number} \
  --admin_dir "${projectpath}/fl_workspace/admin"
