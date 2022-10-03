#!/usr/bin/env bash

# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

if [ $# -ne 2 ]
  then
    echo "Two and only two arguments (number of iterations and output file name) should be supplied!"
    exit 1
fi

cd "$(dirname "$BASH_SOURCE")/../.."

sudo rm -r test_results &> /dev/null
sudo rm -r tools/measure_i8042_boot_time_impact/data &> /dev/null
sudo rm tools/measure_i8042_boot_time_impact/data.csv &> /dev/null

mkdir tools/measure_i8042_boot_time_impact/data

echo "Start measuring..."

for ((i=1;i<=$1;i++))
do
  echo "Iteration no. $i:"
  ./tools/devtool -y test --  \
    integration_tests/performance/test_boottime.py::test_boottime_no_network_i8042  \
    integration_tests/performance/test_boottime.py::test_boottime_with_network_i8042  \
    integration_tests/performance/test_boottime.py::test_boottime_initrd_i8042

  if [ $? -ne 0 ]
    then
      exit 1
  fi

  sudo mv test_results/report/test_report.json tools/measure_i8042_boot_time_impact/data/sample_$i.json
done

echo "Measuring done! Start processing data..."

yes | (sudo yum install python3-pip || sudo apt install python3-pip)
yes | sudo pip3 install pandas
yes | sudo pip3 install tabulate

python3 tools/measure_i8042_boot_time_impact/data_aggregator.py "$2"

if [ $? -ne 0 ]
  then
    exit 1
fi

echo "Data processed. Uploading results to s3..."

aws s3 mv tools/measure_i8042_boot_time_impact/"$2" s3://buildkite-ci-ccpr/i8042_boot_time_results/

if [ $? -ne 0 ]
  then
    exit 1
fi

sudo rm -r tools/measure_i8042_boot_time_impact/data &> /dev/null
sudo rm -r test_results &> /dev/null

echo "Done!"
