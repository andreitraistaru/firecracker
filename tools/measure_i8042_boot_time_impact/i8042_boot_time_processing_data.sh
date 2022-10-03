#!/usr/bin/env bash

# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

if [ $# -ne 0 ]
  then
    echo "Two and only two arguments (number of iterations and output file name) should be supplied!"
    exit 1
fi

cd "$(dirname "$BASH_SOURCE")/../.."

sudo rm -r test_results &> /dev/null
sudo rm -r tools/measure_i8042_boot_time_impact/data &> /dev/null
sudo rm tools/measure_i8042_boot_time_impact/data.csv &> /dev/null

mkdir tools/measure_i8042_boot_time_impact/data

echo "Downloading result files from s3..."

aws s3 cp s3://buildkite-ci-ccpr/i8042_boot_time_results/ tools/measure_i8042_boot_time_impact/data/ --recursive

echo "Download completed! Start doing statistics..."

yes | (sudo yum install python3-pip || sudo apt install python3-pip)
yes | sudo pip3 install numpy
yes | sudo pip3 install pandas
yes | sudo pip3 install tabulate

python3 tools/measure_i8042_boot_time_impact/data_analyzer.py

echo "Data merged. Uploading results to s3..."

aws s3 mv tools/measure_i8042_boot_time_impact/merged_results.csv s3://buildkite-ci-ccpr/i8042_boot_time_results/
aws s3 mv tools/measure_i8042_boot_time_impact/statistics.csv s3://buildkite-ci-ccpr/i8042_boot_time_results/

sudo rm -r tools/measure_i8042_boot_time_impact/data &> /dev/null
sudo rm -r test_results &> /dev/null
echo "Done!"
