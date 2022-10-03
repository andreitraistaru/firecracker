# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import json
import sys
from os import listdir
from os.path import isfile
import pandas as pd
from tabulate import tabulate

PATH_TO_SAMPLES_DIRECTORY = 'tools/measure_i8042_boot_time_impact/data/'
PATH_FOR_OUTPUT_FOLDER = 'tools/measure_i8042_boot_time_impact/'
PATH_FOR_OUTPUT_FILE = 'tools/measure_i8042_boot_time_impact/data.csv'
SAMPLE_NAME_PREFIX = 'sample_'
SAMPLE_NAME_SUFFIX = '.json'

DATA_DIM_ITERATION = 'Iteration'
DATA_DIM_HOST_KERNEL = 'Host kernel version'
DATA_DIM_HOST_KERNEL_VALUES = ['4.14', '5.10']
DATA_DIM_HOST_ARCH = 'Host arch'
DATA_DIM_HOST_ARCH_VALUES = ['Intel', 'AMD', 'aarch64']

DATA_DIM_TEST = 'Test type'
DATA_DIM_TEST_VALUES = ['no_network', 'with_network', 'initrd']

DATA_DIM_I8042 = 'I8042 state (kernel feature)'
DATA_DIM_I8042_VALUES = ['no_i8042', 'with_i8042']

DATA_DIM_GUEST_KERNEL = 'Guest kernel version'
DATA_DIM_GUEST_KERNEL_VALUES = ['4.14', '5.10']

DATA_DIM_BOOT_TIME = 'Boot time'

DATA_DIMS = [DATA_DIM_ITERATION, DATA_DIM_TEST, DATA_DIM_I8042, DATA_DIM_GUEST_KERNEL, DATA_DIM_BOOT_TIME]


def extract_boot_time(sample_data, test, i8042, kernel):
    # Depends on how we configure pytest save test results
    if test == 'initrd':
        test_item_name = 'integration_tests/performance/test_boottime.py::test_boottime_' + test + '_i8042[ubuntu_' + kernel + '_initrd_' + i8042 + ']'
    else:
        test_item_name = 'integration_tests/performance/test_boottime.py::test_boottime_' + test + '_i8042[ubuntu_' + kernel + '_' + i8042 + ']'
    boot_time = 0

    for test_item in sample_data['test_items']:
        if test_item['name'] == test_item_name:
            boot_time = int(test_item['result'])
            break

    return boot_time


def store_data_from_current_sample(iteration, sample_data, data):
    for test in DATA_DIM_TEST_VALUES:
        for i8042 in DATA_DIM_I8042_VALUES:
            for kernel in DATA_DIM_GUEST_KERNEL_VALUES:
                data.loc[len(data.index)] = {DATA_DIM_ITERATION: iteration,
                                             DATA_DIM_TEST: test,
                                             DATA_DIM_I8042: i8042,
                                             DATA_DIM_GUEST_KERNEL: kernel,
                                             DATA_DIM_BOOT_TIME: extract_boot_time(sample_data, test, i8042, kernel)}


def process_data(data, output_filename):
    data.to_csv(output_filename)

    print(tabulate(data, headers='keys', tablefmt='fancy_grid'))


def main():
    args = sys.argv[1:]

    # Getting all sample files that need to be loaded (and discard the ones for which
    # name breaks the naming convention "sample_<iteration_number_without_leading_zeros>.json")
    sample_file_paths = [path
                         for path
                         in listdir(PATH_TO_SAMPLES_DIRECTORY)
                         if isfile(PATH_TO_SAMPLES_DIRECTORY + path) and re.match(SAMPLE_NAME_PREFIX + r"([1-9]+[0-9]*)" + SAMPLE_NAME_SUFFIX, path)]

    # Creating an empty dataframe for storing data from sample files
    data = pd.DataFrame(columns=DATA_DIMS)

    # Start reading data and saving it to dataframe, file by file
    for path in sample_file_paths:
        iteration = re.findall(r'(\d+)', path)

        if len(iteration) != 1:
            continue
        else:
            iteration = iteration[0]

        sample_file = open(PATH_TO_SAMPLES_DIRECTORY + path)

        sample_data = json.load(sample_file)

        store_data_from_current_sample(iteration, sample_data, data)

    if not args:
        process_data(data, PATH_FOR_OUTPUT_FILE)
    else:
        process_data(data, PATH_FOR_OUTPUT_FOLDER + args[0])


if __name__ == "__main__":
    main()
