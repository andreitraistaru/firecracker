# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from os import listdir
from os.path import isfile
import pandas as pd
from numpy import mean, std
from tabulate import tabulate

PATH_TO_RESULT_DIRECTORY = 'tools/measure_i8042_boot_time_impact/data/'
OUTPUT_FILENAME = 'tools/measure_i8042_boot_time_impact/merged_results.csv'
STATISTICS_FILENAME = 'tools/measure_i8042_boot_time_impact/statistics.csv'

DATA_DIM_ITERATION = 'Iteration'
DATA_DIM_HOST_KERNEL = 'Host kernel version'
DATA_DIM_HOST_ARCH = 'Host arch'

DATA_DIM_TEST = 'Test type'
DATA_DIM_TEST_VALUES = ['no_network', 'with_network', 'initrd']

DATA_DIM_I8042 = 'I8042 state (kernel feature)'
DATA_DIM_I8042_VALUES = ['no_i8042', 'with_i8042']

DATA_DIM_GUEST_KERNEL = 'Guest kernel version'
DATA_DIM_GUEST_KERNEL_VALUES = ['4.14', '5.10']

DATA_DIM_BOOT_TIME = 'Boot time'

DATA_DIMS = [DATA_DIM_ITERATION, DATA_DIM_HOST_KERNEL, DATA_DIM_HOST_ARCH, DATA_DIM_TEST, DATA_DIM_I8042, DATA_DIM_GUEST_KERNEL, DATA_DIM_BOOT_TIME]


def do_statistics(data):
    data = data.groupby([DATA_DIM_HOST_ARCH, DATA_DIM_HOST_KERNEL, DATA_DIM_GUEST_KERNEL, DATA_DIM_TEST, DATA_DIM_I8042]).agg(Mean=(DATA_DIM_BOOT_TIME, mean), Std=(DATA_DIM_BOOT_TIME, std), Min=(DATA_DIM_BOOT_TIME, min), Max=(DATA_DIM_BOOT_TIME, max), Quantile_50=(DATA_DIM_BOOT_TIME, lambda x: x.quantile(0.5)), Quantile_90=(DATA_DIM_BOOT_TIME, lambda x: x.quantile(0.9))).reset_index()

    print(tabulate(data, headers='keys', tablefmt='fancy_grid'))

    data.to_csv(STATISTICS_FILENAME)


def main():
    # Getting all results files that need to be loaded (and discard the ones for which
    # name breaks the naming convention "<architecture>_<host kernel version>.csv")
    result_file_paths = [path
                         for path
                         in listdir(PATH_TO_RESULT_DIRECTORY)
                         if isfile(PATH_TO_RESULT_DIRECTORY + path)]

    # Creating an empty dataframe for storing data from sample files
    merged_data = pd.DataFrame(columns=DATA_DIMS)

    # Start reading data and saving it to dataframe, file by file
    for path in result_file_paths:
        tokens = path.split("_")

        if len(tokens) != 2:
            continue

        host_arch = tokens[0]
        host_kernel_version = tokens[1]

        current_data = pd.read_csv(PATH_TO_RESULT_DIRECTORY + path)

        current_data[DATA_DIM_HOST_ARCH] = host_arch
        current_data[DATA_DIM_HOST_KERNEL] = host_kernel_version

        merged_data = pd.concat([merged_data, current_data], ignore_index=True)

    merged_data.to_csv(OUTPUT_FILENAME)

    do_statistics(merged_data)


if __name__ == "__main__":
    main()
