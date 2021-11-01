from numba import njit, prange
import numpy as np
from itertools import combinations


NUM_KERNELS = 84
FILTER_LEN = 9


indices = np.array([_ for _ in combinations(np.arange(FILTER_LEN), FILTER_LEN//3)], dtype=np.int32)
FILTERS = -np.ones((FILTER_LEN, NUM_KERNELS), dtype=np.int32)
for i in range(NUM_KERNELS):
    for c in indices[i]:
        FILTERS[c][i] = 2


@njit("Tuple((float32[:],int32[:],float32[:],int32[:],float32))(float32[:],int32[:],float32[:],int32[:])", fastmath=True, parallel=True, cache=True)
def _sort_input_arrays(timestamps, channels, values, examples_s):

    num_examples = examples_s.shape[0]-1
    min_timestamp_diffs = np.zeros(num_examples, dtype=np.float32)

    for example_index in prange(num_examples):
        example_start = examples_s[example_index]
        example_end = examples_s[example_index+1]

        sorting_indices = np.argsort(timestamps[example_start:example_end])
        timestamps[example_start:example_end] = timestamps[example_start:example_end][sorting_indices]
        channels[example_start:example_end] = channels[example_start:example_end][sorting_indices]
        values[example_start:example_end] = values[example_start:example_end][sorting_indices]
        min_timestamp_diff = timestamps[example_end-1] - timestamps[example_start]
        for i in range(example_start, example_end-1):
            timestamp_diff = timestamps[i+1] - timestamps[i]
            if timestamp_diff > 0 and timestamp_diff < min_timestamp_diff:
                min_timestamp_diff = timestamp_diff
        min_timestamp_diffs[example_index] = min_timestamp_diff

    min_dilation = np.min(min_timestamp_diffs)
    return timestamps, channels, values, examples_s, min_dilation


def _fit_dilations(max_dilation, min_dilation, num_dilations, all_are_integers):
    max_exponent = np.log2(max_dilation + 1)
    min_exponent = np.log2(min_dilation + 1)
    if all_are_integers:
        dilations = (np.logspace(min_exponent, max_exponent, num_dilations, base=2).astype(
            np.float32) - 1).astype(np.int32).astype(np.float32)
    else:
        dilations = (np.logspace(min_exponent, max_exponent, num_dilations, base=2).astype(
            np.float32) - 1)
    return dilations


@njit("Tuple((float32[:],float32[:],int32[:],int32[:],int32[:]))(float32[:],int32[:],float32[:],int32[:],float32)", fastmath=True, parallel=True, cache=True)
def _create_data_structure(timestamps, channels, values, examples_s, dilation):

    num_examples = examples_s.shape[0]-1
    full_size = timestamps.shape[0]*FILTER_LEN
    infinite_priority = 100*max(timestamps)

    T = np.zeros(full_size, dtype=np.float32)
    V = np.zeros(full_size, dtype=np.float32)
    C = np.zeros(full_size, dtype=np.int32)
    W = np.zeros(full_size, dtype=np.int32)

    E = (examples_s * FILTER_LEN).astype(np.int32)

    for example_index in prange(num_examples):
        input_start = examples_s[example_index]
        input_end = examples_s[example_index+1]

        dilation_shifts = np.arange(FILTER_LEN)*dilation

        priority_queue = np.arange(FILTER_LEN).astype(np.int32)
        priority_queue_iters = np.ones(FILTER_LEN).astype(np.int32) * input_start
        priority_queue_T = dilation_shifts + timestamps[input_start]
        priority_queue_C = np.ones(FILTER_LEN).astype(np.int32) * channels[input_start]
        priority_queue_V = np.ones(FILTER_LEN).astype(np.float32) * values[input_start]

        output_start = E[example_index]
        output_end = E[example_index+1]
        for saving_index in range(output_start, output_end):
            head = priority_queue[0]
            T[saving_index] = priority_queue_T[head]
            C[saving_index] = priority_queue_C[head]
            V[saving_index] = priority_queue_V[head]
            W[saving_index] = head
            priority_queue_iters[head] += 1
            new_element_iter = priority_queue_iters[head]

            if new_element_iter < input_end:
                new_element_priority = dilation_shifts[head] + timestamps[new_element_iter]
                priority_queue_C[head] = channels[new_element_iter]
                priority_queue_V[head] = values[new_element_iter]
            else:
                new_element_priority = infinite_priority

            priority_queue_T[head] = new_element_priority
            i = 0
            while i < FILTER_LEN-1:
                if priority_queue_T[priority_queue[i+1]] > new_element_priority:
                    priority_queue[i] = head
                    break
                else:
                    priority_queue[i] = priority_queue[i+1]
                    i += 1
            if i == FILTER_LEN - 1:
                priority_queue[i] = head

    return T, V, C, W, E


def _quantiles(n):
    return np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32)


@njit("float32[:,:](float32[:],float32[:],int32[:],int32[:],int32[:],int32,float32[:],int32[:],int32,int32[:],int32[:],float32[:])", fastmath=True, parallel=True, cache=True)
def _fit_biases(T, V, C, W, E, dilation_index, dilations, num_repeats_c, num_channels, channel_combinations, channel_combinations_s, quantiles):

    num_dilations = dilations.shape[0]
    max_seq_duration = 8*dilations[num_dilations-1]
    num_repeats = num_repeats_c[dilation_index+1] - num_repeats_c[dilation_index]
    num_examples = E.shape[0] - 1
    max_output_len = np.max(E[1:] - E[:E.shape[0]-1])

    dilation = dilations[dilation_index]
    padding_left = 4*dilation
    padding_right = max_seq_duration + 4*dilation

    biases = np.zeros((num_repeats, NUM_KERNELS), dtype=np.float32)
    biases_index = 0
    combination_index = dilation_index * num_channels

    for kernel_index in prange(NUM_KERNELS):
        p_sum = 0
        memorized_index = 1
        memorized_p_sums = np.zeros(max_output_len+1, dtype=np.float32)
        memorized_p_sums_durations = np.zeros(max_output_len+1, dtype=np.float32)
        last_T = padding_left

        example_index = np.random.randint(num_examples)
        input_start = E[example_index]
        input_end = E[example_index+1]

        for i in range(input_start, input_end):
            if T[i] > padding_right:
                break
            active_kernels_start = channel_combinations_s[combination_index+C[i]]
            active_kernels_end = channel_combinations_s[combination_index+C[i]+1]
            if T[i] >= padding_left:
                if kernel_index in channel_combinations[active_kernels_start:active_kernels_end]:
                    p_sum += FILTERS[W[i]][kernel_index]*V[i]
                    memorized_p_sums[memorized_index] = p_sum
                    memorized_p_sums_durations[memorized_index-1] = T[i] - last_T
                    last_T = T[i]
                    memorized_index += 1
            else:
                if kernel_index in channel_combinations[active_kernels_start:active_kernels_end]:
                    p_sum += FILTERS[W[i]][kernel_index]*V[i]

        quantile_index = num_repeats_c[dilation_index]*NUM_KERNELS + kernel_index*num_repeats
        memorized_p_sums_durations[memorized_index-1] = padding_right - last_T

        if memorized_index == 1:
            biases[biases_index:biases_index+num_repeats, kernel_index] = 0
        else:
            _memorized_p_sums = memorized_p_sums[:memorized_index]
            _memorized_p_sums_durations = memorized_p_sums_durations[:memorized_index]
            sorting_indices = np.argsort(_memorized_p_sums)
            this_quantiles = quantiles[quantile_index:quantile_index+num_repeats]*max_seq_duration
            sorting_quantiles_indices = np.argsort(this_quantiles)
            current_quantile_index = 0
            _memorized_p_sums_durations[:] = _memorized_p_sums_durations[sorting_indices]
            accumulated_quantile = 0
            for i in range(memorized_index):
                accumulated_quantile += _memorized_p_sums_durations[i]
                while current_quantile_index < num_repeats and \
                        this_quantiles[sorting_quantiles_indices[current_quantile_index]] < accumulated_quantile:
                    biases[sorting_quantiles_indices[current_quantile_index], kernel_index] = \
                            _memorized_p_sums[sorting_indices[i]]
                    current_quantile_index += 1

    return biases


@njit("float32[:,:](float32[:],float32[:],int32[:],int32[:],int32[:],int32,float32[:],int32[:],int32,int32[:],int32[:],float32[:,:])", fastmath=True, parallel=True, cache=True)
def _transform(T, V, C, W, E, dilation_index, dilations, num_repeats_c, num_channels, channel_combinations, channel_combinations_s, biases):

    num_dilations = dilations.shape[0]
    num_examples = E.shape[0] - 1
    num_repeats = num_repeats_c[dilation_index+1] - num_repeats_c[dilation_index]

    features = np.zeros((num_examples, num_repeats*NUM_KERNELS), dtype=np.float32)

    for example_index in prange(num_examples):
        input_start = E[example_index]
        input_end = E[example_index+1]
        p_sums = np.zeros(NUM_KERNELS)
        local_features = np.zeros((num_repeats, NUM_KERNELS), dtype=np.float32)

        bias_index = num_repeats_c[dilation_index]
        combination_index = dilation_index * num_channels

        dilation = dilations[dilation_index]
        max_seq_duration = 8*dilations[num_dilations-1]
        if bias_index % 2 == 0:
            padding_left = 4*dilation
            padding_right = max_seq_duration + 4*dilation
        else:
            padding_left = 8*dilation
            padding_right = max_seq_duration

        last_Ts = np.ones(NUM_KERNELS, dtype=np.float32)*padding_left

        for i in range(input_start, input_end):
            if T[i] > padding_right:
                break
            filters_for_this_value = FILTERS[W[i]]
            active_kernels_start = channel_combinations_s[combination_index+C[i]]
            active_kernels_end = channel_combinations_s[combination_index+C[i]+1]
            if T[i] >= padding_left:
                for kernel_index in channel_combinations[active_kernels_start:active_kernels_end]:
                    # this multiplication is a little faster than if inside for
                    local_features[:, kernel_index] += (T[i] - last_Ts[kernel_index]) * \
                        (p_sums[kernel_index] > biases[bias_index:bias_index+num_repeats, kernel_index])

                    p_sums[kernel_index] += filters_for_this_value[kernel_index] * V[i]
                    last_Ts[kernel_index] = T[i]
            else:
                for kernel_index in channel_combinations[active_kernels_start:active_kernels_end]:
                    p_sums[kernel_index] += filters_for_this_value[kernel_index] * V[i]

        for kernel_index in range(NUM_KERNELS):
            local_features[:, kernel_index] += (padding_right - last_Ts[kernel_index]) *\
                (p_sums[kernel_index] > biases[bias_index:bias_index+num_repeats, kernel_index])

        features[example_index, :] = local_features.flatten()
    return features


def fit_and_transform_train(train_timestamps, train_channels, train_values, train_examples_s,
                            max_sequence_duration, num_dilations=32, num_features=10_000):

    train_timestamps, train_channels, train_values, train_examples_s, min_dilation = \
        _sort_input_arrays(train_timestamps, train_channels, train_values, train_examples_s)

    max_dilation = max_sequence_duration / 8
    all_are_integers = np.all(np.around(train_timestamps) == train_timestamps)
    dilations = _fit_dilations(max_dilation, min_dilation, num_dilations, all_are_integers)
    num_repeats = (num_features // (NUM_KERNELS * num_dilations)) * np.ones(num_dilations+1, dtype=np.int32)
    num_repeats[0] = 0
    real_num_features = np.sum(num_repeats)*NUM_KERNELS
    repeats_index = 1
    while(real_num_features + NUM_KERNELS <= num_features):
        num_repeats[repeats_index] += 1
        repeats_index += 1
        real_num_features += NUM_KERNELS
    num_repeats_c = num_repeats.cumsum()

    num_channels = np.max(train_channels)+1
    num_combinations = num_dilations*NUM_KERNELS
    max_num_channels = min(num_channels, FILTER_LEN)
    max_exponent = np.log2(max_num_channels + 1)

    num_channels_per_combination = (2 ** np.random.uniform(0, max_exponent, num_combinations)).astype(np.int32)
    values, counts = np.unique(num_channels_per_combination, return_counts=True)

    num_combinations_of_size = np.zeros(FILTER_LEN, dtype=np.int32)
    for v, c in zip(values, counts):
        num_combinations_of_size[v-1] = c

    channel_combinations = []
    channel_combinations_s = [0]
    chosen_size = 0
    for _ in range(num_dilations):
        # for each channel, create a list of kernels that belong
        # to a combination with this channel. This is the same as
        # in MiniRocketMultivariate, only stored in the inverted way
        channel_to_kernels = [[] for _ in range(num_channels)]
        for kernel_index in range(NUM_KERNELS):
            if num_combinations_of_size[chosen_size] == 0:
                chosen_size += 1
            for chosen_channel in np.random.choice(num_channels, chosen_size+1, replace=False):
                channel_to_kernels[chosen_channel].append(kernel_index)
            num_combinations_of_size[chosen_size] -= 1

        channel_combinations.append(np.concatenate(channel_to_kernels).astype(np.int32))
        for kernels in channel_to_kernels:
            channel_combinations_s.append(channel_combinations_s[-1] + len(kernels))

    channel_combinations = np.concatenate(channel_combinations, dtype=np.int32)
    channel_combinations_s = np.asarray(channel_combinations_s, dtype=np.int32)

    quantiles = _quantiles(real_num_features)
    biases = np.zeros((real_num_features//NUM_KERNELS, NUM_KERNELS), dtype=np.float32)
    num_train_examples = train_examples_s.shape[0] - 1
    train_features = np.zeros((num_train_examples, real_num_features), dtype=np.float32)

    for dilation_index in range(num_dilations):
        dilation = dilations[dilation_index]
        T, V, C, W, E = _create_data_structure(
            train_timestamps, train_channels, train_values, train_examples_s, dilation)

        biases_start = num_repeats_c[dilation_index]
        biases_end = num_repeats_c[dilation_index+1]
        biases[biases_start:biases_end, :] = _fit_biases(
            T, V, C, W, E, dilation_index, dilations, num_repeats_c,
            num_channels, channel_combinations, channel_combinations_s, quantiles)

        features_start = num_repeats_c[dilation_index]*NUM_KERNELS
        features_end = num_repeats_c[dilation_index+1]*NUM_KERNELS
        train_features[:, features_start:features_end] = _transform(
            T, V, C, W, E, dilation_index, dilations, num_repeats_c,
            num_channels, channel_combinations, channel_combinations_s, biases)

    parameters = (dilations, num_repeats_c, num_channels, channel_combinations, channel_combinations_s, biases)

    return train_features, parameters


def transform_test(test_timestamps, test_channels, test_values, test_examples_s, parameters):

    dilations, num_repeats_c, num_channels, channel_combinations, channel_combinations_s, biases = parameters
    real_num_features = biases.shape[0] * biases.shape[1]

    test_timestamps, test_channels, test_values, test_examples_s, _ = \
        _sort_input_arrays(test_timestamps, test_channels, test_values, test_examples_s)

    num_test_examples = test_examples_s.shape[0] - 1
    test_features = np.zeros((num_test_examples, real_num_features), dtype=np.float32)

    num_dilations = len(dilations)
    for dilation_index in range(num_dilations):
        dilation = dilations[dilation_index]
        T, V, C, W, E = _create_data_structure(
            test_timestamps, test_channels, test_values, test_examples_s, dilation)

        features_start = num_repeats_c[dilation_index]*NUM_KERNELS
        features_end = num_repeats_c[dilation_index+1]*NUM_KERNELS
        test_features[:, features_start:features_end] = _transform(
            T, V, C, W, E, dilation_index, dilations, num_repeats_c,
            num_channels, channel_combinations, channel_combinations_s, biases)

    return test_features
