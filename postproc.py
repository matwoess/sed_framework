import numpy as np
from itertools import islice


def post_process_predictions(array: np.ndarray) -> np.ndarray:
    pp_array = array.copy()
    # pp_array = _median_filter_predictions(pp_array, window_length=10)
    pp_array = _normalize_segments(pp_array)
    pp_array = _remove_events_if_background(pp_array)
    return pp_array


def _median_filter_predictions(array: np.ndarray, window_length: int = 10) -> np.ndarray:
    filtered = array.copy()

    def windows(seq, n):
        """Returns a sliding window (of width n) over data from the iterable
         s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ..."""
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    background_idx = array.shape[1] - 1
    for b, batch in enumerate(array):
        for c, class_values in enumerate(batch):
            if c == background_idx:
                continue
            sliding_window = windows(class_values[:len(class_values) - window_length // 2], window_length)
            for i, win in enumerate(sliding_window):
                filtered[b, c, i + window_length - 1] = 1 if np.median(win) == 1 else 0
    return filtered


def _remove_events_if_background(array: np.ndarray) -> np.ndarray:
    background_idx = array.shape[1] - 1
    for b, batch in enumerate(array):
        for c, class_values in enumerate(batch):
            if c == background_idx:
                continue
            for i in range(array.shape[-1]):
                if array[b, background_idx, i] == 1:
                    array[b, c, i] = 0
    return array


def _normalize_segments(array: np.ndarray, min_event_len=0.1, min_gap=0.1, hop_size=512, sr=22050) -> np.ndarray:
    threshold = 0.5
    min_length_indices = int(min_event_len * sr / hop_size)
    min_gap_indices = int(min_gap * sr / hop_size)

    def remove_short_segments(bool_array, min_seg_length, replacement):
        # Find the changes in the bool_array
        change_indices = np.diff(bool_array).nonzero()[0]
        # Shift change_index with one, focus on frame after the change
        change_indices += 1
        if bool_array[0]:
            # If the first element of bool_array is True add 0 at the beginning
            change_indices = np.r_[0, change_indices]
        if bool_array[-1]:
            # If the last element of bool_array is True, add the length of the array
            change_indices = np.r_[change_indices, bool_array.size]
        # Reshape the result into two columns
        segments = change_indices.reshape((-1, 2))

        # remove short segments
        for seg in segments:
            if seg[1] - seg[0] < min_seg_length:
                array[b, c, seg[0]:seg[1] + 1] = replacement

    for b, batch in enumerate(array):
        for c, class_values in enumerate(batch):
            # no changes if no events happened
            if np.all(class_values == 0):
                continue
            event_segments = class_values >= threshold
            remove_short_segments(event_segments, min_length_indices, 0)
            gap_segments = class_values < threshold
            remove_short_segments(gap_segments, min_gap_indices, 1)

    return array


if __name__ == '__main__':
    np.random.seed(1)
    test_excerpt_length = 48
    test_predictions = np.where(np.random.rand(1, 1, test_excerpt_length) >= 0.4, 1, 0)
    filtered_predictions = _median_filter_predictions(test_predictions.copy(), window_length=10)
    print("input")
    print(test_predictions)
    print("output")
    print(filtered_predictions)
