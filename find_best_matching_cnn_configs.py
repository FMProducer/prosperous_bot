import itertools
from typing import List, Tuple


def count_conv_params(cnn_maps: List[int], cnn_kernels: List[int], in_channels: int) -> int:
    total_params = 0
    input_channels = in_channels

    for out_channels, kernel_size in zip(cnn_maps, cnn_kernels):
        conv_params = (kernel_size * 1) * input_channels * out_channels + out_channels
        total_params += conv_params
        input_channels = out_channels

    return total_params


def find_matching_cnn_configs(
    target_params: int,
    input_channels: int,
    max_layers: int = 4,
    candidate_maps: List[int] = [8, 16, 32, 64, 128],
    candidate_kernels: List[int] = [3, 5, 7],
    tolerance: float = 0.1,
    top_nearest_configs: int = 5,
) -> List[Tuple[List[int], List[int], int]]:
    """
    Selects CNN block configurations that approximate the target number of parameters (target_params).
    Returns: a list of (cnn_maps, cnn_kernels, total_params)
    """

    matches = []

    for num_layers in range(1, max_layers + 1):
        for maps in itertools.product(candidate_maps, repeat=num_layers):
            for kernels in itertools.product(candidate_kernels, repeat=num_layers):
                total = count_conv_params(list(maps), list(kernels), input_channels)
                if abs(total - target_params) / target_params <= tolerance:
                    matches.append((list(maps), list(kernels), total))

    matches.sort(key=lambda x: abs(x[2] - target_params))
    # top n closest configurations
    return matches[:top_nearest_configs]


if __name__ == "__main__":
    # python find_best_matching_cnn_configs.py
    # Example: the main network has 102,000 parameters in the CNN part, with 7 input channels

    target = 102_000
    input_channels = 7
    max_layers = 3
    results = find_matching_cnn_configs(target, input_channels, max_layers)

    print("CNN configuration search")
    for maps, kernels, total in results:
        print(f"cnn_maps: {maps}, cnn_kernels: {kernels}, parameters: {total}")
