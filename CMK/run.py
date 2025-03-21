from CMK import default_args, main
import itertools
import time
from concurrent.futures import ProcessPoolExecutor

# Configuration parameters
DATA_NAMES = ["BBC2", "bbcsport_2view", "CiteSeer", "Cora", "Movies"]
NORMALIZES = [True, False]
DIMS = [4, 8, 16, 32, 64, 128, 256, 512]
LEARNING_RATES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
EPOCHS_SET = [150, 300, 450]

# Kernel configurations
KERNEL_CONFIGS = {
    "Gaussian": {"type": "Gaussian", "t": 1.0},
    "Linear": {"type": "Linear"},
    "Polynomial": {"type": "Polynomial", "a": 1.0, "b": 1.0, "d": 2.0},
    "Sigmoid": {"type": "Sigmoid", "d": 2.0, "c": 0.0},
    "Cauchy": {"type": "Cauchy", "sigma": 1.0},
    "Isolation": {"type": "Isolation", "eta": 10.0, "psi": 8, "t": 200},
}


def run_experiment(params):
    data_name, normalize, dim, lr, epochs, kernel_name = params

    print(
        f"# {data_name}, norm_{normalize}, dim_{dim}, lr_{lr}, epochs_{epochs}, kernel_{kernel_name}"
    )

    args = default_args(data_name, normalize, dim, lr, epochs)
    for key, value in KERNEL_CONFIGS[kernel_name].items():
        args.kernel_options[key] = value

    start_time = time.time()
    result = main(args)
    elapsed_time = time.time() - start_time

    print(f"Completed {kernel_name} in {elapsed_time:.2f}s")
    return result


def main_parallel():
    # Generate all parameter combinations
    all_params = list(
        itertools.product(
            DATA_NAMES,
            NORMALIZES,
            DIMS,
            LEARNING_RATES,
            EPOCHS_SET,
            KERNEL_CONFIGS.keys(),
        )
    )

    print(f"Total experiments to run: {len(all_params)}")

    # Uncomment to use parallel processing
    # with ProcessPoolExecutor(max_workers=8) as executor:
    #     results = list(executor.map(run_experiment, all_params))

    # Sequential processing
    for params in all_params:
        run_experiment(params)


if __name__ == "__main__":
    main_parallel()
