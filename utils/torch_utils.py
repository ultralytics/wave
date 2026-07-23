# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch


def init_seeds(seed=0):
    """Initialize random seeds for CPU and GPU operations using a given integer seed value."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(force_cpu=False):
    """Select the appropriate device (CPU or CUDA) based on availability and the force_cpu flag."""
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")

    if not cuda:
        print("Using CPU")
    if cuda:
        c = 1024**2  # bytes to MB
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        print(
            f"Using CUDA device0 _CudaDeviceProperties(name='{x[0].name}', total_memory={int(x[0].total_memory / c)}MB)"
        )
        if ng > 0:
            # torch.cuda.set_device(0)  # OPTIONAL: Set GPU ID
            for i in range(1, ng):
                print(
                    f"           device{i:g} _CudaDeviceProperties(name='{x[i].name}', "
                    f"total_memory={int(x[i].total_memory / c)}MB)"
                )

    return device
