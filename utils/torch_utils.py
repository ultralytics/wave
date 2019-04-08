import torch


def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(force_cpu=False):
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    if not cuda:
        print('Using CPU')
    if cuda:
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        print("Using CUDA device0 (name='%s', total_memory=%gMB)" % (x[0].name, x[0].total_memory/c))
        c = 1024 ** 2  # bytes to MB
        if ng > 0:
            # torch.cuda.set_device(0)  # OPTIONAL: Set your GPU if multiple available
            for i in range(1, ng):
                print("           device%g (name='%s', total_memory=%gMB)" % (i, x[i].name, x[i].total_memory/c))

    return device
