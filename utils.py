import torch


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print(f"Using {device} as backend")

    return device

    # return "cpu"
