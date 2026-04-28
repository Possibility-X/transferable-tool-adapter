import torch


def main():
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("arch list:", torch.cuda.get_arch_list())

    x = torch.ones(8, device="cuda")
    y = x + 1
    torch.cuda.synchronize()

    print("cuda kernel test:", y.tolist())

    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) != (12, 0):
        print("Warning: this does not look like sm_120, but CUDA works.")
    else:
        print("sm_120 detected and CUDA kernel execution works.")


if __name__ == "__main__":
    main()
