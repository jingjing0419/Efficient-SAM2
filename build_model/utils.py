import torch

def init_generator(device: torch.device, fallback: torch.Generator = None, seed: int = 1234):
    """
    初始化一个随机数生成器（torch.Generator），并根据指定设备类型设置其状态。
    支持通过固定随机种子实现可重复性。

    参数：
        device (torch.device): 指定的设备类型，可以是 "cpu" 或 "cuda"。
        fallback (torch.Generator, 可选): 当设备类型既不是 CPU 也不是 CUDA 时，
                                          使用的回退随机数生成器。如果未提供，将默认使用 CPU。
        seed (int, 可选): 如果提供，则为生成器设置固定的随机种子，以实现可重复性。

    返回：
        torch.Generator: 初始化的随机数生成器，其状态与当前设备默认生成器的状态同步。
    """
    if device.type == "cpu":
        # 如果设备是 CPU，创建一个新的 CPU 生成器，并同步状态
        generator = torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        # 如果设备是 CUDA，创建一个新的 CUDA 生成器，并同步状态
        generator = torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        # 如果设备既不是 CPU 也不是 CUDA，处理回退逻辑
        if fallback is None:
            # 如果没有提供回退生成器，使用 CPU 初始化生成器
            return init_generator(torch.device("cpu"), seed=seed)
        else:
            # 如果提供了回退生成器，直接返回
            return fallback

    # 如果提供了随机种子，则设置生成器的种子
    if seed is not None:
        generator.manual_seed(seed)

    return generator
