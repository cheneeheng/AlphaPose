import torch.multiprocessing as mp
mp.set_start_method('forkserver', force=True)
mp.set_sharing_strategy('file_system')
