import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int] = 1227, set_cudnn: Optional[bool] = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if set_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
