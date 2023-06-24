import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BaseNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = 32
        self.fn1 = nn.dense