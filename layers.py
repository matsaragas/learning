import numpy as np
import pandas as pd
import torch


x = torch.rand(5, 3)

y = np.zeros((3, 3))

z = {'Olivia': [1, 2, 3], 'Alex': [2, 3, 4]}

z = pd.DataFrame(z)
print(z)