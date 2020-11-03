import torch


class View(torch.nn.Module):
    def __init__(self, *shape):
        """Reshape layer.

        https://discuss.pytorch.org/t/equivalent-of-np-reshape-in-pytorch/144/5

        Examples
        --------

        ```python
            sequential_model = torch.nn.Sequential([Linear(10, 20), View(-1, 5, 4)])
        ```
        """
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)
