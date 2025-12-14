import pytest
import numpy as np
from src.models.lstm_model import create_model


def test_model_forward():
    model = create_model(input_size=7, cfg={'hidden_size':16,'num_layers':1})
    x = np.random.rand(2,32,7).astype('float32')
    import torch
    out = model(torch.from_numpy(x))
    assert out.shape == (2,1)
