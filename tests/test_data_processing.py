"""Tests du prétraitement des données."""

import numpy as np
import torch

from core.data_processing import preprocess_img


def test_preprocess_img_returns_tensor_with_expected_shape():
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    tensor = preprocess_img(image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
