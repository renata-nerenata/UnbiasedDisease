from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

import PIL
from PIL import Image

from diffusers.utils import BaseOutput, is_torch_available, is_transformers_available


@dataclass
class UnbiasedDiseasePipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    inappropriate_content_detected: Optional[List[bool]]


if is_transformers_available() and is_torch_available():
    from .pipeline_unbiased_diffusion import UnbiasedDiseasePipeline
