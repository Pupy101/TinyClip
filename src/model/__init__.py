"""
Module with CLIP Vision part of CLIP models and wrappers for LaBSE from
hugging face
"""

from .clip import CLIP, VisionPartCLIP
from .wrappers import WrapperModelFromHuggingFace, VisionModelPreparator
