"""
LiveKit EOU SDK for Arabic
"""

from .detector import ArabicEOUDetector
from .plugin import ArabicTurnDetectorModel, ArabicEOUPlugin

__version__ = "0.1.0"
__all__ = [
    "ArabicEOUDetector",
    "ArabicTurnDetectorModel",
    "ArabicEOUPlugin"
]
