"""
Multi-task model wrappers para clasificación SMAW.

Exports:
- ECAPAMultiTask: ECAPA-TDNN con 3 cabezas de clasificación
- FeedForwardMultiTask: FeedForward con 3 cabezas de clasificación
- XVectorMultiTask: X-Vector con 3 cabezas de clasificación (desde modelo.py)
"""

from modelo_xvector import SMAWXVectorModel
from modelo_ecapa import ECAPAMultiTask
from modelo_feedforward import FeedForwardMultiTask

__all__ = [
    "SMAWXVectorModel",
    "ECAPAMultiTask", 
    "FeedForwardMultiTask",
]
