"""
BiteBase Restaurant Analytics Models
"""

from .location_change_model import LocationChangeModel
from .realtime_location_change_model import RealtimeLocationChangeModel
from .combined_location_model import CombinedLocationModel

__all__ = [
    'LocationChangeModel',
    'RealtimeLocationChangeModel',
    'CombinedLocationModel'
]

__version__ = '0.1.0'