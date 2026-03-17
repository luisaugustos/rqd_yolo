"""Detection model backend implementations.

Import this package to trigger auto-registration of all built-in backends.
"""

from src.detection.backends.yolo import YOLOBackend
from src.detection.backends.rtdetr import RTDETRBackend
from src.detection import registry

registry.register("yolov11", YOLOBackend)
registry.register("yolov12", YOLOBackend)
registry.register("rtdetrv2", RTDETRBackend)
