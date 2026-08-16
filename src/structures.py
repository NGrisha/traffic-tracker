from dataclasses import dataclass


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple


@dataclass
class TrackedObjects:
    track_id: int
    class_id: int
    # class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]