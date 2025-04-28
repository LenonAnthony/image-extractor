from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, Optional, List


class BoundingBox(BaseModel):
    x1: float = Field(description="Left coordinate (0-1)")
    y1: float = Field(description="Top coordinate (0-1)")
    x2: float = Field(description="Right coordinate (0-1)")
    y2: float = Field(description="Bottom coordinate (0-1)")


class DetectionItem(BaseModel):
    name: str = Field(description="The marked option (A, B, C, D or 'invalid')")
    bounding_box: BoundingBox = Field(description="The coordinates of the answer on the image")
    confidence: float = Field(description="Confidence score for the detection (0-1)")


class AnswerSheetExtract(BaseModel):
    detections: Dict[str, DetectionItem] = Field(description="Detections keyed by question number")
    sheet_id: Optional[str] = Field(default=None, description="Optional identifier for the answer sheet")

    def to_detection_format(self) -> Dict:
        return {"detections": {k: v.model_dump() for k, v in self.detections.items()}}


class AnswerSheetExtractWithImage(AnswerSheetExtract):
    path: Path = Field(description="The original path of the image")