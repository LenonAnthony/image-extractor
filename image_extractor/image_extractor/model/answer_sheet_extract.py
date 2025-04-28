from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union


class BoundingBox(BaseModel):
    x1: float = Field(..., description="The x coordinate of the top-left corner of the bounding box")
    y1: float = Field(..., description="The y coordinate of the top-left corner of the bounding box")
    x2: float = Field(..., description="The x coordinate of the bottom-right corner of the bounding box")
    y2: float = Field(..., description="The y coordinate of the bottom-right corner of the bounding box")


class QuestionAnswer(BaseModel):
    question_number: int = Field(..., description="The question number")
    selected_option: str = Field(..., description="The selected option (A, B, C, D) or 'invalid' if multiple options selected or none selected")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    bounding_box: BoundingBox = Field(..., description="The bounding box of the question area")


class AnswerSheetExtract(BaseModel):
    total_questions: int = Field(..., description="Total number of questions detected in the answer sheet")
    questions: List[QuestionAnswer] = Field(..., description="List of questions with their answers")
    
    model_config = {
        "extra": "ignore"  # Allow additional fields to be ignored
    }


class AnswerSheetExtractWithImage(AnswerSheetExtract):
    path: str = Field(..., description="Path to the image file")
