from pathlib import Path
from pydantic import BaseModel, Field


class TextExtract(BaseModel):
    main_text: str = Field(description="The main text on the file")

class TextExtractWithImage(TextExtract):
    path: Path = Field(description="The original path of the image")
