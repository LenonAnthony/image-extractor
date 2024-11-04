from pathlib import Path
from pydantic import BaseModel, Field


class TextExtract(BaseModel):
    title: str = Field(description="The perceived title on the image")
    main_text: str = Field(description="The main text on the file")
    objects_in_image: str = Field(description="Any other objects observed in the image")


class TextExtractWithImage(TextExtract):
    path: Path = Field(description="The original path of the image")
