from pydantic import BaseModel, Field

class EssayNarrativeEvaluation(BaseModel):
    c1: int = Field(description="Score for competency 1 (0-5)")
    c2: int = Field(description="Score for competency 2 (0-5)")
    c3: int = Field(description="Score for competency 3 (0-5)")
    c4: int = Field(description="Score for competency 4 (0-5)")
    id: int = Field(description="The ID (index) of the essay in the CSV")
