from pydantic import BaseModel, Field

class EssayEvaluation(BaseModel):
    c1: int = Field(description="Score for competency 1 (0-200)")
    c2: int = Field(description="Score for competency 2 (0-200)")
    c3: int = Field(description="Score for competency 3 (0-200)")
    c4: int = Field(description="Score for competency 4 (0-200)")
    c5: int = Field(description="Score for competency 5 (0-200)")
    total_score: int = Field(description="Total score (sum of all competencies)")
    id: int = Field(description="The ID (index) of the essay in the CSV")
