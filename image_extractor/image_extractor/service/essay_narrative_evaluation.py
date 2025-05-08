from typing import List, Dict, Any
import re
import time
import pandas as pd
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from image_extractor.config import cfg
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from image_extractor.model.essay_narrative_evaluate import EssayNarrativeEvaluation

PROMPT_INSTRUCTION = """Avalie a redação fornecida a seguir, com base no texto motivador, atribuindo uma nota entre 0 e 5 para cada uma das quatro competências.

Responda apenas com a pontuação de cada competência, sem explicações adicionais.

### Critérios de avaliação:
- Competência 1 - Registro: Avalia o uso adequado da norma-padrão: ortografia, pontuação, concordância e construção de frases. Redações com muitos erros que dificultam a leitura recebem nota menor.
- Competência 2 - Coerência temática: Verifica se o texto segue o tema proposto e apresenta uma sequência lógica de ideias, com começo, meio e fim que façam sentido juntos.
- Competência 3 - Tipologia textual: Julga se o texto é realmente uma narrativa, com elementos como personagem, tempo, lugar, e um enredo com situação inicial, conflito e desfecho.
- Competência 4 - Coesão: Analisa se as partes do texto estão bem conectadas, com uso de pronomes, conectores e organização de frases que facilitem a leitura e a continuidade das ideias.

### Formato da resposta:
c1: [nota de 0 a 5]  
c2: [nota de 0 a 5]  
c3: [nota de 0 a 5]  
c4: [nota de 0 a 5]

Redação a ser avaliada:  
{essay_text}

Texto motivador:  
{prompt_text}
"""

def create_essay_evaluation_chain(chat_model: BaseChatModel):
    if isinstance(chat_model, ChatOpenAI):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Você é um avaliador experiente de redações narrativas, com base nos critérios de correção utilizados em avaliações oficiais do ensino fundamental"),
            ("user", PROMPT_INSTRUCTION)
        ])
    elif isinstance(chat_model, ChatVertexAI):
        prompt_template = ChatPromptTemplate.from_messages([
            ("user", PROMPT_INSTRUCTION)
        ])
    elif isinstance(chat_model, ChatAnthropic):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Você é um avaliador experiente de redações narrativas, com base nos critérios de correção utilizados em avaliações oficiais do ensino fundamental."),
            ("user", PROMPT_INSTRUCTION)
        ])
    else:
        raise ValueError(f"Model type {type(chat_model)} not supported")
    
    return prompt_template | chat_model.with_structured_output(EssayNarrativeEvaluation)

def execute_essay_evaluation(
    chat_model: BaseChatModel, essay_text: str, prompt_text: str, essay_id: int
) -> Dict[str, Any]:
    start_time = time.time()
    chain = create_essay_evaluation_chain(chat_model)
    evaluation = chain.invoke({
        "essay_text": essay_text,
        "prompt_text": prompt_text
    })
    
    evaluation.id = essay_id
    
    elapsed_time = time.time() - start_time
    result = evaluation.model_dump()
    result["elapsed"] = elapsed_time
    
    return result

def load_essays_from_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def clean_essay_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)  # remove tags HTML como <i>, <t>, etc.
    text = re.sub(r"\s+", " ", text)  # normaliza espaços e quebras de linha
    return text.strip()

class EssayEvaluator:
    def __init__(self, model):
        self.model = model
        
    def evaluate_narrative_essay(self, essay_text: str, prompt_text: str, essay_id: int) -> Dict[str, Any]:
        cleaned_essay = clean_essay_text(essay_text)
        cleaned_prompt = clean_essay_text(prompt_text)
        return execute_essay_evaluation(self.model, cleaned_essay, cleaned_prompt, essay_id)
    
    def evaluate_essays_from_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        df = load_essays_from_csv(csv_path)
        results = []
        
        for idx, row in df.iterrows():
            essay_text = row["essay"]
            prompt_text = row["prompt"]
            essay_id = idx
            
            result = self.evaluate_narrative_essay(essay_text, prompt_text, essay_id)
            results.append(result)
            
        return results

class OpenAiEssayEvaluator(EssayEvaluator):
    def __init__(self):
        super().__init__(cfg.chat_openai)

class VertexAiEssayEvaluator(EssayEvaluator):
    def __init__(self):
        super().__init__(cfg.vertexai_gemini)

class AnthropicEssayEvaluator(EssayEvaluator):
    def __init__(self):
        super().__init__(cfg.chat_anthropic)
