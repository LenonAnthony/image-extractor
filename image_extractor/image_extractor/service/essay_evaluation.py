from typing import List, Dict, Any
import time
from langchain_huggingface import ChatHuggingFace
import pandas as pd
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from image_extractor.config import cfg
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from image_extractor.model.essay_evaluate import EssayEvaluation

PROMPT_INSTRUCTION = """Avalie a redação abaixo de acordo com os cinco critérios estabelecidos, atribuindo uma nota entre 0 e 200 para cada competência, totalizando um máximo de 1000 pontos. A pontuação deve ser dada em intervalos de 40 pontos, e a distribuição deve se aproximar das correções oficiais de redações semelhantes. Considere os seguintes critérios:

Competência 1: Demonstrar domínio da modalidade escrita formal da língua portuguesa.
Competência 2: Compreender a proposta de redação e aplicar conceitos das várias áreas de conhecimento para desenvolver o tema, dentro dos limites estruturais do texto dissertativo-argumentativo em prosa.
Competência 3: Selecionar, relacionar, organizar e interpretar informações, fatos, opiniões e argumentos em defesa de um ponto de vista.
Competência 4: Demonstrar conhecimento dos mecanismos linguísticos necessários para a construção da argumentação.
Competência 5: Elaborar proposta de intervenção para o problema abordado, respeitando os direitos humanos.

Raciocine sobre a justificativa da sua resposta, explicando por que você fez as escolhas que realmente fez.
Pense nas etapas passo a passo.

Avalie a redação: {essay_text}

Tema da redação: {prompt_text}"""

def create_essay_evaluation_chain(chat_model: BaseChatModel):
    if isinstance(chat_model, ChatOpenAI):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Você é um avaliador de redações experiente com foco nos critérios de avaliação do ENEM."),
            ("user", PROMPT_INSTRUCTION)
        ])
    elif isinstance(chat_model, ChatVertexAI):
        prompt_template = ChatPromptTemplate.from_messages([
            ("user", PROMPT_INSTRUCTION)
        ])
    elif isinstance(chat_model, ChatAnthropic):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Você é um avaliador de redações experiente com foco nos critérios de avaliação do ENEM."),
            ("user", PROMPT_INSTRUCTION)
        ])
    elif isinstance(chat_model, ChatMistralAI):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Você é um avaliador de redações experiente com foco nos critérios de avaliação do ENEM."),
            ("user", PROMPT_INSTRUCTION)
        ])
    elif isinstance(chat_model, ChatOllama):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Você é um avaliador de redações experiente com foco nos critérios de avaliação do ENEM."),
            ("user", PROMPT_INSTRUCTION)
        ])
    elif isinstance(chat_model, ChatHuggingFace):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Você é um avaliador de redações experiente com foco nos critérios de avaliação do ENEM."),
            ("user", PROMPT_INSTRUCTION + """Retorne APENAS AS NOTAS com o seguinte formato, SEM NENHUMA INTRODUÇÃO, EXPLICAÇÃO, JUSTIFICATIVA OU QUALQUER OUTRO TEXTO ADICIONAL. ENVOLVA SUAS NOTAS EXATAMENTE ENTRE OS MARCADORES '---INICIO_NOTAS---' E '---FIM_NOTAS---'.
                ---INICIO_NOTAS---
                C1: [nota]
                C2: [nota]
                C3: [nota]
                C4: [nota]
                C5: [nota]
                ---FIM_NOTAS---
             """),
        ])
        return prompt_template | chat_model  
    else:
        raise ValueError(f"Model type {type(chat_model)} not supported")
    
    return prompt_template | chat_model.with_structured_output(EssayEvaluation)

def parse_huggingface_response(response: str, essay_id: int) -> EssayEvaluation:
    start = response.index("---INICIO_NOTAS---") + len("---INICIO_NOTAS---")
    end = response.index("---FIM_NOTAS---")
    response_section = response[start:end].strip()
    print("response: ", response_section)

    scores = {}
    for line in response_section.split('\n'):
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = int(value.strip())
            scores[key] = value
    
    evaluation = EssayEvaluation(
        c1=scores.get('c1', 0),
        c2=scores.get('c2', 0),
        c3=scores.get('c3', 0),
        c4=scores.get('c4', 0),
        c5=scores.get('c5', 0),
        total_score=scores.get('c1', 0) + scores.get('c2', 0) + scores.get('c3', 0) + scores.get('c4', 0) + scores.get('c5', 0),
        id=essay_id
    )
    
    return evaluation

def execute_essay_evaluation(
    chat_model: BaseChatModel, essay_text: str, prompt_text: str, essay_id: int
) -> Dict[str, Any]:
    start_time = time.time() 
    chain = create_essay_evaluation_chain(chat_model)
    if isinstance(chat_model, ChatHuggingFace):
        response = chain.invoke({
            "essay_text": essay_text,
            "prompt_text": prompt_text
        })
        evaluation = parse_huggingface_response(response.content, essay_id)
    else:
        evaluation = chain.invoke({
            "essay_text": essay_text,
            "prompt_text": prompt_text
        })
        
        evaluation.total_score = evaluation.c1 + evaluation.c2 + evaluation.c3 + evaluation.c4 + evaluation.c5
        evaluation.id = essay_id
    
    elapsed_time = time.time() - start_time
    result = evaluation.model_dump()
    result["elapsed"] = elapsed_time
    
    return result

def load_essays_from_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

class EssayEvaluator:
    def __init__(self, model):
        self.model = model
        
    def evaluate_essay(self, essay_text: str, prompt_text: str, essay_id: int) -> Dict[str, Any]:
        return execute_essay_evaluation(self.model, essay_text, prompt_text, essay_id)
    
    def evaluate_essays_from_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        df = load_essays_from_csv(csv_path)
        results = []
        
        for idx, row in df.iterrows():
            if idx == 0 and "text" in row and "prompt" in row:
                continue
                
            essay_text = row["text"]
            prompt_text = row["prompt"]
            essay_id = idx
            
            result = self.evaluate_essay(essay_text, prompt_text, essay_id)
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

class MistralEssayEvaluator(EssayEvaluator):
    def __init__(self):
        super().__init__(cfg.chat_mistral)

class OllamaEssayEvaluator(EssayEvaluator):
    def __init__(self):
        super().__init__(cfg.chat_ollama)

class HuggingFaceEssayEvaluator(EssayEvaluator):
    def __init__(self):
        super().__init__(cfg.chat_huggingface)
