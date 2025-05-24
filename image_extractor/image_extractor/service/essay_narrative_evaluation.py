from typing import List, Dict, Any
import re
import time
import pandas as pd
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from image_extractor.config import cfg
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI, GemmaVertexAIModelGarden, GemmaLocalHF, GemmaChatLocalHF
from langchain_anthropic import ChatAnthropic
from image_extractor.model.essay_narrative_evaluate import EssayNarrativeEvaluation

PROMPT_INSTRUCTION = """Avalie a redação com base no texto motivador, atribuindo notas de 0 a 5 para cada competência, conforme a **Matriz de Correção por Competência**. 
Lembre-se de que você estará no papel de um professor corrigindo redações de alunos do ensino fundamental (5º ao 9º ano).  
**Responda apenas com as notas**, seguindo o formato:  

c1: [nota de 0 a 5]  
c2: [nota de 0 a 5]  
c3: [nota de 0 a 5]  
c4: [nota de 0 a 5]  

#### Critérios Detalhados:  
- **C1 - Registro Formal (Ortografia, Gramática e Pontuação)**  
  - Nível 5: Estrutura morfossintática bem empregada com até 5 desvios pontuais e não recorrentes.  
  - Nível 4: Estrutura morfossintática consistente com desvios pontuais.  
  - Nível 3: Estrutura morfossintática consistente com 1 tipo de desvio recorrente.  
  - Nível 2: Estrutura morfossintática escassa, com poucos desvios.  
  - Nível 1: Estrutura morfossintática escassa com desvios recorrentes ou escrita em nível silábico-alfabético.  

- **C2 - Coerência Temática (Manutenção do Tema e Progressão Textual)**  
  - Nível 5: Progressão textual completa e repertório consistente que ultrapassa a situação motivadora.  
  - Nível 4: Progressão textual completa, com ideias previsíveis, mas detalhadas.  
  - Nível 3: Progressão completa, mas com paráfrases ou ideias previsíveis e sem detalhamento.  
  - Nível 2: Progressão insuficiente ou com cópias da situação motivadora.  
  - Nível 1: Tangenciamento do tema ou falta de progressão textual.  

- **C3 - Tipologia Textual (Estrutura Narrativa)**  
  - Nível 5: Desenvolve todas as partes da narrativa (orientação, complicação e desfecho) com personagens, narrador, organização temporal e espaço.  
  - Nível 4: Apresenta todas as partes, mas desenvolve parcialmente uma delas e/ou apresenta 3 elementos da narrativa.  
  - Nível 3: Desenvolve duas partes ou apresenta todas, mas sem detalhamento; dois elementos da narrativa.  
  - Nível 2: Desenvolve uma parte ou apresenta um elemento da narrativa.  
  - Nível 1: Descreve elementos isolados ou apresenta traços de outros tipos textuais.  

- **C4 - Coesão (Encadeamento de Ideias)**  
  - Nível 5: Repertório coesivo diversificado, com raras inadequações que não prejudicam a compreensão.  
  - Nível 4: Repertório coesivo diversificado com desvios pontuais que afetam parcialmente a inteligibilidade.  
  - Nível 3: Repertório pouco diversificado com um tipo de desvio recorrente.  
  - Nível 2: Repertório coesivo escasso, com desvios pontuais.  
  - Nível 1: Palavras e períodos justapostos e desconexos ou repertório coesivo escasso com desvios recorrentes.  

### Instruções importantes:

1. **Ignore completamente os tokens técnicos** presentes na redação. Eles **não fazem parte do conteúdo real** e foram adicionados por quem digitalizou o texto.  
   Os tokens a ignorar incluem:
   - `[P]`, `[p]` → nova linha/parágrafo
   - `[S]`, `[s]` → símbolo gráfico
   - `[T]` → título
   - `[R]`, `[X]` → rasura
   - `[?]`→ símbolo desconhecido
   - `[LC]`, `[LT]`, `[lt]` → escrita fora da linha
2. **Avalie somente o conteúdo real** da redação. Ignore espaçamentos, quebras de linha ou formatações erradas.
3. **Compare o conteúdo da redação com o texto motivador**.
4. **Não gere explicações, apenas as notas.**

**Redação a ser avaliada**:  
{essay_text}  

**Texto motivador**:  
{prompt_text}
"""

def create_essay_evaluation_chain(chat_model: BaseChatModel):
    if isinstance(chat_model, ChatOpenAI):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Você é um avaliador experiente de redações narrativas, com base nos critérios de correção utilizados em avaliações oficiais do ensino fundamental"),
            ("user", PROMPT_INSTRUCTION)
        ])
    elif isinstance(chat_model, (ChatVertexAI, GemmaVertexAIModelGarden, GemmaLocalHF, GemmaChatLocalHF)):
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
    text = re.sub(r"<[^>]+>", "", text) 
    text = re.sub(r"\s+", " ", text)  
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

class GemmaEssayEvaluator(EssayEvaluator):
    def __init__(self):
        super().__init__(cfg.gemma_vertexai)

class GemmaHFEssayEvaluator(EssayEvaluator):
    def __init__(self):
        super().__init__(cfg.gemma_hf)

class GemmaChatHFEssayEvaluator(EssayEvaluator):
    def __init__(self):
        super().__init__(cfg.gemma_chat_hf)

class AnthropicEssayEvaluator(EssayEvaluator):
    def __init__(self):
        super().__init__(cfg.chat_anthropic)
