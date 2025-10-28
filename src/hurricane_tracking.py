import requests
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from datetime import datetime
import openai
import os

# --- Configuração ---
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


# --- Estado do grafo ---
class HurricaneState(TypedDict):
    question: str
    storms_data: List[str]
    synthesis: str

# --- Tool: NOAA Hurricane Data ---
def get_active_storms():
    url = "https://www.nhc.noaa.gov/CurrentStorms.json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    active_storms = []

    for storm in data.get("activeStorms", []):
        s = {
            "id": storm.get("id"),
            "nome": storm.get("name"),
            "classificacao": {
                "codigo": storm.get("classification"),
                "descricao": (
                    "Furacão" if storm.get("classification") == "HU" else
                    "Tempestade Tropical" if storm.get("classification") == "TS" else
                    "Depressão Tropical" if storm.get("classification") == "TD" else
                    "Distúrbio Tropical" if storm.get("classification") == "SD" else
                    "Outro"
                )
            },
            "intensidade_ventos_mph": storm.get("intensity"),
            "pressao_mb": storm.get("pressure"),
            "localizacao": {
                "latitude": storm.get("latitudeNumeric"),
                "longitude": storm.get("longitudeNumeric"),
                "latitude_texto": storm.get("latitude"),
                "longitude_texto": storm.get("longitude")
            },
            "movimento": {
                "direcao_graus": storm.get("movementDir"),
                "velocidade_nos": storm.get("movementSpeed")
            },
            "ultima_atualizacao": storm.get("lastUpdate"),
            "fontes": {
                "aviso_publico": storm.get("publicAdvisory", {}).get("url"),
                "grafico_previsao": storm.get("forecastGraphics", {}).get("url"),
                "tracado_previsao": storm.get("forecastTrack", {}).get("kmzFile"),
                "cone_previsao": storm.get("trackCone", {}).get("kmzFile"),
                "discussao": storm.get("forecastDiscussion", {}).get("url")
            }
        }
        active_storms.append(s)

    return active_storms or ["Nenhum furacão ativo detectado no momento."]



# --- Etapa 1: Busca ---
def fetch_hurricane_data(state: HurricaneState):
    print("🌀 Buscando dados de furacões ativos no Caribe...")
    storms = get_active_storms()
    return {"storms_data": storms}

# --- Etapa 2: Síntese ---
def summarize_storms(state: HurricaneState):
    print("🧠 Gerando resumo meteorológico...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

    if not state["storms_data"]:
        return {"synthesis": "Nenhum sistema tropical ativo detectado no momento no Caribe."}

    # Cria contexto legível
    context = ""
    for s in state["storms_data"]:
        context += f"""
        🌀 {s['nome']} ({s['classificacao']['descricao']})
        - Intensidade: {s['intensidade_ventos_mph']} mph
        - Pressão: {s['pressao_mb']} mb
        - Localização: {s['localizacao']['latitude_texto']}, {s['localizacao']['longitude_texto']}
        - Movimento: {s['movimento']['direcao_graus']}° a {s['movimento']['velocidade_nos']} nós
        - Última atualização: {s['ultima_atualizacao']}
        - Fontes: {s['fontes']['aviso_publico']}
        """

    prompt = f"""
    Você é um meteorologista experiente especializado em sistemas tropicais.
    Abaixo estão os dados mais recentes da NOAA sobre furacões ativos:

    {context}

    Gere um resumo técnico e claro sobre o estado atual dos furacões e tempestades tropicais
    no Atlântico e Caribe, incluindo:
    - Nome e tipo do sistema (ex: Furacão Categoria 3)
    - Localização geográfica
    - Direção e velocidade
    - Intensidade e pressão
    - Possíveis regiões em risco
    - Links úteis (se disponíveis)
    """

    result = llm.invoke(prompt)
    return {"synthesis": result.content}


# --- Construção do grafo ---
workflow = StateGraph(HurricaneState)
workflow.add_node("fetch", fetch_hurricane_data)
workflow.add_node("summarize", summarize_storms)
workflow.set_entry_point("fetch")
workflow.add_edge("fetch", "summarize")
workflow.add_edge("summarize", END)

graph = workflow.compile()

# --- Execução ---
if __name__ == "__main__":
    question = "Quais furacões estão ativos no Caribe neste momento?"
    result = graph.invoke({"question": question})
    print("\n🌪️ RELATÓRIO ATUAL:")
    print(result["synthesis"])
