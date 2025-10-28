import os
from pprint import pprint
import requests
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.messages import HumanMessage, ToolMessage

load_dotenv()


def get_location() -> Dict[str, Any]:
    """Infer user location by IP using ipinfo.io (public, no key required for basic info).

    Returns: dict with city, region, country, loc (lat,lon)
    """
    try:
        r = requests.get("https://ipinfo.io/json", timeout=5)
        r.raise_for_status()
        data = r.json()
        loc = data.get("loc", "")
        lat, lon = (loc.split(",") if loc else (None, None))
        return {
            "city": data.get("city"),
            "region": data.get("region"),
            "country": data.get("country"),
            "loc": {"lat": lat, "lon": lon},
        }
    except Exception as e:
        return {"city": None, "region": None, "country": None, "loc": {"lat": None, "lon": None}, "error": str(e)}


def fetch_news(gnews_key: Optional[str], country: Optional[str] = None, max_results: int = 3) -> List[Dict[str, Any]]:
    """Fetch top headlines from gnews.io. Expects API key in `gnews_key`.

    Returns list of {title, description, url}
    """
    if not gnews_key:
        return [{"title": "API key for GNews not provided", "description": "", "url": ""}]
    params = {
        "token": gnews_key,
        "lang": "pt",
        "max": max_results,
    }
    if country:
        params["country"] = country.lower()
    try:
        r = requests.get("https://gnews.io/api/v4/top-headlines", params=params, timeout=7)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
        results = []
        for a in articles[:max_results]:
            results.append({
                "title": a.get("title"),
                "description": a.get("description"),
                "url": a.get("url"),
            })
        return results
    except Exception as e:
        return [{"title": "Erro ao buscar notícias", "description": str(e), "url": ""}]


def fetch_weather(owm_key: Optional[str], lat: Optional[str], lon: Optional[str]) -> Dict[str, Any]:
    """Fetch current weather from OpenWeatherMap using lat/lon. Returns summary dict."""
    if not owm_key or not lat or not lon:
        return {"error": "missing openweathermap key or coordinates"}
    try:
        params = {"lat": lat, "lon": lon, "units": "metric", "appid": owm_key, "lang": "pt"}
        r = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params, timeout=7)
        r.raise_for_status()
        data = r.json()
        main = data.get("main", {})
        weather = data.get("weather", [{}])[0]
        return {
            "temp": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "description": weather.get("description"),
            "raw": data,
        }
    except Exception as e:
        return {"error": str(e)}


COUNTRY_TO_CURRENCY = {
    "BR": "BRL",
    "US": "USD",
    "PT": "EUR",
    "DE": "EUR",
    "FR": "EUR",
    "IN": "INR",
    "GB": "GBP",
    "CA": "CAD",
    # add more as needed
}


def fetch_usd_rate(target_currency: Optional[str]) -> Dict[str, Any]:
    """Fetch USD -> target_currency conversion using exchangerate.host"""
    if not target_currency:
        return {"error": "target currency missing"}
    try:
        r = requests.get("https://api.exchangerate.host/convert", params={"from": "USD", "to": target_currency}, timeout=5)
        r.raise_for_status()
        data = r.json()
        return {"rate": data.get("result"), "info": data}
    except Exception as e:
        return {"error": str(e)}


def fetch_advice() -> str:
    try:
        r = requests.get("https://api.adviceslip.com/advice", timeout=4)
        r.raise_for_status()
        data = r.json()
        return data.get("slip", {}).get("advice", "")
    except Exception:
        return "Mantenha o foco e siga em frente."

@tool
def _tool_location() -> str:
    """Infer user location by IP using ipinfo.io (public, no key required for basic info).

    Returns: dict with city, region, country, loc (lat,lon)
    """
    loc = get_location()
    city = loc.get("city")
    region = loc.get("region")
    country = loc.get("country")
    return f"{city or ''}, {region or ''}, {country or ''} | loc={loc.get('loc')}"

@tool
def _tool_news(input_text: str) -> str:
    """Fetch top headlines from gnews.io. Expects API key in `gnews_key`.

    Returns list of {title, description, url}
    """
    key = os.getenv("GNEWS_API_KEY")
    # input may contain a country code or city; try to use country if provided
    country = None
    if input_text and len(input_text) == 2:
        country = input_text
    articles = fetch_news(key, country=country)
    out = []
    for a in articles:
        out.append(f"- {a.get('title')}: {a.get('description')}")
    return "\n".join(out)

@tool
def _tool_weather(input_text: str) -> str:
    """Fetch current weather from OpenWeatherMap using lat/lon. Returns summary dict."""
    key = os.getenv("OPENWEATHER_API_KEY")
    # input_text expected to be 'lat,lon'
    if not input_text:
        return "Coordinates missing"
    lat, lon = [p.strip() for p in input_text.split(",")]
    w = fetch_weather(key, lat, lon)
    if "error" in w:
        return f"Erro: {w['error']}"
    return f"{w.get('temp')}°C, {w.get('description')} (sensação {w.get('feels_like')}°C)"

@tool
def _tool_exchange(input_text: str) -> str:
    """Fetch USD -> target_currency conversion using exchangerate.host"""
    # input_text expected to be currency code like BRL
    target = input_text.strip().upper() if input_text else "BRL"
    r = fetch_usd_rate(target)
    if "error" in r:
        return f"Erro: {r['error']}"
    return f"1 USD = {r.get('rate')} {target}"

@tool
def _tool_advice() -> str:
    """Fetch a motivational advice phrase."""
    return fetch_advice()


def create_agent() -> ChatOpenAI:
    """
    Create a lightweight agent object compatible with LangChain v1.0.1 usage in this script.

    Instead of using the higher-level `initialize_agent` helper (which may vary
    between LangChain versions), we return the LLM instance and a mapping of tool
    callables. This keeps the code explicit and compatible with v1+.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    with_tools = llm.bind_tools([_tool_location, _tool_advice, _tool_exchange, _tool_news, _tool_weather])
    prompt = (
        "Você é um assistente que produz um briefing curto e objetivo em português (PT-BR). "
        "Use as ferramentas disponíveis para: 1) inferir a localização do usuário; "
        "2) obter notícias recentes relevantes para o país/município; 3) obter o clima atual a partir das coordenadas; "
        "4) obter a cotação atual do dólar na moeda local; 5) finalizar com uma frase motivacional. "
        "Responda de forma casual, direta e em poucas frases."
    )
    # result = with_tools.invoke([HumanMessage(content=prompt)])
    # print('setup prompt result:')
    # print(result.to_json())
    return with_tools


def build_briefing() -> dict[str, Any]:
    """Runs the agent to produce a briefing in Portuguese, casual and objective.

    It will: infer location, fetch news for the country, weather for coordinates, USD rate to local currency, and an advice.
    """

    # Build a prompt that instructs the AgentExecutor to use the provided tools.
    # The AgentExecutor will call the tools (location, news, weather, exchange, advice)
    # and produce the final briefing in Portuguese.
    system_prompt = (
        "Você é um assistente que produz um briefing curto e objetivo em português (PT-BR). "
        "Use as ferramentas disponíveis para: 1) inferir a minha localização; "
        "2) obter notícias recentes relevantes para o país/município que estou; 3) obter o clima atual a partir das coordenadas; "
        "4) obter a cotação atual do dólar na moeda local; 5) finalizar com uma frase motivacional. "
        "Responda de forma casual, direta e em poucas frases."
    )
    agent = create_agent()
    user_prompt = "olá"
    full_prompt = system_prompt + "\n" + user_prompt
    result = agent.invoke([HumanMessage(content=full_prompt)])
    if hasattr(result, "tool_calls") and result.tool_calls:
        for call in result.tool_calls:
            tool_name = call['name']
            args = call['args']
            print(f"[Tool called: {tool_name}({args})]")
            if tool_name == "_tool_location":

                location_result = _tool_location.invoke(args)
                print(f"[Tool output: {location_result}]")
                location_message = ToolMessage(
                    name=tool_name,
                    content=str(location_result),
                    tool_call_id=call["id"]
                )
                print(f"[Tool result: {location_message}]")
                followup = agent.invoke([
                    HumanMessage(content=full_prompt),
                    result,
                    location_message
                ])
                print(followup.content)
                result = followup

    return result.to_json()


if __name__ == "__main__":
    # simple CLI runner
    print("Gerando briefing...\n")
    try:
        text = build_briefing()
        pprint(text)
    except Exception as e:
        print("Erro ao gerar briefing:", e)
