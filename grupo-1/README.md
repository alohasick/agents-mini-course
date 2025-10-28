# Agente de Briefing (grupo-1)

Este é um agente em Python que monta um briefing rápido do momento atual. Ele:

- Infere a localização do usuário por IP (ipinfo.io).
- Busca notícias recentes com a API do GNews.
- Busca o clima atual com OpenWeatherMap.
- Busca a cotação USD -> moeda local usando exchangerate.host.
- Fecha a mensagem com uma frase motivacional via adviceslip.com.

A saída é em português, casual e objetiva.

## Arquivos

- `agent_briefing.py`: script principal que implementa as ferramentas e monta o briefing.
- `.env.example`: variáveis de ambiente necessárias.

## Variáveis de ambiente

Copie `.env.example` para `.env` e preencha as chaves necessárias:

- `OPENAI_API_KEY` - chave OpenAI (usada pelo LangChain `ChatOpenAI`).
- `GNEWS_API_KEY` - chave para https://gnews.io/ (obrigatório para notícias).
- `OPENWEATHER_API_KEY` - chave para https://openweathermap.org/api (obrigatório para clima).

Algumas APIs usadas não precisam de chave:

- `https://api.exchangerate.host/convert` — sem chave.
- `https://api.adviceslip.com/advice` — sem chave.
- `https://ipinfo.io/json` — para inferir localização (uso básico sem chave).

## Instalação

Recomendo usar um virtualenv. Exemplo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Preencha `.env` com suas chaves.

## Uso

Executar o script:

```bash
python grupo-1/agent_briefing.py
```

O script imprime um briefing em português, terminando com uma frase motivacional.

## Observações e limites

- O script tenta inferir a moeda local a partir do código do país via um mapeamento simples. Caso o país não esteja no mapeamento, usa `BRL` como fallback.
- A qualidade do briefing depende das chaves (GNews/OpenWeather) e conectividade.
- Para produção, recomendo melhorar o mapeamento país->moeda com uma dependência como `pycountry` ou `forex-python` e lidar com caches/erros de rede.
