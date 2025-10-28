<context>
    Você é um engenheiro de IA desenvolvendo um sistema de agentes inteligentes.
    Seu objetivo é criar agentes que possam interagir com APIs externas para realizar tarefas complexas.
    Você deve focar em como definir funções, estruturar chamadas de API e processar respostas para maximizar a eficiência e a precisão dos agentes.
    Utilize https://gnews.io/ como API de noticias
    Utilize https://openweathermap.org/api como API de clima
    Utilize https://api.exchangerate.host/convert como API de cotação do dólar
    Utilize https://api.adviceslip.com/advice como API de mensagem motivacional
</context>

<tasks>
    Criar um agente que produza um briefing do momento atual.
    O agente deve chamar uma função que busca notícias recentes, informações meteorológicas e atualizações da cotação do dólar.
    A messagem produzida pelo agente deve ser finalizada com uma mensagem motivacional.
    A linguagem utilizada na mensagem deve ser casual e objetiva.
    O agente deve inferir automaticamente a localização do usuário para buscar as informações utilizando Tools.
    Escreva o agente em Python, utilizando a biblioteca LangChain.
</tasks>

<example>
    "Olá, me atualize": Bom dia! No momento em Recife faz 30 graus Celsius. Os presidentes do Brasil e EUA se encontraram na Malásia para discutir tarifas. A cotação atual do dólar é de 6 reais. [frase motivacional]
</example>
