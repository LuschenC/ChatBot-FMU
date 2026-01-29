import json
import textwrap
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import requests
import urllib.parse

#CARREGA ARQUIVOS
# CSV com frases e categorias
dados = pd.read_csv("content/perguntas.csv")
frases = dados["frase"].astype(str).tolist()
categorias = dados["categoria"].astype(str).tolist()

# Treino do modelo
vetorizador = CountVectorizer()
X = vetorizador.fit_transform(frases)
modelo = MultinomialNB()
modelo.fit(X, categorias)

# JSON com respostas
with open("content/respostas.json", "r", encoding="utf-8") as f:
    respostas = json.load(f)

# CONFIGURAÇÕES ADICIONAIS
# Lista de categorias que NÃO devem usar a API do Google (assuntos gerais)
# Se a pergunta cair em uma dessas, o bot dirá que não pode responder.
CATEGORIAS_BLOQUEADAS = [
    "cachorro",
    "animais",
    "xingamentos",
    "festa",
    "politica",
    "viagem",
    "futebol",
    "musica", # Exemplo adicional
    "esportes", # Exemplo adicional
]

# Limiar de confiança para a previsão da categoria
LIMIAR = 0.20

#     CONFIGURAÇÕES API GOOGLE

API_KEY = "AIzaSyBNibFWwDXjFfp93g72DiaiJS7x8BA8TrQ"
SEARCH_ENGINE_ID = "12ff780e639874667"

def pesquisar_google(texto):
    """Consulta a API do Google e retorna títulos e links."""
    try:
        query = urllib.parse.quote(texto)
        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}"
        )
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return ["Erro ao pesquisar na API."]

    resultados = []
    # Limita a 5 resultados
    for item in response.json().get("items", [])[:5]:
        titulo = item.get("title", "Sem título")
        link = item.get("link", "Sem link")
        resultados.append(f"{titulo} — {link}")
    return resultados or ["Nenhum resultado encontrado na API."]

#     FUNÇÃO PARA RESPONDER
def responder_categoria(categoria, entrada, confianca):
    resposta_final = {
        "resposta": "",
        "links": []
    }

    # 1. Categoria bloqueada
    if categoria in CATEGORIAS_BLOQUEADAS:
        resposta_final["resposta"] = (
            f"Sua pergunta sobre '{categoria}' não faz parte do meu escopo de conhecimento. "
            "Por favor, pergunte sobre um assunto relacionado ao meu treinamento."
        )
        return resposta_final

    # 2. Tenta responder com JSON
    if categoria in respostas:
        resp = respostas[categoria]
        texto = (resp.get("texto") or "").strip()
        links = resp.get("links") or []

        if texto or links:
            resposta_final["resposta"] = texto
            resposta_final["links"] = links
            return resposta_final   # <-- agora retorna corretamente

    # 3. Não encontrou -> usa API
    resultados_api = pesquisar_google(entrada)

    resposta_final["resposta"] = (
        "Não encontrei uma resposta cadastrada. Fiz uma pesquisa para você:"
    )
    resposta_final["links"] = resultados_api
    return resposta_final

def responder_pergunta(entrada):
    # Previsão de categoria
    probs = modelo.predict_proba(vetorizador.transform([entrada]))[0]
    idx = np.argmax(probs)
    categoria_prevista = modelo.classes_[idx]
    confianca = probs[idx]

    # Baixa confiança
    if confianca < LIMIAR:
        return {
            "resposta": (
                "Não consegui identificar a categoria da sua pergunta. "
                "Por favor, tente reformular."
            ),
        }

    # Chama responder_categoria
    return responder_categoria(categoria_prevista, entrada, confianca)

    # Responde ou pesquisa (incluindo as novas regras de bloqueio/API)
    responder_categoria(categoria_prevista, entrada)
# Iniciador do Flask
app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template('index.html')

# Rota para receber entradas JSON
@app.route("/chat", methods=["POST"])
def chat():
    dados = request.get_json()
    pergunta = dados.get("pergunta", "")

    if pergunta == "":
        return jsonify({
            "resposta": "Por favor, digite sua pergunta para que eu possa responder.",
            "links": []
        })

    resposta = responder_pergunta(pergunta)
    return jsonify(resposta)

if __name__ == "__main__":
    app.run(debug=True)