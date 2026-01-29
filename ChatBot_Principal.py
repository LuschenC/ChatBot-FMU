import json
import textwrap
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import requests
import urllib.parse

# ======================================================
# CARREGAR CSV E JSON (SEUS ARQUIVOS REAIS)
# ======================================================

dados = pd.read_csv("perguntas.csv")  
frases = dados["frase"].astype(str).tolist()
categorias = dados["categoria"].astype(str).tolist()

with open("respostas.json", "r", encoding="utf-8") as f:
    respostas = json.load(f)

# ======================================================
# TREINAR O MODELO
# ======================================================

vetorizador = CountVectorizer()
X = vetorizador.fit_transform(frases)

modelo = MultinomialNB()
modelo.fit(X, categorias)

# Limite mínimo de confiança
LIMIAR = 0.20

# ======================================================
# CONFIGURAÇÃO DA API GOOGLE
# ======================================================

API_KEY = "AIzaSyBNibFWwDXjFfp93g72DiaiJS7x8BA8TrQ"
SEARCH_ENGINE_ID = " 12ff780e639874667"

def pesquisar_google(texto):
    """Consulta a API do Google e retorna resultados."""
    try:
        query = urllib.parse.quote(texto)
        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}"
        )
        response = requests.get(url)
        response.raise_for_status()
    except:
        return ["Erro ao acessar a API Google."]

    resultados = []
    for item in response.json().get("items", [])[:5]:
        titulo = item.get("title", "Sem título")
        link = item.get("link", "Sem link")
        resultados.append(f"{titulo} — {link}")

    return resultados or ["Nenhum resultado encontrado."]

# ======================================================
# FUNÇÃO PRINCIPAL DO CHATBOT
# ======================================================

def responder_pergunta(entrada):

    # prever categoria
    entrada_transformada = vetorizador.transform([entrada])
    probs = modelo.predict_proba(entrada_transformada)[0]

    idx = np.argmax(probs)
    categoria_prevista = modelo.classes_[idx]
    confianca = probs[idx]

    # baixa confiança
    if confianca < LIMIAR:
        return {
            "resposta": "Não consegui identificar a categoria da sua pergunta. Tente reformular.",
            "links": []
        }

    # 1 → PRIMEIRO buscar no JSON (seu arquivo)
    categoria_normalizada = categoria_prevista.lower().strip()

    # Procurar a chave no JSON independentemente de maiúsculas/minúsculas
    for chave_json in respostas.keys():
        if chave_json.lower().strip() == categoria_normalizada:
            texto = (respostas[chave_json].get("texto") or "").strip()
            links = respostas[chave_json].get("links") or []

            return {
                "resposta": textwrap.fill(texto, width=80),
                "links": links
            }

    # 2 → SE NÃO existir resposta no JSON → Usa API Google
    resultados = pesquisar_google(entrada)
    return {
        "resposta": "Não encontrei resposta cadastrada. Pesquisei e encontrei estas informações:",
        "links": resultados
    }

# ======================================================
# FLASK
# ======================================================

app = Flask(__name__)

@app.route("/")
def home():
    return "<h2>Chatbot Receita Federal — API operando!</h2>"

@app.route("/chat", methods=["POST"])
def chat():
    dados = request.get_json()
    pergunta = dados.get("pergunta", "").strip()

    if pergunta == "":
        return jsonify({
            "resposta": "Por favor, digite uma pergunta.",
            "links": []
        })

    resposta = responder_pergunta(pergunta)
    return jsonify(resposta)

# ======================================================
# EXECUTAR SERVIDOR
# ======================================================

if __name__ == "__main__":
    app.run(debug=True)
