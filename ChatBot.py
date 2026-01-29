import pandas as pd
import json
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# 1) Carregar dados de treino
dados = pd.read_csv("content/perguntas.csv")

# 2) Selecionar as colunas e converter em listas
frases = dados["frase"].astype(str).tolist()
categorias = dados["categoria"].astype(str).tolist()

# 3) Vetorizar e treinar modelo
vetorizador = CountVectorizer()
X = vetorizador.fit_transform(frases)

modelo = MultinomialNB()
modelo.fit(X, categorias)

# 4) Carregar respostas.json
with open("content/respostas.json", encoding="utf-8") as f:
    respostas = json.load(f)

# 5 Classificação da frase digitada
def responder_pergunta(entrada):
    categoria_prevista = modelo.predict(vetorizador.transform([entrada]))[0]

    # 6 Buscar a resposta correspondente no JSON
    resp = respostas.get(categoria_prevista, {"texto": "", "links": []})
    texto = (resp.get("texto") or "").strip()
    links = resp.get("links") or []

    if not texto and not links:
        return {
            "resposta": "Ainda não tenho uma resposta cadastrada para essa dúvida.",
            "links": []
        }

    return {
        "resposta": texto,
        "links": links
    }

# Iniciador do Flask

app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template("index.html")

# Rota para receber entradas JSON
@app.route("/chat", methods=["POST"])
def chat():
    dados = request.get_json()
    pergunta = dados.get("pergunta", "")

    if pergunta == "":
        return jsonify({
            "resposta": "Porfavor, digite sua pergunta para que possa responder.",
            "links": []
        })

    resposta = responder_pergunta(pergunta)
    return jsonify(resposta)

if __name__ == "__main__":
    app.run(debug=True)