import pandas as pd
import json
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

# 5) Loop do chat
print("RF/SEBRAE Bot: Olá! Como posso ajudar? (digite 'sair' para terminar)")
while True:
    entrada = input("Você: ").strip()

    if entrada.lower() == "sair":
        print("RF/SEBRAE Bot: Até logo!")
        break

    if not entrada:
        print("RF/SEBRAE Bot: Pode digitar sua dúvida :)")
        continue

    # Classificação da frase digitada
    categoria_prevista = modelo.predict(vetorizador.transform([entrada]))[0]

    # Buscar a resposta correspondente no JSON
    resp = respostas.get(categoria_prevista, {"texto": "", "links": []})
    texto = (resp.get("texto") or "").strip()
    links = resp.get("links") or []

    # Caso não tenha resposta
    if not texto and not links:
        print("RF/SEBRAE Bot: Desculpe, não encontrei uma resposta. Pode reformular?")
        continue

    # Exibir texto da resposta
    if texto:
        print(f"RF/SEBRAE Bot: {texto}")

    # Exibir links, se houver
    if links:
        print("Links oficiais:")
        for url in links:
            print(f"- {url}")