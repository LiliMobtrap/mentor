import os
import json
from openai import OpenAI
from setup import embed_text

# Defina a chave da API do OpenAI
os.environ['OPENAI_API_KEY'] = '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'
client = OpenAI()

# Caminho para a pasta onde os arquivos de histórico serão armazenados
RECORD_DIR = "record"


def get_history_file(index_name):
    """Retorna o caminho do arquivo de histórico baseado no nome do índice."""
    return os.path.join(RECORD_DIR, f"{index_name}_history.jsonl")


def load_history(index_name):
    """Carrega o histórico a partir de um arquivo JSONL."""
    historico = []
    history_file = get_history_file(index_name)
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            for line in f:
                historico.append(json.loads(line))
    return historico


def save_interactions(index_name, role, content):
    """Salva uma interação (usuário ou assistente) no arquivo JSONL."""
    history_file = get_history_file(index_name)
    with open(history_file, 'a') as f:
        json.dump({"role": role, "content": content}, f)
        f.write('\n')  # JSONL usa uma linha por registro


def reset_history(index_name):
    """Limpa o histórico (tanto na memória quanto no arquivo)."""
    history_file = get_history_file(index_name)
    open(history_file, 'w').close()


def chat(prompt, index, index_name, gf_name):
    _historico = load_history(index_name)  # Carrega o histórico do arquivo

    # Adiciona a nova interação do usuário ao histórico e salva no arquivo
    _historico.append({"role": "user", "content": prompt})
    save_interactions(index_name, "user", prompt)

    # Resuma o histórico completo até o momento, incluindo o prompt atual
    last_interactions = _historico[-10:]  # Pega até as últimas 10 interações
    chat_history = "\n".join([f"{entry['role']}: {entry['content']}" for entry in last_interactions])
    query = summarize_chat(chat_history)

    # Faz a query no índice usando o resumo para obter o contexto relevante
    query_response = index.query(
        vector=embed_text(query),
        top_k=3,
        include_metadata=True
    )
    context = "\n".join([match['metadata']['text'] for match in query_response['matches']])

    # Adiciona o contexto recuperado ao histórico como uma interação do sistema
    _historico.append({"role": "system", "content": f"Contexto recuperado:\n{context}"})
    # save_interactions(index_name, "system", f"Contexto recuperado:\n{context}")

    # Gera a resposta usando o histórico completo (resumo + contexto + prompt)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are {gf_name} and your task is to answer user's questions using the expressions, the style and the informations of the given context. Don't make up anything, just answer the user's message. Focus on the context and attempt to reflect the author's real opinion as it's your own. Avoid being verbose and use only the informations that answer the user's message. You can just discard useless information."}
        ] + _historico,  # Concatena o histórico completo com o prompt
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "text"}
    )

    # Adiciona a resposta gerada ao histórico e salva no arquivo
    response_content = response.choices[0].message.content
    _historico.append({"role": "assistant", "content": response_content})
    save_interactions(index_name, "assistant", response_content)

    return response_content


def summarize_chat(history):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You job is to read the conversation given and build a contextualized query for a vector database based on the last user's message. Return only the string that will be used as a query mantaining the words and expressions used by the user."},
            {"role": "user", "content": f"{history}"}
        ],
        temperature=0.5,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "text"}
    )
    print(f"Query: {response.choices[0].message.content}")

    return response.choices[0].message.content
