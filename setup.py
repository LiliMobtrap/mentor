import re
from openai import OpenAI
import os
import tiktoken
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from doc_to_string import string

pinecone_api_key = "89333e65-16ca-4198-b9a1-f01b5065c042"
os.environ['OPENAI_API_KEY'] = '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'
client = OpenAI()


def split_text_by_tokens(input_string, embedding_tokens=500):
    encoding = tiktoken.get_encoding("o200k_base")
    tokens = encoding.encode(input_string)

    embedding_chunks = [tokens[i:i + embedding_tokens] for i in range(0, len(tokens), embedding_tokens)]
    embedding_texts = [encoding.decode(chunk) for chunk in embedding_chunks]

    return embedding_texts


def embed_text(text_chunk):
    model = "text-embedding-3-small"
    res = client.embeddings.create(input=text_chunk, model=model)
    embed = res.data[0].embedding
    return embed


def get_vectordb(input_string, gf_name):
    gf_name = re.sub(r'\W+', '', gf_name.lower().replace(' ', '-'))
    index_name = gf_name + "mybeloved"
    pc = Pinecone(api_key=pinecone_api_key)

    if index_name not in pc.list_indexes().names():
        values = []
        text_chunks = split_text_by_tokens(input_string)
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
            deletion_protection="disabled"
        )
        for i, text in enumerate(text_chunks, start=1):
            values.append(
                (
                    f"vector_{i}",
                    embed_text(text),
                    {"text": text}
                )
            )
        index = pc.Index(index_name)
        index.upsert(vectors=values)

    else:
        index = pc.Index(index_name)

    return index, index_name  # !!!


def setup(gf_name, input_string=string):
    index, index_name = get_vectordb(input_string, gf_name)  # !!!
    return index, index_name




