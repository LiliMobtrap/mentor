from chat import chat
from pinecone.grpc import PineconeGRPC as Pinecone
from setup import setup

pc = Pinecone(api_key="insert_key")

gf_name = "Bronze Age Pervert"

index, index_name = setup(gf_name)
print(index_name)
prompt = "How can I become Peter Thiel's twink?"
# index_name = "bronzeagepervertmybeloved"
# index = pc.Index(index_name)
response = chat(prompt=prompt, index=index, index_name=index_name, gf_name=gf_name)

print(response)
