# from langchain_ollama import ChatOllama

# model = ChatOllama(
#     model="deepseek-r1:1.5b",
#     base_url="http://localhost:11434",
#     temperature=0.1
# )

## langchain 1.0
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="ollama:deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.1,
    max_tokens=2000
)

for chunk in model.stream("给我唱首歌"):
    print (chunk.content, end='', flush=True)