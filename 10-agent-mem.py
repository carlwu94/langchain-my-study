from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

load_dotenv()

def get_weather(city: str) -> str:
    """
    Get weather for a given city
    """
    return f"Its always sunny in {city}!"

local_model = init_chat_model(
    # model="ollama:deepseek-r1:1.5b",
    model="ollama:llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0.1,
    max_tokens=2000
)

agent = create_agent(
    # model="deepseek:deepseek-chat",
    model = local_model,
    tools=[get_weather]
)

#### Round 1 ####
results = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "给我唱首歌"
            }
        ]
    }
)

messages = results["messages"]
print (f"💡 [HISTORY MESSAGES]: {len(messages)}")
for m in messages:
    m.pretty_print()

his_messages = messages

#### Round 2 ####
message_2 = {"role": "user","content": "再来"}
his_messages.append(message_2)
results = agent.invoke(
    {
        "messages": his_messages
    }
)

messages = results["messages"]
print (f"💡 [HISTORY MESSAGES]: {len(messages)}")
for m in messages:
    m.pretty_print()