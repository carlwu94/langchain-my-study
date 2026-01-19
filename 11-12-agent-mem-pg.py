from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

def get_weather(city: str) -> str:
    """
    Get weather for a given city
    """
    return f"Its always sunny in {city}!"

local_model = init_chat_model(
        # model="ollama:deepseek-r1:1.5b",
        model = "ollama:llama3.2:3b",
        base_url = "http://localhost:11434",
        temperature = 0.1,
        max_tokens = 2000
    )

# checkpointer = InMemorySaver
db_url = "postgresql://postgres:123456@localhost:5432/postgres?sslmode=disable"

with PostgresSaver.from_conn_string(db_url) as checkpointer:
    # checkpointer.setup()

    agent = create_agent(
        model="deepseek:deepseek-chat",
        # model = local_model,
        tools = [get_weather],
        checkpointer = checkpointer
    )

    config = {"configurable": {"thread_id": "1"}}

    #### Round 1 ####
    results = agent.invoke(
        {"messages": [{"role": "user", "content": "Give me a shit"}]},
        config = config
    )
    messages = results["messages"]
    print (f"💡 [HISTORY MESSAGES]: {len(messages)}")
    for m in messages:
        m.pretty_print()

    #### Round 2 ####
    results = agent.invoke(
        {"messages": [{"role": "user", "content": "More!"}]},
        config = config
    )
    messages = results["messages"]
    print (f"💡 [HISTORY MESSAGES]: {len(messages)}")
    for m in messages:
        m.pretty_print()