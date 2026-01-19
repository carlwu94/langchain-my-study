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

# print (agent)
# langgraph.graph.state.CompiledStateGraph
# print (agent.nodes)
# {
#   '__start__': <langgraph.pregel._read.PregelNode object at 0x0000026F12A9A6D0>, 
#   'model': <langgraph.pregel._read.PregelNode object at 0x0000026F12A9AA10>
# }


# results = agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "What is a apple's colour?"
#                 # "content": "What is the weather in Suzhou?"
#             }
#         ]
#     }
# )

# messages = results["messages"]
# print (f"history messages: {len(messages)}")
# for m in messages:
#     m.pretty_print()

# for event in agent.stream(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "What is a apple's colour?"
#                 # "content": "What is the weather in Suzhou?"
#             }
#         ]
#     },
#     stream_mode="values"  # 消息单位回复
# ):
#     messages = event["messages"]
#     print (f"💡 [HISTORY MESSAGES]: {len(messages)}")
#     messages[-1].pretty_print()

for chunk in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is a apple's colour?"
                # "content": "What is the weather in Suzhou?"
            }
        ]
    },
    stream_mode="messages"  # token by token
):
    print (chunk[0].content, end='')