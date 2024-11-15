from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage

# Initialize the Bedrock Chat model
chat = ChatBedrock(model_id="anthropic.claude-v2")

# Single response
messages = [
    HumanMessage(content="Hello, how are you?"),
    AIMessage(content="I'm doing well, thank you for asking. How can I assist you today?"),
    HumanMessage(content="Can you explain what quantum computing is?")
]

response = chat.invoke(messages)
print("Single response:")
print(response.content)

# Streaming response
print("\nStreaming response:")
for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)
print()