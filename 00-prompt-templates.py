from langchain_aws import BedrockLLM
from langchain.prompts import PromptTemplate

# Initialize the Bedrock LLM
llm = BedrockLLM(model_id="anthropic.claude-v2")

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short paragraph about {topic}.",
)

# Use the prompt template
topic = "artificial intelligence"
formatted_prompt = prompt.format(topic=topic)

# Generate a response
response = llm.invoke(formatted_prompt)

print(f"Prompt: {formatted_prompt}")
print(f"Response: {response}")