import boto3
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from botocore.config import Config

#creating config for all required parameters to be passed to boto3.

retry_config = Config(
        region_name = 'us-east-1',
        retries = {
            'max_attempts': 10,
            'mode': 'standard'
        }
)

# Creating boto3 session by passing profile information. Profile can be parametrized depeding upon the env you are using
session = boto3.session.Session(profile_name='default')

"""" 
btot3 provides two different client to ivoke bedrock operation.
1. bedrock : creating and managing Bedrock models.
2. bedrock-runtime : Running inference using Bedrock models.
"""
boto3_bedrock = session.client("bedrock", config=retry_config)
boto3_bedrock_runtime = session.client("bedrock-runtime", config=retry_config)


"""
Here we try to see the details of foundation models available.
Using bedrcok client you can do various model operation.
"""
print(boto3_bedrock.list_foundation_models()['modelSummaries'][0])

""" 
Here we will invoke anthropic claude model to answer a question using prompt template 
This is a inference call. We need model to generate answer for the question we provide.
Hence using bedrock-runtime module
"""
llm = Bedrock(
        model_id="anthropic.claude-instant-v1",
        client=boto3_bedrock_runtime,
        model_kwargs={
            "temperature": 0,
            "max_tokens_to_sample": 2048,
            "top_p": 1,
            "top_k": 250,
            "stop_sequences": ["\n\nHuman:"],
        },
    )

#   Define prompt template
diet_template = '''I want you to act as a acting dietician for people.
In an easy way, explain the benefits of {fruit}.'''

prompt_template_1 = PromptTemplate(input_variables = ['fruit'], template=diet_template)
llm_chain = LLMChain(llm=llm, prompt=prompt_template_1)
print(llm_chain.run({'fruit':'apple'}))


# Define another prompt template
exercise_template = '''
I want you to act as a acting gym trainer for people.
In an easy way, explain the benefits of consuming {dish} and also consider that they go to gym everyday at {gym_time}.
Also advice on the best times to go to gym with timings for everyday of the week. 
'''

prompt_template_2 = PromptTemplate(input_variables = ['dish', 'gym_time'], template=exercise_template)
second_llm_chain = LLMChain(llm=llm, prompt=prompt_template_2)
print(second_llm_chain.run({'dish': 'milk', 'gym_time': '04:00am'}))