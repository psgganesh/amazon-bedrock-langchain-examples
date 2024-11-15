
### How to use

1. First, create a virtual environment and activate it:

```
python -m venv langchain_env
source langchain_env/bin/activate  # On Windows, use: langchain_env\Scripts\activate
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Set up your AWS credentials and region in your environment variables or AWS configuration file.

4. Run each Python file individually to see the demonstrations:


#### 00-prompt-templates.py
This file demonstrates the use of prompt templates with Amazon Bedrock's Claude v2 model.

#### 01-chains.py
This file showcases the implementation of LLM chains using Amazon Bedrock's Claude v2 model.

#### 02-chat-models.py
This file illustrates how to use chat models with Amazon Bedrock's Claude v2 model, including both single responses and streaming.

#### 03-text-embeddings.py
This file demonstrates text embeddings using Amazon Titan embeddings model.

#### 04-document-loader.py
This file shows how to use document loaders for PDFs and text files, along with text splitting.

#### 05-retrievers.py
This file implements a custom retriever using FAISS and Amazon Titan embeddings.