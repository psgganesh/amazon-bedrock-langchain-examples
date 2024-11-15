# Example of langchain retrievers with amazon bedrock knowledgebases
import boto3
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain_community.chat_models import BedrockChat
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import List, Dict

class BedrockKnowledgeBaseDemo:
    def __init__(self, knowledge_base_id: str):
        self.knowledge_base_id = knowledge_base_id
        self.retriever = self._initialize_retriever()
        self.chat_model = self._initialize_chat_model()
        self.qa_chain = self._initialize_qa_chain()

    def _create_bedrock_client(self):
        """Initialize Bedrock client"""
        return boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'  # Change to your region
        )

    def _initialize_retriever(self):
        """Initialize the Knowledge Base retriever"""
        return AmazonKnowledgeBasesRetriever(
            knowledge_base_id=self.knowledge_base_id,
            retrieval_config={
                "vectorSearchConfiguration": {
                    "numberOfResults": 3,
                    "vectorSearchMethod": "SEMANTIC"
                }
            },
            client=boto3.client('bedrock-agent-runtime')
        )

    def _initialize_chat_model(self):
        """Initialize the Bedrock Chat Model"""
        return BedrockChat(
            client=self._create_bedrock_client(),
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            model_kwargs={
                "max_tokens": 1000,
                "temperature": 0.7,
            }
        )

    def _initialize_qa_chain(self):
        """Initialize the QA chain with custom prompt"""
        prompt_template = """
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know. 
        Don't try to make up an answer.

        Context: {context}

        Question: {question}

        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.chat_model,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

    def simple_retrieval(self, query: str) -> List[Dict]:
        """Perform simple retrieval from knowledge base"""
        documents = self.retriever.get_relevant_documents(query)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]

    def qa_with_sources(self, question: str) -> Dict:
        """Perform QA with source attribution"""
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }

    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """Perform similarity search with custom parameters"""
        self.retriever.retrieval_config["vectorSearchConfiguration"]["numberOfResults"] = k
        documents = self.retriever.get_relevant_documents(query)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]

class BedrockKnowledgeBaseManager:
    def __init__(self):
        self.client = boto3.client('bedrock-agent')

    def list_knowledge_bases(self):
        """List all available knowledge bases"""
        try:
            response = self.client.list_knowledge_bases()
            return response['knowledgeBaseSummaries']
        except Exception as e:
            print(f"Error listing knowledge bases: {str(e)}")
            return []

    def get_knowledge_base_details(self, knowledge_base_id: str):
        """Get details of a specific knowledge base"""
        try:
            response = self.client.get_knowledge_base(
                knowledgeBaseId=knowledge_base_id
            )
            return response
        except Exception as e:
            print(f"Error getting knowledge base details: {str(e)}")
            return None

def main():
    # Initialize the knowledge base manager
    kb_manager = BedrockKnowledgeBaseManager()
    
    # List available knowledge bases
    knowledge_bases = kb_manager.list_knowledge_bases()
    
    if not knowledge_bases:
        print("No knowledge bases found. Please create a knowledge base first.")
        return

    # Use the first available knowledge base for demonstration
    knowledge_base_id = knowledge_bases[0]['knowledgeBaseId']
    print(f"Using knowledge base: {knowledge_bases[0]['name']}")

    # Initialize the demo class
    kb_demo = BedrockKnowledgeBaseDemo(knowledge_base_id)

    try:
        # Example 1: Simple Retrieval
        print("\n=== Simple Retrieval Example ===")
        query = "What are the best practices for cloud security?"
        results = kb_demo.simple_retrieval(query)
        print(f"\nQuery: {query}")
        print("\nRetrieved Documents:")
        for idx, result in enumerate(results, 1):
            print(f"\nDocument {idx}:")
            print(f"Content: {result['content'][:200]}...")
            print(f"Metadata: {result['metadata']}")

        # Example 2: QA with Sources
        print("\n=== QA with Sources Example ===")
        question = "How do you implement multi-factor authentication?"
        response = kb_demo.qa_with_sources(question)
        print(f"\nQuestion: {question}")
        print(f"Answer: {response['answer']}")
        print("\nSources:")
        for source in response['sources']:
            print(f"- {source}")

        # Example 3: Similarity Search
        print("\n=== Similarity Search Example ===")
        query = "AWS security best practices"
        results = kb_demo.similarity_search(query, k=2)
        print(f"\nQuery: {query}")
        print("\nSimilar Documents:")
        for idx, result in enumerate(results, 1):
            print(f"\nDocument {idx}:")
            print(f"Content: {result['content'][:200]}...")
            print(f"Metadata: {result['metadata']}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
