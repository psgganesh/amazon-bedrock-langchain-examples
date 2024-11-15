import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'documents' folder
documents_folder = os.path.join(current_dir, 'documents')

# Load a PDF file
pdf_filename = "well-architected.pdf"
pdf_path = os.path.join(documents_folder, pdf_filename)
pdf_loader = PyPDFLoader(pdf_path)
pdf_docs = pdf_loader.load()

print(f"Loaded {len(pdf_docs)} pages from PDF")

# Load a text file
text_filename = "well-architected-document-revisions.txt"
text_path = os.path.join(documents_folder, text_filename)
text_loader = TextLoader(text_path)
text_docs = text_loader.load()

print(f"Loaded {len(text_docs)} documents from text file")

# Combine documents
all_docs = pdf_docs + text_docs

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

splits = text_splitter.split_documents(all_docs)

print(f"Total splits: {len(splits)}")
print(f"First split content: {splits[0].page_content[:100]}...")