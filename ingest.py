from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Step 1: Load text from index.md using TextLoader
loader = TextLoader("state_of_the_union.txt",encoding="UTF-8")
documents = loader.load()

# Step 2: Split documents into chunks using CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Step 3: Generate embeddings for each chunk using OllamaEmbeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Step 4: Build FAISS index from the embeddings
db = FAISS.from_documents(docs, embeddings)

# Step 5: Print the number of vectors (optional)
print(f"Number of vectors in the FAISS index: {db.index.ntotal}")

# Step 6: Save the FAISS index locally
db.save_local("faiss_index")
print("FAISS index saved successfully.")

# Optionally, you can also load the FAISS index later:
# db = FAISS.load_local("faiss_index")
