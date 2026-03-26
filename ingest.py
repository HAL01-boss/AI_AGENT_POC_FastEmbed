import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

load_dotenv()

# --- Configuration ---
DOSSIER_ACENOS = r"C:\Users\HoudaALOUANE\AppData\Local\Python\bin\knowledge_agent\POC_CohereEmbed\docs"  # ← adapte si besoin
EXTENSIONS_AUTORISEES = [".pdf", ".docx", ".pptx", ".txt", ".xlsx"]

# --- Embeddings Cohere (gratuit, multilingue, excellent pour le français) ---
Settings.embed_model = CohereEmbedding(
    api_key=os.getenv("COHERE_API_KEY"),
    model_name="embed-multilingual-v3.0",
    input_type="search_document",  # mode indexation
)
Settings.chunk_size = 512
Settings.chunk_overlap = 64

# --- Connexion Qdrant ---
client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
vector_store = QdrantVectorStore(client=client, collection_name="acenos_kb")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- Chargement récursif de tous les fichiers ---
print(f"📂 Scan du dossier : {DOSSIER_ACENOS}")
documents = SimpleDirectoryReader(
    input_dir=DOSSIER_ACENOS,
    recursive=True,
    required_exts=EXTENSIONS_AUTORISEES,
    filename_as_id=True,
    errors="ignore",
).load_data()

print(f"✅ {len(documents)} chunks chargés")
print(f"🔄 Indexation dans Qdrant en cours...")

VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True,
)

print("🎉 Indexation terminée !")