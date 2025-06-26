import os.path
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from llama_index.core import (
 Settings,
 VectorStoreIndex,
 SimpleDirectoryReader,
 StorageContext,
 load_index_from_storage,
 )

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike


Settings.llm = OpenAILike(model="deepseek-chat",
                          api_base="https://api.deepseek.com",
                          api_key="sk-e973b0ca02334e9e9664bb2736babc5c",
                          is_chat_model=True)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")

 
# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
 # load the documents and create the index
 documents = SimpleDirectoryReader("data").load_data()
 index = VectorStoreIndex.from_documents(documents)
 # store it for later
 index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
 # load the existing index
 storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
 index = load_index_from_storage(storage_context)


 # Either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)

