
# https://docs.llamaindex.ai/en/stable/getting_started/starter_example/

import os
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    load_index_from_storage,
)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# query your data
query_engine = index.as_query_engine()
response = query_engine.query("Who is Paul Graham?")
print(response)

# logging
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

# storing your index
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the index from storage
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST_DIR))

# Either way we can now query the index
query_engine = index.as_query_engine()
query_str = "What did the author do growing up?"
response = query_engine.query(query_str)
print(query_str)
print(response)
