# pgvector integration with llama-index

# Documentation:
# https://docs.llamaindex.ai/en/v0.10.17/examples/vector_stores/postgres.html

# This low-level interface is very helpful
# https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/


# The sequence of operations: Init'ing, Loading, indexing, storing, querying

from typing import Any, Dict, List, Optional, Type, cast
import os
from typing import Any, Optional, cast

from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import SimpleDirectoryReader, StorageContext

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
import psycopg2

# class PGVectorStoreWithLlamaIndex(LlamaIndexVectorStore):
class PGVectorStoreWithLlamaIndex():
    _li_class = None # LlamaIndex integration check

    def _get_li_class(self):
        try:
            from llama_index.vector_stores.postgres import PGVectorStore as LIPGVectorStore
        except ImportError:
            raise ImportError("Please install missing package: pip install llama-index-vector-stores-postgres")
        return LIPGVectorStore

    def __init__(self, url: str, **kwargs: Any):
        import logging
        import sys

        # Uncomment to see debug logs
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


        connection_string = "postgresql://postgres:password@localhost:5432"
        db_name = "vector_db"
        conn = psycopg2.connect(connection_string)
        conn.autocommit = True

        with conn.cursor() as c:
            c.execute(f"DROP DATABASE IF EXISTS {db_name}")
            c.execute(f"CREATE DATABASE {db_name}")
        
        from sqlalchemy import make_url
        self.url = make_url(connection_string)

        print(f"url -- host: {self.url.host}, port: {self.url.port}, user: {self.url.username}, password: {self.url.password} ")

        # creat a pgvector store client
        self.vector_store = PGVectorStore.from_params(
            database=db_name,
            host=self.url.host,
            password=self.url.password,
            port=self.url.port,
            user=self.url.username,
            table_name="generic_table",
            embed_dim=1536,  # openai embedding dimension
        )
        # assign pgvector as the vector_store to the context    
        # self.storage_context = StorageContext.from_defaults(vector_store=vector_store)

    
    from llama_index.core import Document
    def generate_embeddings(self, documents: List[Document]):

        from llama_index.embeddings.openai import OpenAIEmbedding

        self.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")       
        
        from llama_index.core.node_parser import SentenceSplitter
        text_parser = SentenceSplitter(
            chunk_size=1024,
            # separator=" ",
        )
        text_chunks = []
        # maintain relationship with source doc index, to help inject doc metadata in (3)
        doc_idxs = []
        for doc_idx, doc in enumerate(documents):
            cur_text_chunks = text_parser.split_text(doc.text)
            text_chunks.extend(cur_text_chunks)
            doc_idxs.extend([doc_idx] * len(cur_text_chunks))
        
        # 3. Manually Construct Nodes from Text Chunks
        from llama_index.core.schema import TextNode

        nodes = []
        for idx, text_chunk in enumerate(text_chunks):
            node = TextNode(
                text=text_chunk,
            )
            src_doc = documents[doc_idxs[idx]]
            node.metadata = src_doc.metadata
            nodes.append(node)
        
        # 4. Generate Embeddings for each Node
        # Here we generate embeddings for each Node using OpenAI embedding model.
        for node in nodes:
            node_embedding = self.embed_model.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            node.embedding = node_embedding
        return nodes
    
    from llama_index.core.schema import TextNode
    def add_nodes_to_pgvector_DB(self, nodes: List[TextNode]):
        # 5. Load Nodes into a Vector Store
        # We now insert these nodes into our PostgresVectorStore.

        self.vector_store.add(nodes)

    def load_data(self, document_names: List[str]):
        self.documents = SimpleDirectoryReader(document_names).load_data()
        # print(f"Loaded {len(self.documents)} documents")
        # print(f"Document ID: {self.documents[0].doc_id}")
        # print(f"Document type: {type(self.documents)}")
        # print(f"Each Document type: {type(self.documents[0])}")
        return self.documents
    
    # def build_index(self):
    #     self.index = VectorStoreIndex.from_documents(
    #         self.documents, storage_context=self.storage_context, show_progress=True
    #     )
    #     self.query_engine = self.index.as_query_engine()

    def query(self, query_string: str) -> str:
        return self.query_engine.query(query_string)

    def query_low_level(self, query_string: str) -> str:

        # Generate a Query Embedding
        query_embedding = self.embed_model.get_query_embedding(query_string)
        
        # 2. Query the Vector Database
        # construct vector store query
        from llama_index.core.vector_stores import VectorStoreQuery

        query_mode = "default"
        # query_mode = "sparse"
        # query_mode = "hybrid"

        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding, similarity_top_k=5, mode=query_mode
        )
        # returns a VectorStoreQueryResult
        query_result = self.vector_store.query(vector_store_query)
        return query_result.nodes[0].get_content()


if __name__ == "__main__":
    pgvector = PGVectorStoreWithLlamaIndex(url="postgresql://postgres:password@localhost:5432", document_names="data/") # directory, not file
    documents = pgvector.load_data("data")
    nodes = pgvector.generate_embeddings(documents)
    pgvector.add_nodes_to_pgvector_DB(nodes)
    print(pgvector.query_low_level("Who does Paul Graham think of with the word schtick?"))