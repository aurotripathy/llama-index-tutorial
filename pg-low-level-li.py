# pgvector integration with llama-index (at a low level)

# Documentation:
# https://docs.llamaindex.ai/en/v0.10.17/examples/vector_stores/postgres.html

# This low-level interface is very helpful
# Look for side tab, Building Retrieval from Scratch
# https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/


# The sequence of operations for RAG: 
# Init'ing, Loading, indexing, storing, querying, generating

from typing import Any, Dict, List, Optional, Type, cast
import os
from typing import Any, Optional, cast

from llama_index.core import SimpleDirectoryReader, StorageContext

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.schema import NodeWithScore
from llama_index.core import Document
from llama_index.core.schema import TextNode, QueryBundle
from llama_index.embeddings.openai import OpenAIEmbedding

import psycopg2

# constants
embed_model_name = "text-embedding-3-small"
generating_model_name = "gpt-4o-mini"


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
        # Thats it! We're using the low-level interface to create a context
    
    
    def generate_embeddings(self, documents: List[Document]) -> List[TextNode]:

        self.embed_model = OpenAIEmbedding(model=embed_model_name)       
        
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
    
    
    def add(self, embeddings: List[TextNode]):
        """Add vector embeddings to vector stores

        Args:
            nodes: List of nodes
        Returns:
            List of ids of the embeddings
        """
        # 5. Load Nodes into a Vector Store
        # We now insert these nodes into our PostgresVectorStore.

        self.vector_store.add(embeddings)

    def load_data(self, document_names: List[str]):
        self.documents = SimpleDirectoryReader(document_names).load_data()
        # print(f"Loaded {len(self.documents)} documents")
        # print(f"Document ID: {self.documents[0].doc_id}")
        # print(f"Document type: {type(self.documents)}")
        # print(f"Each Document type: {type(self.documents[0])}")
        return self.documents
    

    def query(self, query_string: str) -> str:
        return self.query_engine.query(query_string)

    def build_retriever(self):
        from retriever import VectorDBRetriever
        retriever = VectorDBRetriever(vector_store=self.vector_store, embed_model=self.embed_model)
        return retriever



if __name__ == "__main__":

    from llama_index.llms.openai import OpenAI

    query_str = "Who does Paul Graham think of with the word, 'schtick'?"
    print(f"query_str: {query_str}")

    llm = OpenAI(
    model=generating_model_name, # api_key="some key",  # uses OPENAI_API_KEY env var by default
    )
    pgvector = PGVectorStoreWithLlamaIndex(url="postgresql://postgres:password@localhost:5432", document_names="data/") # directory, not file
    documents = pgvector.load_data("data")
    nodes = pgvector.generate_embeddings(documents)
    pgvector.add(nodes)
    # retrieved_nodes = pgvector.query_for_low_level_results(query_str)
    retriever = pgvector.build_retriever()
    from llama_index.core.query_engine import RetrieverQueryEngine

    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
    retrieved_nodes = query_engine.retriever._retrieve(QueryBundle(query_str))   
    print(f"retrieved_nodes: {retrieved_nodes}")

    # kotaemon integration format
    retrieved_nodes_for_kotaemon, scores_for_kotaemon, ids_for_kotaemon = query_engine.retriever.retrieve_for_kotaemon(QueryBundle(query_str))
    print(f"retrieved_nodes_for_kotaemon: {retrieved_nodes_for_kotaemon}")
    print(f"scores_for_kotaemon: {scores_for_kotaemon}")
    print(f"ids_for_kotaemon: {ids_for_kotaemon}")

    response = query_engine.query(query_str)
    print(str(response))
