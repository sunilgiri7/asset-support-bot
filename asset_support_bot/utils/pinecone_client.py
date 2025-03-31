import logging
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache
import time
import concurrent.futures

logger = logging.getLogger(__name__)

class PineconeClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PineconeClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize Pinecone client and load an optimized embedding model."""
        try:
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise ValueError("Pinecone API key is not set")
            
            self.index_name = os.getenv('PINECONE_INDEX_NAME', 'asset-support-index')
            self.pc = Pinecone(api_key=api_key)
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # Using a lighter model dimension (MiniLM)
                    metric="cosine",
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            self.index = self.pc.Index(self.index_name)
            
            # Use a lighter and faster embedding model by default.
            model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            self.embedding_model = SentenceTransformer(model_name)
            
            # In-memory cache for query results (key: query parameters, value: result)
            self.query_cache = {}
            
            logger.info(f"Pinecone client initialized successfully using model {model_name}")
        except Exception as e:
            logger.error(f"Pinecone initialization error: {str(e)}")
            raise

    @lru_cache(maxsize=1024)
    def generate_embedding(self, text):
        """Generate and cache embedding for a text segment."""
        try:
            if not text or not text.strip():
                logger.warning("Attempted to generate embedding for empty text")
                return None
            
            start_time = time.perf_counter()
            embedding = self.embedding_model.encode(text)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                logger.warning("Zero norm encountered during embedding generation")
                return None
            embedding = embedding / norm
            elapsed = time.perf_counter() - start_time
            logger.debug(f"Generated embedding in {elapsed:.4f} seconds for text: {text[:30]}...")
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding generation error: {str(e)}")
            return None

    def store_document_chunks(self, chunks, asset_id, document_id):
        """
        Store document chunks in Pinecone with robust error handling.
        """
        try:
            if not chunks:
                logger.warning("No chunks provided for embedding")
                return False
            
            vectors = []
            for i, chunk in enumerate(chunks):
                if not chunk or not chunk.strip():
                    logger.debug(f"Skipping empty chunk {i}")
                    continue
                
                vector_id = f"{document_id}_{i}"
                embedding = self.generate_embedding(chunk)
                if embedding is None:
                    logger.warning(f"Failed to generate embedding for chunk {i}")
                    continue
                
                asset_id_str = str(asset_id)
                metadata = {
                    "asset_id": asset_id_str,
                    "document_id": str(document_id),
                    "chunk_index": i,
                    "text": chunk,
                }
                vectors.append((vector_id, embedding, metadata))
            
            if vectors:
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i+batch_size]
                    ids, embeddings, metadatas = zip(*batch)
                    self.index.upsert(vectors=zip(ids, embeddings, metadatas))
                logger.info(f"Successfully stored {len(vectors)} chunks for document {document_id}")
                return True
            else:
                logger.warning("No valid vectors to store")
                return False
        
        except Exception as e:
            logger.error(f"Error storing document chunks: {str(e)}")
            return False

    def _generate_cache_key(self, query_text, asset_id, top_k, similarity_threshold):
        """Generate a cache key for query results."""
        return f"{query_text}:{asset_id}:{top_k}:{similarity_threshold}"
    
    def query_similar_chunks(self, query_text, asset_id, top_k=3, similarity_threshold=0.7):
        """
        Query similar chunks with robust logging and filtering.
        Results are cached to avoid repeated expensive queries.
        This version offloads the Pinecone query to a thread to reduce blocking time.
        """
        try:
            asset_id_str = str(asset_id)
            cache_key = self._generate_cache_key(query_text, asset_id_str, top_k, similarity_threshold)
            if cache_key in self.query_cache:
                logger.info("Returning cached query results.")
                return self.query_cache[cache_key]
            
            logger.info(f"Querying Pinecone with Asset ID: {asset_id_str}")
            query_embedding = self.generate_embedding(query_text)
            if query_embedding is None:
                logger.warning("Query embedding generation failed")
                return []
            
            start_time = time.perf_counter()
            # Offload the blocking query to a thread.
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self.index.query,
                    vector=query_embedding,
                    filter={"asset_id": asset_id_str},
                    top_k=top_k,
                    include_metadata=True
                )
                # Set a timeout for the query (adjust as necessary).
                results = future.result(timeout=5)
            query_time = time.perf_counter() - start_time
            logger.info(f"Pinecone index query completed in {query_time:.4f} seconds")
            
            chunks = []
            for match in results.matches:
                score = match.score
                logger.debug(f"Match Score: {score}")
                if score >= similarity_threshold:
                    chunk_info = {
                        "text": match.metadata.get("text", ""),
                        "score": score,
                        "document_id": match.metadata.get("document_id", ""),
                        "chunk_index": match.metadata.get("chunk_index", -1)
                    }
                    chunks.append(chunk_info)
            
            logger.info(f"Found {len(chunks)} similar chunks for asset {asset_id_str}")
            # Cache the result.
            self.query_cache[cache_key] = chunks
            return chunks
        
        except Exception as e:
            logger.error(f"Error querying similar chunks: {str(e)}")
            return []

    def debug_index_contents(self, asset_id):
        """
        Debug method to inspect index contents for a specific asset.
        """
        try:
            asset_id_str = str(asset_id)
            results = self.index.query(
                vector=[0]*384,  # Dummy vector adjusted for 384 dimensions
                filter={"asset_id": asset_id_str},
                top_k=100,
                include_metadata=True
            )
            debug_info = []
            for match in results.matches:
                debug_info.append({
                    "id": match.id,
                    "metadata": match.metadata,
                    "score": match.score
                })
            logger.info(f"Debug: Found {len(debug_info)} vectors for asset {asset_id_str}")
            return debug_info
        
        except Exception as e:
            logger.error(f"Error in debug_index_contents: {str(e)}")
            return []
        
    def get_fallback_chunks(self, asset_id, limit=3):
        """
        Retrieve fallback document chunks for an asset when the main query does not return enough results.
        This fetches general information related to the asset from the index.
        """
        try:
            asset_id_str = str(asset_id)
            logger.info(f"Fetching fallback chunks for asset: {asset_id_str}")

            # Perform a broad query with a dummy vector (zero vector) to retrieve asset-related chunks
            fallback_results = self.index.query(
                vector=[0] * 384,  # Zero vector for broad retrieval
                filter={"asset_id": asset_id_str},
                top_k=limit,
                include_metadata=True
            )

            fallback_chunks = []
            for match in fallback_results.matches:
                chunk_info = {
                    "text": match.metadata.get("text", ""),
                    "score": match.score,
                    "document_id": match.metadata.get("document_id", ""),
                    "chunk_index": match.metadata.get("chunk_index", -1)
                }
                fallback_chunks.append(chunk_info)

            logger.info(f"Fetched {len(fallback_chunks)} fallback chunks for asset {asset_id_str}")
            return fallback_chunks

        except Exception as e:
            logger.error(f"Error retrieving fallback chunks: {str(e)}")
            return []


