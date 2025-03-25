import logging
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class PineconeClient:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PineconeClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize Pinecone client and load embedding model"""
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
                    dimension=1024,
                    metric="cosine",
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            
            self.index = self.pc.Index(self.index_name)
            self.embedding_model = SentenceTransformer('BERT-large-nli-stsb-mean-tokens', device='cpu')
            
            logger.info("Pinecone client initialized successfully")
        
        except Exception as e:
            logger.error(f"Pinecone initialization error: {str(e)}")
            raise
    
    def generate_embedding(self, text):
        """Generate embedding for a text segment"""
        try:
            if not text or not text.strip():
                logger.warning("Attempted to generate embedding for empty text")
                return None
            
            embedding = self.embedding_model.encode(text)
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Embedding generation error: {str(e)}")
            return None
    
    def store_document_chunks(self, chunks, asset_id, document_id):
        """
        Store document chunks in Pinecone with robust error handling
        
        Args:
            chunks (list): List of text chunks
            asset_id (str): Asset identifier (ensure it's a string)
            document_id (str): Document identifier
        
        Returns:
            bool: Success status
        """
        try:
            if not chunks:
                logger.warning("No chunks provided for embedding")
                return False
            
            vectors = []
            for i, chunk in enumerate(chunks):
                if not chunk or not chunk.strip():
                    logger.warning(f"Skipping empty chunk {i}")
                    continue
                
                vector_id = f"{document_id}_{i}"
                embedding = self.generate_embedding(chunk)
                
                if embedding is None:
                    logger.warning(f"Failed to generate embedding for chunk {i}")
                    continue
                
                # Explicitly convert asset_id to string
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
    
    def query_similar_chunks(self, query_text, asset_id, top_k=5, similarity_threshold=0.7):
        """
        Enhanced method to query similar chunks with robust logging and filtering
        
        Args:
            query_text (str): Text to find similar chunks for
            asset_id (str or int): Asset identifier
            top_k (int): Number of top chunks to retrieve
            similarity_threshold (float): Minimum similarity score to include chunks
        
        Returns:
            list: Relevant context chunks
        """
        try:
            # Convert asset_id to string and log
            asset_id_str = str(asset_id)
            logger.info(f"Querying Pinecone with Asset ID: {asset_id_str}")
            
            # Generate query embedding
            query_embedding = self.generate_embedding(query_text)
            
            if query_embedding is None:
                logger.warning("Query embedding generation failed")
                return []
            
            # Perform Pinecone query with explicit filtering
            results = self.index.query(
                vector=query_embedding,
                filter={"asset_id": asset_id_str},
                top_k=top_k,
                include_metadata=True
            )
            
            # Enhanced chunk filtering and logging
            chunks = []
            for match in results.matches:
                score = match.score
                logger.info(f"Match Score: {score}")
                
                if score >= similarity_threshold:
                    chunk_info = {
                        "text": match.metadata.get("text", ""),
                        "score": score,
                        "document_id": match.metadata.get("document_id", ""),
                        "chunk_index": match.metadata.get("chunk_index", -1)
                    }
                    chunks.append(chunk_info)
            
            logger.info(f"Found {len(chunks)} similar chunks for asset {asset_id_str}")
            return chunks
        
        except Exception as e:
            logger.error(f"Error querying similar chunks: {str(e)}")
            return []

    def debug_index_contents(self, asset_id):
        """
        Debug method to inspect index contents for a specific asset
        
        Args:
            asset_id (str or int): Asset identifier to inspect
        
        Returns:
            list: Metadata of stored vectors
        """
        try:
            asset_id_str = str(asset_id)
            results = self.index.query(
                vector=[0]*1024,  # Dummy vector to match all
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