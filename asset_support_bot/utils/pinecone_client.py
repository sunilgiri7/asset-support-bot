import os
import logging
from pinecone import Pinecone, ServerlessSpec
from django.conf import settings
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
            # Get environment variables
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise ValueError("Pinecone API key is not set")
            
            self.index_name = os.getenv('PINECONE_INDEX_NAME', 'asset-support-index')
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=api_key)
            
            # Create index if it doesn't exist
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1024,  # Dimension for all-MiniLM-L6-v2
                    metric="cosine",
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            
            # Load embedding model
            self.embedding_model = SentenceTransformer('BERT-large-nli-stsb-mean-tokens', device='cpu')
            
            logger.info("Pinecone client initialized successfully")
        
        except Exception as e:
            logger.error(f"Pinecone initialization error: {str(e)}")
            raise
    
    def generate_embedding(self, text):
        """
        Generate embedding for a text segment
        
        Args:
            text (str): Input text to embed
        
        Returns:
            list: Embedding vector
        """
        try:
            # Ensure text is not empty
            if not text or not text.strip():
                logger.warning("Attempted to generate embedding for empty text")
                return None
            
            # Generate embedding
            embedding = self.embedding_model.encode(text)
            
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Embedding generation error: {str(e)}")
            return None
    
    def store_document_chunks(self, chunks, asset_id, document_id):
        """
        Store document chunks in Pinecone
        
        Args:
            chunks (list): List of text chunks
            asset_id (str): Asset identifier
            document_id (str): Document identifier
        
        Returns:
            bool: Success status
        """
        try:
            # Validate inputs
            if not chunks:
                logger.warning("No chunks provided for embedding")
                return False
            
            vectors = []
            for i, chunk in enumerate(chunks):
                # Skip empty chunks
                if not chunk or not chunk.strip():
                    logger.warning(f"Skipping empty chunk {i}")
                    continue
                
                # Generate unique vector ID
                vector_id = f"{document_id}_{i}"
                
                # Generate embedding
                embedding = self.generate_embedding(chunk)
                
                if embedding is None:
                    logger.warning(f"Failed to generate embedding for chunk {i}")
                    continue
                
                # Prepare metadata
                metadata = {
                    "asset_id": str(asset_id),
                    "document_id": str(document_id),
                    "chunk_index": i,
                    "text": chunk,
                }
                
                # Add to vectors list
                vectors.append((vector_id, embedding, metadata))
            
            # Upsert vectors in batches
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
    
    def query_similar_chunks(self, query_text, asset_id, top_k=5):
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query_text)
            
            if query_embedding is None:
                logger.warning("Query embedding generation failed")
                return []
            
            # Log the asset_id and query embedding
            logger.info(f"Querying for asset_id: {asset_id}")
            logger.info(f"Query embedding shape: {len(query_embedding)}")
            
            # Query Pinecone with metadata filter for the specific asset
            results = self.index.query(
                vector=query_embedding,
                filter={"asset_id": str(asset_id)},
                top_k=top_k,
                include_metadata=True
            )
            
            # Log the query results
            logger.info(f"Query results: {results}")
            
            # Extract and return the relevant chunks
            chunks = []
            for match in results.matches:
                chunks.append({
                    "text": match.metadata.get("text", ""),
                    "score": match.score,
                    "document_id": match.metadata.get("document_id", ""),
                    "chunk_index": match.metadata.get("chunk_index", -1)
                })
            
            logger.info(f"Found {len(chunks)} similar chunks for asset {asset_id}")
            return chunks
        except Exception as e:
            logger.error(f"Error querying similar chunks: {str(e)}")
            return []