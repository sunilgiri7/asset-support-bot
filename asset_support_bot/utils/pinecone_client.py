import os
from pinecone import Pinecone, ServerlessSpec
from django.conf import settings
from sentence_transformers import SentenceTransformer

class PineconeClient:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PineconeClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize Pinecone client and load embedding model"""
        # Get environment variables
        api_key = os.getenv('PINECONE_API_KEY')
        # environment variable can be used for additional configuration if needed
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'asset-support-index')
        
        # Create an instance of the Pinecone class
        self.pc = Pinecone(api_key=api_key)
        
        # Optionally check if the index exists; since you already created it on the website, this step may be skipped
        existing_indexes = self.pc.list_indexes().names()
        if self.index_name not in existing_indexes:
            # Create index if it does not exist (using your index's dimensions and region)
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,  # your index dimension is 1024
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        
        # Connect to the index
        self.index = self.pc.Index(self.index_name)
        
        # Load the embedding model (ensure that settings.EMBEDDING_MODEL_ID is defined, e.g., 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_ID)
    
    def generate_embedding(self, text):
        """Generate embedding for a text segment"""
        return self.embedding_model.encode(text).tolist()
    
    def store_document_chunks(self, chunks, asset_id, document_id):
        vectors = []
        for i, chunk in enumerate(chunks):
            # Generate a unique ID for this chunk
            vector_id = f"{document_id}_{i}"
            # Generate embedding for the chunk
            embedding = self.generate_embedding(chunk)
            # Prepare metadata
            metadata = {
                "asset_id": str(asset_id),
                "document_id": str(document_id),
                "chunk_index": i,
                "text": chunk,
            }
            # Add to vectors list
            vectors.append((vector_id, embedding, metadata))
        
        # Upsert in batches (Pinecone recommends upserting in batches, e.g., batch_size=100)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            ids, embeddings, metadatas = zip(*batch)
            self.index.upsert(vectors=zip(ids, embeddings, metadatas))
        return True
    
    def query_similar_chunks(self, query_text, asset_id, top_k=5):
        # Generate query embedding
        query_embedding = self.generate_embedding(query_text)
        # Query Pinecone with metadata filter for the specific asset
        results = self.index.query(
            vector=query_embedding,
            filter={"asset_id": str(asset_id)},
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract and return the relevant chunks
        chunks = []
        for match in results.matches:
            chunks.append({
                "text": match.metadata["text"],
                "score": match.score,
                "document_id": match.metadata["document_id"]
            })
        return chunks
    
    def delete_document(self, document_id):
        """
        Delete all chunks associated with a document
        
        Args:
            document_id (str): The document ID to delete
            
        Returns:
            bool: Success status
        """
        self.index.delete(filter={"document_id": str(document_id)})
        return True
