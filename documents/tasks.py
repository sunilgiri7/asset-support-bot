# documents/tasks.py
from celery import shared_task
import os
import logging
from .models import Document
from .utils import extract_text_from_file, chunk_text
from asset_support_bot.utils.pinecone_client import PineconeClient

logger = logging.getLogger(__name__)

@shared_task
def process_document(document_id):
    """
    Process a document to extract text and generate embeddings
    
    Args:
        document_id (str): UUID of the document to process
    """
    try:
        # Get the document
        document = Document.objects.get(id=document_id)
        
        # Get file path
        file_path = document.file.path
        
        # Extract text from document
        logger.info(f"Extracting text from document: {document.title}")
        text = extract_text_from_file(file_path)
        
        # Split text into chunks
        logger.info(f"Chunking text from document: {document.title}")
        chunks = chunk_text(text)
        
        if not chunks:
            logger.warning(f"No chunks created for document: {document.title}")
            document.is_processed = True
            document.processing_error = "No text content found or processed"
            document.save()
            return
        
        # Generate and store embeddings
        logger.info(f"Generating embeddings for document: {document.title}")
        pinecone_client = PineconeClient()
        pinecone_client.store_document_chunks(
            chunks=chunks,
            asset_id=document.asset_id,
            document_id=document.id
        )
        
        # Update document status
        document.is_processed = True
        document.save()
        
        logger.info(f"Successfully processed document: {document.title}")
        
    except Document.DoesNotExist:
        logger.error(f"Document with ID {document_id} not found")
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        # Update document with error
        try:
            document = Document.objects.get(id=document_id)
            document.processing_error = str(e)
            document.save()
        except:
            pass