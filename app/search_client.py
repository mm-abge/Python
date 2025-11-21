# ==================== app/search_client.py ====================
import os
import logging
from typing import List, Dict, Any
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)

# Environment variables
SEARCH_ENDPOINT = os.environ.get("SEARCH_ENDPOINT")
SEARCH_ADMIN_KEY = os.environ.get("SEARCH_ADMIN_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "incident-index")

if not all([SEARCH_ENDPOINT, SEARCH_ADMIN_KEY]):
    raise EnvironmentError(
        "Missing required environment variables: SEARCH_ENDPOINT, SEARCH_ADMIN_KEY"
    )

# Initialize Search client
search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_ADMIN_KEY)
)

def upsert_documents(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Upload or update documents in the search index.
    
    Args:
        documents: List of document dictionaries matching index schema
        
    Returns:
        Result summary with success/failure counts
        
    Raises:
        Exception: If document upload fails
    """
    try:
        if not documents:
            raise ValueError("Documents list cannot be empty")
            
        result = search_client.upload_documents(documents=documents)
        
        succeeded = sum(1 for r in result if r.succeeded)
        failed = len(result) - succeeded
        
        if failed > 0:
            logger.warning(f"Failed to upload {failed} out of {len(documents)} documents")
        
        return {
            "total": len(documents),
            "succeeded": succeeded,
            "failed": failed
        }
    except Exception as e:
        logger.error(f"Error upserting documents: {str(e)}")
        raise Exception(f"Failed to upsert documents: {str(e)}")

def vector_search(
    query_vector: List[float], 
    top_k: int = 3,
    filter_expr: str = None
) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search.
    
    Args:
        query_vector: Query embedding vector
        top_k: Number of top results to return
        filter_expr: Optional OData filter expression
        
    Returns:
        List of matching documents with scores
        
    Raises:
        Exception: If search fails
    """
    try:
        if not query_vector:
            raise ValueError("Query vector cannot be empty")
        
        if top_k < 1 or top_k > 50:
            raise ValueError("top_k must be between 1 and 50")
        
        from azure.search.documents.models import VectorizedQuery
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="embedding"
        )
        
        results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=filter_expr,
            select=["id", "content", "timestamp", "resource", "severity"],
            top=top_k
        )
        
        found = []
        for result in results:
            doc = {
                "id": result.get("id"),
                "content": result.get("content"),
                "timestamp": result.get("timestamp"),
                "resource": result.get("resource"),
                "severity": result.get("severity"),
                "score": result.get("@search.score")
            }
            found.append(doc)
        
        logger.info(f"Vector search returned {len(found)} results")
        return found
        
    except Exception as e:
        logger.error(f"Error performing vector search: {str(e)}")
        raise Exception(f"Failed to perform vector search: {str(e)}")