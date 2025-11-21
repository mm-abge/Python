import os
import uuid
import json
import logging
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    EmbedRequest, EmbedResponse, AnalyzeRequest, AnalyzeResponse,
    SimilarIncident, ErrorResponse
)
from .openai_client import get_embedding, chat_completion
from .search_client import upsert_documents, vector_search
from .utils import parse_timestamp, chunk_text, sanitize_log_content

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Incident GenAI API",
    description="AI-powered incident analysis and knowledge base",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.get("/", tags=["root"])
def read_root():
    """Health check endpoint"""
    return {
        "service": "Incident GenAI API",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/embed-and-store", response_model=EmbedResponse, tags=["incidents"])
async def embed_and_store(payload: EmbedRequest):
    """
    Embed incident logs and store in vector database.
    
    - Chunks long logs into manageable pieces
    - Generates embeddings for each chunk
    - Stores in Azure Cognitive Search with metadata
    """
    try:
        # Sanitize input
        logs_text = sanitize_log_content(payload.logs)
        if not logs_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Logs content is empty after sanitization"
            )
        
        timestamp = parse_timestamp(payload.timestamp)
        chunks = chunk_text(logs_text, max_chars=2000, overlap=200)
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No content to index after chunking"
            )
        
        logger.info(f"Processing {len(chunks)} chunks for resource '{payload.resource}'")
        
        docs = []
        doc_ids = []
        
        for idx, chunk in enumerate(chunks):
            try:
                emb = get_embedding(chunk)
                doc_id = str(uuid.uuid4())
                
                doc = {
                    "id": doc_id,
                    "content": chunk,
                    "embedding": emb,
                    "timestamp": timestamp,
                    "resource": payload.resource,
                    "severity": payload.severity or "unknown",
                    "chunk_index": idx,
                    "total_chunks": len(chunks)
                }
                docs.append(doc)
                doc_ids.append(doc_id)
                
            except Exception as e:
                logger.error(f"Failed to process chunk {idx}: {str(e)}")
                continue
        
        if not docs:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate embeddings for any chunks"
            )
        
        # Upload to search index
        result = upsert_documents(docs)
        
        logger.info(f"Successfully indexed {result['succeeded']} documents")
        
        return EmbedResponse(
            status="success",
            indexed_documents=result['succeeded'],
            document_ids=doc_ids
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in embed_and_store: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to embed and store: {str(e)}"
        )

@app.post("/analyze-incident", response_model=AnalyzeResponse, tags=["incidents"])
async def analyze_incident(payload: AnalyzeRequest):
    """
    Analyze new incident using RAG (Retrieval Augmented Generation).
    
    - Finds similar historical incidents
    - Uses LLM to generate analysis with context
    - Returns summary, root cause, and recommendations
    """
    try:
        # Sanitize input
        logs_text = sanitize_log_content(payload.logs)
        if not logs_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Logs content is empty after sanitization"
            )
        
        # Create search snippet (limit for embedding)
        snippet = logs_text[:1500] if len(logs_text) > 1500 else logs_text
        
        logger.info(f"Analyzing incident for resource '{payload.resource}'")
        
        # Generate embedding and search
        query_vector = get_embedding(snippet)
        similar_docs = vector_search(query_vector, top_k=payload.top_k or 3)
        
        # Build context from similar incidents
        similar_contexts = []
        similar_incidents = []
        
        for doc in similar_docs:
            context = f"[{doc.get('resource', 'unknown')}] [{doc.get('severity', 'unknown')}] {doc.get('content', '')[:800]}"
            similar_contexts.append(context)
            
            similar_incidents.append(SimilarIncident(
                id=doc.get('id', 'unknown'),
                content=doc.get('content', '')[:500],
                timestamp=doc.get('timestamp', ''),
                resource=doc.get('resource', 'unknown'),
                severity=doc.get('severity'),
                score=doc.get('score', 0.0)
            ))
        
        similar_text = "\n\n".join(similar_contexts) if similar_contexts else "No similar incidents found."
        
        # Build RAG prompt
        user_prompt = f"""
Analyze this new incident:

INCIDENT LOGS:
{snippet}

SIMILAR HISTORICAL INCIDENTS:
{similar_text}

Provide a JSON response with:
1. "summary": Brief 2-3 sentence summary of the incident
2. "probable_root_cause": Most likely root cause based on logs and similar incidents
3. "recommended_action": Specific actionable steps to resolve
4. "confidence": "high", "medium", or "low" based on similarity to past incidents

Return ONLY valid JSON, no markdown or extra text.
"""
        
        system_prompt = """You are an expert SRE/DevOps engineer analyzing system incidents. 
Be concise, technical, and actionable. Base your analysis on the provided logs and historical context.
Always return valid JSON only."""
        
        # Get LLM analysis
        model_output = chat_completion(system_prompt, user_prompt, max_tokens=600)
        
        # Parse JSON response
        try:
            # Clean potential markdown formatting
            clean_output = model_output.strip()
            if clean_output.startswith("```json"):
                clean_output = clean_output[7:]
            if clean_output.startswith("```"):
                clean_output = clean_output[3:]
            if clean_output.endswith("```"):
                clean_output = clean_output[:-3]
            clean_output = clean_output.strip()
            
            parsed = json.loads(clean_output)
            
            return AnalyzeResponse(
                summary=parsed.get("summary", "No summary provided"),
                probable_root_cause=parsed.get("probable_root_cause", "Unknown"),
                recommended_action=parsed.get("recommended_action", "No recommendation available"),
                similar_incidents=similar_incidents,
                confidence=parsed.get("confidence", "medium")
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}. Raw output: {model_output}")
            # Fallback response
            return AnalyzeResponse(
                summary=model_output[:200],
                probable_root_cause="Unable to determine - check raw analysis",
                recommended_action="Review the summary and similar incidents manually",
                similar_incidents=similar_incidents,
                confidence="low"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_incident: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze incident: {str(e)}"
        )

@app.get("/health", tags=["health"])
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "openai": "configured" if os.environ.get("OPENAI_KEY") else "missing",
            "search": "configured" if os.environ.get("SEARCH_ADMIN_KEY") else "missing"
        }
    }