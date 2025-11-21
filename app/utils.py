import logging
from datetime import datetime
from typing import List
from dateutil import parser as dtparser

logger = logging.getLogger(__name__)

def parse_timestamp(ts: str = None) -> str:
    """
    Parse timestamp string to ISO format.
    
    Args:
        ts: Timestamp string in various formats
        
    Returns:
        ISO format timestamp string
    """
    if not ts:
        return datetime.utcnow().isoformat() + "Z"
    
    try:
        parsed = dtparser.parse(ts)
        return parsed.isoformat() + ("Z" if parsed.tzinfo is None else "")
    except Exception as e:
        logger.warning(f"Failed to parse timestamp '{ts}': {e}. Using current time.")
        return datetime.utcnow().isoformat() + "Z"

def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Create overlap
    
    return chunks

def sanitize_log_content(content: str, max_length: int = 10000) -> str:
    """
    Sanitize log content for safe processing.
    
    Args:
        content: Raw log content
        max_length: Maximum allowed length
        
    Returns:
        Sanitized content
    """
    if not content:
        return ""
    
      # Remove null bytes and control characters
    content = ''.join(char for char in content if char.isprintable() or char in '\n\r\t')
    
    # Truncate if too long
    if len(content) > max_length:
        logger.warning(f"Content truncated from {len(content)} to {max_length} chars")
        content = content[:max_length]
    
    return content.strip()
