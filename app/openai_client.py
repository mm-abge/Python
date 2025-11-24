import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
from typing import List

# Load environment variables from .env file in project root
# project_root = Path(__file__).parent.parent
# env_file = project_root / '.env'
# load_dotenv(dotenv_path=env_file, override=True)

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)  # ← Fixed: removed extra indentation

logger = logging.getLogger(__name__)

# Environment variables
OPENAI_ENDPOINT = os.environ.get("OPENAI_ENDPOINT")
OPENAI_KEY = os.environ.get("OPENAI_KEY")
OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION", "2024-02-15-preview")
EMBED_DEPLOY = os.environ.get("EMBED_DEPLOY")
CHAT_DEPLOY = os.environ.get("CHAT_DEPLOY")

# Debug print (remove after testing)
print(f"DEBUG: OPENAI_ENDPOINT loaded: {OPENAI_ENDPOINT}")
print(f"DEBUG: EMBED_DEPLOY loaded: {EMBED_DEPLOY}")

if not all([OPENAI_ENDPOINT, OPENAI_KEY, EMBED_DEPLOY, CHAT_DEPLOY]):
    raise EnvironmentError(  # ← Fixed: proper indentation
        "Missing required environment variables: OPENAI_ENDPOINT, OPENAI_KEY, "
        "EMBED_DEPLOY, CHAT_DEPLOY"
    )

# Initialize Azure OpenAI client
_openai_client = AzureOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version=OPENAI_API_VERSION
)

def get_embedding(text: str) -> List[float]:
    """Generate embedding for a single text string."""
    try:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
            
        response = _openai_client.embeddings.create(
            model=EMBED_DEPLOY,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise Exception(f"Failed to generate embedding: {str(e)}")

def chat_completion(
    system_prompt: str, 
    user_prompt: str, 
    max_tokens: int = 400,
    temperature: float = 0.7
) -> str:
    """Generate chat completion using Azure OpenAI."""
    try:
        response = _openai_client.chat.completions.create(
            model=CHAT_DEPLOY,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating chat completion: {str(e)}")
        raise Exception(f"Failed to generate chat completion: {str(e)}")