import os
import tiktoken
from neo4j import GraphDatabase
from typing import List, Union, Optional

# Import Bedrock embedding functionality
from embedding import BedrockEmbedding, create_embeddings

# Import Strands for agent functionality
try:
    from strands import Agent
    from strands.models import BedrockModel
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    print("Warning: Strands not available. Agent functionality will be disabled.")

# Neo4j Docker connection configuration
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password123")

neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    notifications_min_severity="OFF",
    max_connection_lifetime=30 * 60,  # 30 minutes
    max_connection_pool_size=50,
    connection_acquisition_timeout=60  # 60 seconds
)

# Only use Bedrock/Strands - no OpenAI
OPENAI_AVAILABLE = False
open_ai_client = None

# Initialize Bedrock embedding client
bedrock_embedder = BedrockEmbedding(
    region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2")
)

# Initialize Strands agent if available
if STRANDS_AVAILABLE:
    bedrock_model = BedrockModel(
        model_id=os.environ.get("BEDROCK_MODEL_ID", "apac.anthropic.claude-sonnet-4-20250514-v1:0"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2"),
        temperature=0.3,
    )
    strands_agent = Agent(model=bedrock_model)
else:
    strands_agent = None


def test_neo4j_connection():
    """Test Neo4j connection and return basic info"""
    try:
        with neo4j_driver.session() as session:
            result = session.run("CALL db.ping()")
            return {"status": "connected", "message": "Neo4j connection successful"}
    except Exception as e:
        return {"status": "error", "message": f"Neo4j connection failed: {str(e)}"}


def get_neo4j_info():
    """Get Neo4j database information"""
    try:
        with neo4j_driver.session() as session:
            result = session.run("CALL dbms.components() YIELD name, versions, edition")
            info = result.single()
            return {
                "name": info["name"],
                "versions": info["versions"],
                "edition": info["edition"]
            }
    except Exception as e:
        return {"error": f"Failed to get Neo4j info: {str(e)}"}


def chunk_text(text, chunk_size, overlap, split_on_whitespace_only=True):
    chunks = []
    index = 0

    while index < len(text):
        if split_on_whitespace_only:
            prev_whitespace = 0
            left_index = index - overlap
            while left_index >= 0:
                if text[left_index] == " ":
                    prev_whitespace = left_index
                    break
                left_index -= 1
            next_whitespace = text.find(" ", index + chunk_size)
            if next_whitespace == -1:
                next_whitespace = len(text)
            chunk = text[prev_whitespace:next_whitespace].strip()
            chunks.append(chunk)
            index = next_whitespace + 1
        else:
            start = max(0, index - overlap + 1)
            end = min(index + chunk_size + overlap, len(text))
            chunk = text[start:end].strip()
            chunks.append(chunk)
            index += chunk_size

    return chunks


def num_tokens_from_string(string: str, model: str = "gpt-4") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def embed(texts: Union[str, List[str]], **kwargs) -> List[List[float]]:
    """
    Create embeddings using Bedrock Titan
    
    Args:
        texts: Text or list of texts to embed
        **kwargs: Additional arguments for embedding models (dimensions, normalize)
    
    Returns:
        List of embedding vectors
    """
    # Always use Bedrock Titan embedding
    return bedrock_embedder.embed_text(texts, **kwargs)


def embed_bedrock(texts: Union[str, List[str]], dimensions: int = 1024, normalize: bool = True) -> List[List[float]]:
    """
    Create embeddings using Amazon Bedrock Titan
    
    Args:
        texts: Text or list of texts to embed
        dimensions: Output dimensions (256, 512, or 1024)
        normalize: Whether to normalize embeddings
    
    Returns:
        List of embedding vectors
    """
    return bedrock_embedder.embed_text(texts, dimensions=dimensions, normalize=normalize)


def embed_openai(texts: Union[str, List[str]], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    OpenAI embeddings are disabled. Use embed_bedrock() instead.
    """
    raise RuntimeError("OpenAI functionality is disabled. Use embed_bedrock() or embed() instead.")


def chat(messages, model: Optional[str] = None, model_id: Optional[str] = None, temperature: float = 0.3, **kwargs) -> str:
    """
    Chat using Bedrock models via Strands (Bedrock-only)
    
    Args:
        messages: List of message dictionaries or single message string
        model: Model name (ignored, always uses Bedrock)
        model_id: Bedrock model ID (optional, uses default)
        temperature: Model temperature
        **kwargs: Additional arguments (ignored)
    
    Returns:
        Model response content
    """
    if not STRANDS_AVAILABLE:
        raise RuntimeError("Strands is not available. Please install it to use chat functionality.")
    
    # Always ignore the model parameter and use Bedrock
    if model and model.startswith("gpt"):
        print(f"⚠️  Model '{model}' requested but using Bedrock instead (OpenAI disabled)")
    
    # Handle both message formats
    if isinstance(messages, list) and len(messages) > 0:
        # Extract content from message list
        if isinstance(messages[0], dict) and "content" in messages[0]:
            message = messages[0]["content"]
        else:
            message = str(messages[0])
    else:
        message = str(messages)
    
    # Always use Bedrock
    return chat_bedrock(message, model_id, temperature)


def tool_choice(messages, model="gpt-4o", temperature=0, tools=[], config={}):
    """
    OpenAI tool choice functionality is disabled.
    """
    raise RuntimeError("OpenAI functionality is disabled. Use Strands/Bedrock alternatives instead.")


def chat_bedrock(message: str, model_id: Optional[str] = None, temperature: float = 0.3) -> str:
    """
    Chat using Bedrock model via Strands
    
    Args:
        message: Input message
        model_id: Bedrock model ID (optional, uses default)
        temperature: Model temperature
    
    Returns:
        Model response as string
    """
    if not STRANDS_AVAILABLE:
        raise RuntimeError("Strands is not available. Please install it to use Bedrock chat.")
    
    if model_id and model_id != bedrock_model.model_id:
        # Create new model with different ID
        temp_model = BedrockModel(
            model_id=model_id,
            region_name=bedrock_model.region_name,
            temperature=temperature,
        )
        temp_agent = Agent(model=temp_model)
        result = temp_agent(message)
    else:
        result = strands_agent(message)
    
    # Extract text from AgentResult if needed
    if hasattr(result, 'text'):
        return result.text
    elif hasattr(result, 'content'):
        return result.content
    elif hasattr(result, '__str__'):
        return str(result)
    else:
        return result


def compare_embeddings(text1: str, text2: str, method: str = "bedrock") -> dict:
    """
    Compare two texts using embeddings and calculate similarity
    
    Args:
        text1: First text
        text2: Second text
        method: "bedrock" or "openai"
    
    Returns:
        Dictionary with embeddings and similarity score
    """
    import numpy as np
    
    # Get embeddings
    if method == "bedrock":
        emb1 = embed_bedrock(text1)
        emb2 = embed_bedrock(text2)
    else:
        emb1 = embed_openai(text1)
        emb2 = embed_openai(text2)
    
    # Calculate cosine similarity
    emb1_np = np.array(emb1)
    emb2_np = np.array(emb2)
    
    similarity = np.dot(emb1_np, emb2_np) / (np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np))
    
    return {
        "text1": text1,
        "text2": text2,
        "embedding1": emb1,
        "embedding2": emb2,
        "similarity": float(similarity),
        "method": method
    }


def test_all_connections():
    """Test all available connections and services"""
    results = {}
    
    # Test Neo4j
    results["neo4j"] = test_neo4j_connection()
    
    # Test Bedrock embedding
    try:
        test_embedding = bedrock_embedder.embed_text("Test embedding")
        results["bedrock_embedding"] = {
            "status": "connected",
            "message": f"Bedrock embedding successful, dimensions: {len(test_embedding)}"
        }
    except Exception as e:
        results["bedrock_embedding"] = {
            "status": "error",
            "message": f"Bedrock embedding failed: {str(e)}"
        }
    
    # Test Strands agent
    if STRANDS_AVAILABLE:
        try:
            test_response = strands_agent("Hello")
            results["strands_agent"] = {
                "status": "connected",
                "message": f"Strands agent successful, response length: {len(test_response)}"
            }
        except Exception as e:
            results["strands_agent"] = {
                "status": "error",
                "message": f"Strands agent failed: {str(e)}"
            }
    else:
        results["strands_agent"] = {
            "status": "unavailable",
            "message": "Strands not installed"
        }
    
    # OpenAI is disabled
    results["openai"] = {
        "status": "disabled",
        "message": "OpenAI functionality is disabled by design"
    }
    
    return results
