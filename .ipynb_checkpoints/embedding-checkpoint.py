import json
import os
import boto3
from typing import List, Union
from botocore.exceptions import ClientError


class BedrockEmbedding:
    """Amazon Bedrock embedding client for Titan Embed Text v2"""
    
    def __init__(self, region_name: str = None):
        self.region_name = region_name or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self.model_id = "amazon.titan-embed-text-v2:0"
        
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.region_name
        )
    
    def embed_text(self, text: Union[str, List[str]], dimensions: int = 1024, normalize: bool = True):
        """Create embeddings using Amazon Titan Embed Text v2"""
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        embeddings = []
        
        for text_item in texts:
            try:
                body = {
                    "inputText": text_item,
                    "dimensions": dimensions,
                    "normalize": normalize
                }
                
                response = self.bedrock_client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body),
                    contentType='application/json',
                    accept='application/json'
                )
                
                response_body = json.loads(response['body'].read())
                embedding = response_body.get('embedding', [])
                embeddings.append(embedding)
                
            except ClientError as e:
                print(f"Error creating embedding: {e}")
                embeddings.append([0.0] * dimensions)
        
        if single_input:
            return embeddings[0]
        
        return embeddings


def create_embeddings(texts: Union[str, List[str]], region_name: str = None, **kwargs):
    """Convenience function to create embeddings"""
    embedder = BedrockEmbedding(region_name=region_name)
    return embedder.embed_text(texts, **kwargs)


def test_embedding():
    """Test the embedding functionality"""
    try:
        embedder = BedrockEmbedding()
        
        test_text = "This is a test sentence for embedding."
        embedding = embedder.embed_text(test_text)
        
        print(f"✅ Embedding successful")
        print(f"   Dimensions: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False


if __name__ == "__main__":
    test_embedding()