1. docker install
2. neo4j install
   
   docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  -v neo4j_data:/data \
  -v neo4j_logs:/logs \
  neo4j:latest

4. ec2 deploy in ap-northeast-2
5. setting ec2-instance-profile ( bedrock full access )
6. uv or conda install
7. setting python=3.11
8. pip install -r requirements.txt
9. start jupyterlab
10. execute rag_full.ipynb

