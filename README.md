1. ec2 deploy in ap-northeast-2
2. docker install
3. neo4j install
   
   docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  -v neo4j_data:/data \
  -v neo4j_logs:/logs \
  neo4j:latest

4. setting ec2-instance-profile ( bedrock full access )
5. uv or conda install (python venv setting python=3.11)
6. activate venv 
7. pip install -r requirements.txt
8. start jupyterlab
9. execute rag_full.ipynb

