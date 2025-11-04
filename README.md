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

3. ec2 deploy in ap-northeast-2
4. setting ec2-instance-profile ( bedrock full access )
5. 
