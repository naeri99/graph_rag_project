import json
import re
from typing import List, Dict, Any, Optional

# Import utilities from utils_strand
try:
    from utils_strand import (
        neo4j_driver, 
        chat, 
        chat_bedrock, 
        embed, 
        embed_bedrock, 
        embed_openai,
        test_neo4j_connection,
        test_all_connections,
        STRANDS_AVAILABLE,
        OPENAI_AVAILABLE,
        num_tokens_from_string,
        chunk_text
    )
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import from utils_strand: {e}")
    print("Some functionality may be limited.")
    UTILS_AVAILABLE = False
    neo4j_driver = None

GRAPH_EXTRACTION_PROMPT = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,PERSON
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
("entity"{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}ORGANIZATION{tuple_delimiter}The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday)
{record_delimiter}
("entity"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}PERSON{tuple_delimiter}Martin Smith is the chair of the Central Institution)
{record_delimiter}
("entity"{tuple_delimiter}MARKET STRATEGY COMMITTEE{tuple_delimiter}ORGANIZATION{tuple_delimiter}The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply)
{record_delimiter}
("relationship"{tuple_delimiter}MARTIN SMITH{tuple_delimiter}CENTRAL INSTITUTION{tuple_delimiter}Martin Smith is the Chair of the Central Institution and will answer questions at a press conference{tuple_delimiter}9)
{completion_delimiter}

######################
Example 2:
Entity_types: ORGANIZATION
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
("entity"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}ORGANIZATION{tuple_delimiter}TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones)
{record_delimiter}
("entity"{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}ORGANIZATION{tuple_delimiter}Vision Holdings is a firm that previously owned TechGlobal)
{record_delimiter}
("relationship"{tuple_delimiter}TECHGLOBAL{tuple_delimiter}VISION HOLDINGS{tuple_delimiter}Vision Holdings formerly owned TechGlobal from 2014 until present{tuple_delimiter}5)
{completion_delimiter}

######################
Example 3:
Entity_types: ORGANIZATION,GEO,PERSON
Text:
Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.

The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.

The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who are also Firuzi nationals, boarding a chartered flight to Krohaara.

They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality.
######################
Output:
("entity"{tuple_delimiter}FIRUZABAD{tuple_delimiter}GEO{tuple_delimiter}Firuzabad held Aurelians as hostages)
{record_delimiter}
("entity"{tuple_delimiter}AURELIA{tuple_delimiter}GEO{tuple_delimiter}Country seeking to release hostages)
{record_delimiter}
("entity"{tuple_delimiter}QUINTARA{tuple_delimiter}GEO{tuple_delimiter}Country that negotiated a swap of money in exchange for hostages)
{record_delimiter}
{record_delimiter}
("entity"{tuple_delimiter}TIRUZIA{tuple_delimiter}GEO{tuple_delimiter}Capital of Firuzabad where the Aurelians were being held)
{record_delimiter}
("entity"{tuple_delimiter}KROHAARA{tuple_delimiter}GEO{tuple_delimiter}Capital city in Quintara)
{record_delimiter}
("entity"{tuple_delimiter}CASHION{tuple_delimiter}GEO{tuple_delimiter}Capital city in Aurelia)
{record_delimiter}
("entity"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}PERSON{tuple_delimiter}Aurelian who spent time in Tiruzia's Alhamia Prison)
{record_delimiter}
("entity"{tuple_delimiter}ALHAMIA PRISON{tuple_delimiter}GEO{tuple_delimiter}Prison in Tiruzia)
{record_delimiter}
("entity"{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}PERSON{tuple_delimiter}Aurelian journalist who was held hostage)
{record_delimiter}
("entity"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}PERSON{tuple_delimiter}Bratinas national and environmentalist who was held hostage)
{record_delimiter}
("relationship"{tuple_delimiter}FIRUZABAD{tuple_delimiter}AURELIA{tuple_delimiter}Firuzabad negotiated a hostage exchange with Aurelia{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}QUINTARA{tuple_delimiter}AURELIA{tuple_delimiter}Quintara brokered the hostage exchange between Firuzabad and Aurelia{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}QUINTARA{tuple_delimiter}FIRUZABAD{tuple_delimiter}Quintara brokered the hostage exchange between Firuzabad and Aurelia{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}ALHAMIA PRISON{tuple_delimiter}Samuel Namara was a prisoner at Alhamia prison{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}Samuel Namara and Meggie Tazbah were exchanged in the same hostage release{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}Samuel Namara and Durke Bataglani were exchanged in the same hostage release{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}SAMUEL NAMARA{tuple_delimiter}FIRUZABAD{tuple_delimiter}Samuel Namara was a hostage in Firuzabad{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}MEGGIE TAZBAH{tuple_delimiter}FIRUZABAD{tuple_delimiter}Meggie Tazbah was a hostage in Firuzabad{tuple_delimiter}2)
{record_delimiter}
("relationship"{tuple_delimiter}DURKE BATAGLANI{tuple_delimiter}FIRUZABAD{tuple_delimiter}Durke Bataglani was a hostage in Firuzabad{tuple_delimiter}2)
{completion_delimiter}

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

def create_extraction_prompt(entity_types, input_text, tuple_delimiter=";"):
    prompt = GRAPH_EXTRACTION_PROMPT.format(
        entity_types=entity_types,
        input_text=input_text,
        tuple_delimiter=tuple_delimiter,
        record_delimiter="|",
        completion_delimiter="\n\n",
    )
    return prompt

def parse_extraction_output(output_str, record_delimiter=None, tuple_delimiter=None):
    """
    Parse a structured output string containing "entity" and "relationship" records into a list of dictionaries.

    The expected format for each record is:

        ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

    or

        ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

    Records are separated by a record delimiter. The output string may end with a completion marker
    (for example, "{completion_delimiter}") which will be removed.

    If not provided, this function attempts to auto-detect:
      - record_delimiter: looks for "{record_delimiter}" then "|" then falls back to newlines.
      - tuple_delimiter: looks for "{tuple_delimiter}" then ";" then falls back to a tab.

    Parameters:
        output_str: The complete string output or AgentResult object.
        record_delimiter (str, optional): The delimiter that separates records.
        tuple_delimiter (str, optional): The delimiter that separates fields within a record.

    Returns:
        List[dict]: A list of dictionaries where each dictionary represents an entity or relationship.

        For an "entity", the dictionary has the keys:
            - record_type (always "entity")
            - entity_name
            - entity_type
            - entity_description

        For a "relationship", the dictionary has the keys:
            - record_type (always "relationship")
            - source_entity
            - target_entity
            - relationship_description
            - relationship_strength (as an int or float)
    """
    # Convert AgentResult to string if needed
    if hasattr(output_str, 'text'):
        output_str = output_str.text
    elif hasattr(output_str, 'content'):
        output_str = output_str.content
    elif not isinstance(output_str, str):
        output_str = str(output_str)
    
    # Remove the completion delimiter if present.
    completion_marker = "{completion_delimiter}"
    if completion_marker in output_str:
        output_str = output_str.replace(completion_marker, "")
    output_str = output_str.strip()

    # Determine the record delimiter if not provided.
    if record_delimiter is None:
        if "{record_delimiter}" in output_str:
            record_delimiter = "{record_delimiter}"
        elif "|" in output_str:
            record_delimiter = "|"
        else:
            # Fallback: split on newlines
            record_delimiter = "\n"

    # Determine the tuple delimiter if not provided.
    if tuple_delimiter is None:
        if "{tuple_delimiter}" in output_str:
            tuple_delimiter = "{tuple_delimiter}"
        elif ";" in output_str:
            tuple_delimiter = ";"
        else:
            tuple_delimiter = "\t"

    # Split the output into individual record strings.
    raw_records = [r.strip() for r in output_str.split(record_delimiter)]

    parsed_records = []
    for rec in raw_records:
        if not rec:
            continue  # skip empty strings

        # Remove leading/trailing parentheses if present.
        if rec.startswith("(") and rec.endswith(")"):
            rec = rec[1:-1]
        rec = rec.strip()

        # Split the record into tokens using the tuple delimiter.
        tokens = [token.strip() for token in rec.split(tuple_delimiter)]
        if not tokens:
            continue

        # The first token should be either "entity" or "relationship".
        rec_type = tokens[0].strip(' "\'').lower()

        if rec_type == "entity":
            if len(tokens) != 4:
                # Optionally log or raise an error for malformed records.
                continue
            record = {
                "record_type": "entity",
                "entity_name": tokens[1],
                "entity_type": tokens[2],
                "entity_description": tokens[3]
            }
            parsed_records.append(record)
        elif rec_type == "relationship":
            if len(tokens) != 5:
                continue
            # Attempt to convert relationship_strength to a number.
            try:
                strength = float(tokens[4])
                # Convert to int if it has no fractional part.
                if strength.is_integer():
                    strength = int(strength)
            except ValueError:
                strength = tokens[4]
            record = {
                "record_type": "relationship",
                "source_entity": tokens[1],
                "target_entity": tokens[2],
                "relationship_description": tokens[3],
                "relationship_strength": strength
            }
            parsed_records.append(record)
        else:
            # Unknown record type; skip it or handle accordingly.
            continue
    nodes = [el for el in parsed_records if el.get("record_type") == "entity"]
    relationships = [el for el in parsed_records if el.get("record_type") == "relationship"]
    return nodes, relationships

import_nodes_query = """
MERGE (b:Book {id: $book_id})
MERGE (b)-[:HAS_CHUNK]->(c:__Chunk__ {id: $chunk_id})
SET c.text = $text
WITH c
UNWIND $data AS row
MERGE (n:__Entity__ {name: row.entity_name})
SET n:$(row.entity_type),
    n.description = coalesce(n.description, []) + [row.entity_description]
MERGE (n)<-[:MENTIONS]-(c)
"""

import_relationships_query = """
UNWIND $data AS row
MERGE (s:__Entity__ {name: row.source_entity})
MERGE (t:__Entity__ {name: row.target_entity})
CREATE (s)-[r:RELATIONSHIP {description: row.relationship_description, strength: row.relationship_strength}]->(t)
"""

SUMMARIZE_PROMPT = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

def get_summarize_prompt(entity_name, description_list):
    return SUMMARIZE_PROMPT.format(
        entity_name=entity_name,
        description_list=description_list)

def calculate_communities(driver=None):
    """Calculate communities using simple connected components algorithm (GDS-free)"""
    if driver is None:
        if not UTILS_AVAILABLE or neo4j_driver is None:
            raise RuntimeError("Neo4j driver not available. Please check utils_strand connection.")
        driver = neo4j_driver
    
    print("🔍 Calculating communities without GDS...")
    
    # 1. 모든 엔티티의 연결 정보 가져오기
    entities_result = driver.execute_query("""
    MATCH (e:__Entity__)
    OPTIONAL MATCH (e)-[r:RELATIONSHIP]-(connected:__Entity__)
    RETURN e.name as entity, collect(DISTINCT connected.name) as connections
    """)
    
    # 2. Python에서 연결 컴포넌트 계산
    entities_data = {}
    all_entities = set()
    
    for record in entities_result[0]:
        entity = record["entity"]
        connections = [c for c in record["connections"] if c is not None]
        entities_data[entity] = connections
        all_entities.add(entity)
        all_entities.update(connections)
    
    # 3. Union-Find 알고리즘으로 연결 컴포넌트 찾기
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # 모든 연결 처리
    for entity, connections in entities_data.items():
        for connected in connections:
            union(entity, connected)
    
    # 4. 커뮤니티 그룹핑
    communities = {}
    for entity in all_entities:
        root = find(entity)
        if root not in communities:
            communities[root] = []
        communities[root].append(entity)
    
    # 5. 커뮤니티 ID 할당 (크기 순으로 정렬)
    community_list = sorted(communities.values(), key=len, reverse=True)
    community_assignment = {}
    
    for i, community in enumerate(community_list):
        for entity in community:
            community_assignment[entity] = i
    
    # 6. Neo4j에 커뮤니티 정보 저장
    update_data = [{"entity": entity, "community": comm_id} 
                   for entity, comm_id in community_assignment.items()]
    
    driver.execute_query("""
    UNWIND $data AS row
    MATCH (e:__Entity__ {name: row.entity})
    SET e.louvain = row.community
    """, data=update_data)
    
    # 7. 통계 계산
    community_sizes = [len(comm) for comm in community_list]
    community_distribution = {}
    for size in community_sizes:
        community_distribution[size] = community_distribution.get(size, 0) + 1
    
    result = {
        "communityCount": len(community_list),
        "communityDistribution": community_distribution,
        "nodeCount": len(all_entities),
        "relationshipCount": sum(len(connections) for connections in entities_data.values()) // 2,
        "largest_community_size": max(community_sizes) if community_sizes else 0,
        "smallest_community_size": min(community_sizes) if community_sizes else 0
    }
    
    print(f"✅ Found {result['communityCount']} communities")
    print(f"   - Largest community: {result['largest_community_size']} nodes")
    print(f"   - Total nodes: {result['nodeCount']}")
    print(f"   - Total relationships: {result['relationshipCount']}")
    
    return result

COMMUNITY_REPORT_PROMPT = """
You are an AI assistant that helps a human analyst to perform general information discovery. Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
        ]
    }}

# Grounding Rules

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."

where 1, 5, 7, 23, 2, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


# Example Input
-----------
Text:

Entities

id,entity,description
5,VERDANT OASIS PLAZA,Verdant Oasis Plaza is the location of the Unity March
6,HARMONY ASSEMBLY,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza

Relationships

id,source,target,description
37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza
39,VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza
40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi is speaking at Verdant Oasis Plaza about the march
43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March

Output:
{{
    "title": "Verdant Oasis Plaza and Unity March",
    "summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of which are associated with the march event.",
    "rating": 5.0,
    "rating_explanation": "The impact severity rating is moderate due to the potential for unrest or conflict during the Unity March.",
    "findings": [
        {{
            "summary": "Verdant Oasis Plaza as the central location",
            "explanation": "Verdant Oasis Plaza is the central entity in this community, serving as the location for the Unity March. This plaza is the common link between all other entities, suggesting its significance in the community. The plaza's association with the march could potentially lead to issues such as public disorder or conflict, depending on the nature of the march and the reactions it provokes. [Data: Entities (5), Relationships (37, 38, 39, 40, 41,+more)]"
        }},
        {{
            "summary": "Harmony Assembly's role in the community",
            "explanation": "Harmony Assembly is another key entity in this community, being the organizer of the march at Verdant Oasis Plaza. The nature of Harmony Assembly and its march could be a potential source of threat, depending on their objectives and the reactions they provoke. The relationship between Harmony Assembly and the plaza is crucial in understanding the dynamics of this community. [Data: Entities(6), Relationships (38, 43)]"
        }},
        {{
            "summary": "Unity March as a significant event",
            "explanation": "The Unity March is a significant event taking place at Verdant Oasis Plaza. This event is a key factor in the community's dynamics and could be a potential source of threat, depending on the nature of the march and the reactions it provokes. The relationship between the march and the plaza is crucial in understanding the dynamics of this community. [Data: Relationships (39)]"
        }},
        {{
            "summary": "Role of Tribune Spotlight",
            "explanation": "Tribune Spotlight is reporting on the Unity March taking place in Verdant Oasis Plaza. This suggests that the event has attracted media attention, which could amplify its impact on the community. The role of Tribune Spotlight could be significant in shaping public perception of the event and the entities involved. [Data: Relationships (40)]"
        }}
    ]
}}


# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:
{input_text}

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
        ]
    }}

# Grounding Rules

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."

where 1, 5, 7, 23, 2, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.

Output:"""

def get_summarize_community_prompt(nodes, relationships):
    """커뮤니티 요약을 위한 개선된 프롬프트"""
    
    # 노드 정보를 더 안전하게 처리
    node_info = []
    for i, node in enumerate(nodes[:10]):  # 최대 10개만
        name = node.get('name', f'Unknown_{i}')
        node_type = node.get('type', 'Entity')
        description = node.get('description', 'No description available')
        node_info.append(f"{i+1},{name},{description}")
    
    # 관계 정보를 더 안전하게 처리
    rel_info = []
    for i, rel in enumerate(relationships[:10]):  # 최대 10개만
        source = rel.get('source', 'Unknown')
        target = rel.get('target', 'Unknown')
        rel_type = rel.get('type', 'RELATED')
        description = rel.get('description', f'{source} is related to {target}')
        rel_info.append(f"{i+1},{source},{target},{description}")
    
    input_text = f"""Entities

id,entity,description
{chr(10).join(node_info)}

Relationships

id,source,target,description
{chr(10).join(rel_info)}
"""
    
    return f"""
다음 노드들과 관계들로 구성된 커뮤니티를 분석하고 요약해주세요.

{input_text}

**요구사항:**
1. 반드시 유효한 JSON 형식으로만 응답하세요
2. 다른 설명이나 텍스트는 포함하지 마세요
3. 아래 형식을 정확히 따라주세요

응답 형식:
{{
    "title": "커뮤니티의 간단한 제목",
    "summary": "커뮤니티에 대한 2-3문장 설명",
    "rating": 5.0,
    "rating_explanation": "평점에 대한 한 문장 설명",
    "findings": [
        {{
            "summary": "주요 발견사항 1",
            "explanation": "상세 설명"
        }},
        {{
            "summary": "주요 발견사항 2", 
            "explanation": "상세 설명"
        }}
    ]
}}

JSON만 응답하세요:
"""

def extract_json(text):
    """LLM 응답에서 JSON을 안전하게 추출"""
    # Convert AgentResult to string if needed
    if hasattr(text, 'text'):
        text = text.text
    elif hasattr(text, 'content'):
        text = text.content
    elif not isinstance(text, str):
        text = str(text)
    
    try:
        # 1. 직접 JSON 파싱 시도
        cleaned = text.removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    try:
        # 2. ```json 블록에서 추출
        json_match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1).strip())
    except json.JSONDecodeError:
        pass
    
    try:
        # 3. 첫 번째 { }로 둘러싸인 부분 추출
        start = text.find('{')
        if start != -1:
            brace_count = 0
            end = start
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            json_str = text[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    try:
        # 4. [ ]로 둘러싸인 배열 추출
        start = text.find('[')
        if start != -1:
            bracket_count = 0
            end = start
            for i, char in enumerate(text[start:], start):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end = i + 1
                        break
            
            json_str = text[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # 5. 모든 방법 실패시 기본값 반환
    print(f"JSON 추출 실패. 원본 텍스트: {text[:200]}...")
    return {
        "title": "Unknown Community",
        "summary": "Failed to parse community summary",
        "rating": 5.0,
        "rating_explanation": "Default rating due to parsing failure",
        "findings": []
    }

# import_community_query = """
# UNWIND $data AS row
# MERGE (c:__Community__ {communityId: row.communityId})
# SET c.title = row.community.title,
#     c.summary = row.community.summary,
#     c.rating = row.community.rating,
#     c.rating_explanation = row.community.rating_explanation
# WITH c, row
# UNWIND row.nodes AS node
# MERGE (n:__Entity__ {name: node})
# MERGE (n)-[:IN_COMMUNITY]->(c)
# """

import_community_query = """
UNWIND $data AS row
MERGE (c:__Community__ {communityId: row.communityId})
SET c.title = row.community.title,
    c.summary = row.community.summary,
    c.rating = row.community.rating,
    c.rating_explanation = row.community.rating_explanation
WITH c, row
UNWIND row.nodes AS node
MERGE (n:__Entity__ {name: node})
MERGE (n)-[:IN_COMMUNITY]->(c)
"""

def import_entity_summary(entity_information, driver=None):
    """Import entity summaries to Neo4j"""
    if driver is None:
        if not UTILS_AVAILABLE or neo4j_driver is None:
            raise RuntimeError("Neo4j driver not available. Please check utils_strand connection.")
        driver = neo4j_driver
    
    driver.execute_query("""
    UNWIND $data AS row
    MATCH (e:__Entity__ {name: row.entity})
    SET e.summary = row.summary
    """, data=entity_information)
    
    # If there was only 1 description use that
    driver.execute_query("""
    MATCH (e:__Entity__)
    WHERE size(e.description) = 1
    SET e.summary = e.description[0]
    """)

def import_rels_summary(rel_summaries, driver=None):
    """Import relationship summaries to Neo4j"""
    if driver is None:
        if not UTILS_AVAILABLE or neo4j_driver is None:
            raise RuntimeError("Neo4j driver not available. Please check utils_strand connection.")
        driver = neo4j_driver
    
    driver.execute_query("""
    UNWIND $data AS row
    MATCH (s:__Entity__ {name: row.source}), (t:__Entity__ {name: row.target})
    MERGE (s)-[r:SUMMARIZED_RELATIONSHIP]-(t)
    SET r.summary = row.summary
    """, data=rel_summaries)
    
    # If there was only 1 description use that
    driver.execute_query("""
    MATCH (s:__Entity__)-[e:RELATIONSHIP]-(t:__Entity__)
    WHERE NOT (s)-[:SUMMARIZED_RELATIONSHIP]-(t)
    MERGE (s)-[r:SUMMARIZED_RELATIONSHIP]-(t)
    SET r.summary = e.description
    """)

community_info_query = """MATCH (e:__Entity__)
WHERE e.louvain IS NOT NULL
WITH e.louvain AS louvain, collect(e) AS nodes
WHERE size(nodes) > 1
WITH louvain, nodes,
     [n in nodes | {id: n.name, description: n.summary, type: [el in labels(n) WHERE el <> '__Entity__'][0]}] AS nodeData
OPTIONAL MATCH (source:__Entity__)-[r:RELATIONSHIP]-(target:__Entity__)
WHERE source IN nodes AND target IN nodes
WITH louvain, nodeData,
     collect(DISTINCT {start: source.name, type: type(r), end: target.name, description: r.description}) AS rels
RETURN louvain AS communityId, nodeData AS nodes, rels"""


def extract_entities_with_llm(text: str, entity_types: str, model: str = "bedrock") -> tuple:
    """
    Extract entities and relationships from text using LLM
    
    Args:
        text: Input text to process
        entity_types: Comma-separated entity types
        model: "bedrock", "openai", or specific model name
    
    Returns:
        Tuple of (entities, relationships)
    """
    if not UTILS_AVAILABLE:
        raise RuntimeError("utils_strand not available. Cannot use LLM functionality.")
    
    prompt = create_extraction_prompt(entity_types, text)
    
    try:
        if STRANDS_AVAILABLE:
            response = chat_bedrock(prompt)
            
            # Ensure response is a string
            if hasattr(response, 'text'):
                response = response.text
            elif hasattr(response, 'content'):
                response = response.content
            elif not isinstance(response, str):
                response = str(response)
        else:
            raise RuntimeError("Strands not available. Cannot perform entity extraction.")
        
        entities, relationships = parse_extraction_output(response)
        return entities, relationships
        
    except Exception as e:
        print(f"Error in entity extraction: {e}")
        print(f"Response type: {type(response) if 'response' in locals() else 'Unknown'}")
        return [], []


def generate_community_report_with_llm(nodes: str, relationships: str, model: str = "bedrock") -> dict:
    """
    Generate community report using LLM
    
    Args:
        nodes: Node information as string
        relationships: Relationship information as string
        model: "bedrock", "openai", or specific model name
    
    Returns:
        Dictionary containing the community report
    """
    if not UTILS_AVAILABLE:
        raise RuntimeError("utils_strand not available. Cannot use LLM functionality.")
    
    prompt = get_summarize_community_prompt(nodes, relationships)
    
    try:
        if STRANDS_AVAILABLE:
            response = chat_bedrock(prompt)
            
            # Ensure response is a string
            if hasattr(response, 'text'):
                response = response.text
            elif hasattr(response, 'content'):
                response = response.content
            elif not isinstance(response, str):
                response = str(response)
        else:
            raise RuntimeError("Strands not available. Cannot generate community report.")
        
        # Extract JSON from response
        json_str = extract_json(response)
        return json.loads(json_str)
        
    except Exception as e:
        print(f"Error in community report generation: {e}")
        print(f"Response type: {type(response) if 'response' in locals() else 'Unknown'}")
        return {}


def import_nodes_and_relationships(book_id: str, chunk_id: str, text: str, 
                                 entities: List[Dict], relationships: List[Dict], 
                                 driver=None):
    """
    Import nodes and relationships to Neo4j
    
    Args:
        book_id: Book identifier
        chunk_id: Chunk identifier  
        text: Original text
        entities: List of entity dictionaries
        relationships: List of relationship dictionaries
        driver: Neo4j driver (optional)
    """
    if driver is None:
        if not UTILS_AVAILABLE or neo4j_driver is None:
            raise RuntimeError("Neo4j driver not available. Please check utils_strand connection.")
        driver = neo4j_driver
    
    # Import nodes
    driver.execute_query(
        import_nodes_query,
        book_id=book_id,
        chunk_id=chunk_id,
        text=text,
        data=entities
    )
    
    # Import relationships
    if relationships:
        driver.execute_query(
            import_relationships_query,
            data=relationships
        )


def get_community_info(driver=None) -> List[Dict]:
    """
    Get community information from Neo4j
    
    Args:
        driver: Neo4j driver (optional)
    
    Returns:
        List of community information dictionaries
    """
    if driver is None:
        if not UTILS_AVAILABLE or neo4j_driver is None:
            raise RuntimeError("Neo4j driver not available. Please check utils_strand connection.")
        driver = neo4j_driver
    
    records, _, _ = driver.execute_query(community_info_query)
    return [record.data() for record in records]


def test_ch07_tools_connectivity():
    """Test connectivity of ch07_tools with utils_strand"""
    results = {}
    
    # Test utils_strand availability
    results["utils_strand"] = {"available": UTILS_AVAILABLE}
    
    if UTILS_AVAILABLE:
        # Test Neo4j connection
        try:
            connection_test = test_neo4j_connection()
            results["neo4j"] = connection_test
        except Exception as e:
            results["neo4j"] = {"status": "error", "message": str(e)}
        
        # Test LLM availability
        results["strands"] = {"available": STRANDS_AVAILABLE}
        
        # Test basic functionality
        try:
            # Test entity extraction with Bedrock
            test_entities, test_rels = extract_entities_with_llm(
                "Apple Inc. is a technology company founded by Steve Jobs.", 
                "ORGANIZATION,PERSON",
                model="bedrock"
            )
            results["entity_extraction"] = {
                "status": "success",
                "entities_found": len(test_entities),
                "relationships_found": len(test_rels)
            }
        except Exception as e:
            results["entity_extraction"] = {"status": "error", "message": str(e)}
    
    return results

MAP_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}},
        {{"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}}
    ]
}}

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Points supported by data should list the relevant reports as references as follows:
"This is an example sentence supported by data references [Data: Reports (report ids)]"

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in the provided tables.

Do not include information where the supporting evidence for it is not provided.


---Data tables---

{context_data}

---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Points supported by data should list the relevant reports as references as follows:
"This is an example sentence supported by data references [Data: Reports (report ids)]"

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in the provided tables.

Do not include information where the supporting evidence for it is not provided.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}},
        {{"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}}
    ]
}}
"""

REDUCE_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Analyst Reports---

{report_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

def get_map_system_prompt(context):
    return MAP_SYSTEM_PROMPT.format(context_data=context)

def get_reduce_system_prompt(report_data, response_type: str = "multiple paragraphs"):
    return REDUCE_SYSTEM_PROMPT.format(report_data=report_data, response_type=response_type)

LOCAL_SEARCH_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (15, 16), Reports (1), Entities (5, 7); Relationships (23); Claims (2, 7, 34, 46, 64, +more)]."

where 15, 16, 1, 5, 7, 23, 2, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Data tables---

{context_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Sources (15, 16), Reports (1), Entities (5, 7); Relationships (23); Claims (2, 7, 34, 46, 64, +more)]."

where 15, 16, 1, 5, 7, 23, 2, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

def get_local_system_prompt(report_data, response_type: str = "multiple paragraphs"):
    return LOCAL_SEARCH_SYSTEM_PROMPT.format(context_data=report_data, response_type=response_type)
# =============================================================================
# EXPORTED FUNCTIONS AND VARIABLES
# =============================================================================
# Re-export commonly used functions from utils_strand for convenience

# Database connection
__all__ = [
    # Neo4j connection
    'neo4j_driver',
    
    # Text processing utilities
    'num_tokens_from_string',
    'chunk_text',
    
    # LLM functions
    'chat',
    'chat_bedrock',
    'embed',
    'embed_bedrock', 
    'embed_openai',
    
    # Graph extraction and processing
    'create_extraction_prompt',
    'parse_extraction_output',
    'extract_entities_with_llm',
    'extract_entities',
    'process_book_chunks',
    'bedrock_only_pipeline',
    
    # Community analysis
    'calculate_communities',
    'get_summarize_community_prompt',
    'generate_community_report_with_llm',
    'get_community_info',
    
    # Data import functions
    'import_nodes_and_relationships',
    'import_entity_summary',
    'import_rels_summary',
    
    # Utility functions
    'extract_json',
    'get_summarize_prompt',
    'test_ch07_tools_connectivity',
    'test_neo4j_connection',
    'test_all_connections',
    
    # Constants and queries
    'import_nodes_query',
    'import_relationships_query',
    'import_community_query',
    'community_info_query',
    
    # Availability flags
    'UTILS_AVAILABLE',
    'STRANDS_AVAILABLE',
    'OPENAI_AVAILABLE'
]

# Convenience aliases for backward compatibility
if UTILS_AVAILABLE:
    # Make sure all functions are available at module level
    globals().update({
        'neo4j_driver': neo4j_driver,
        'num_tokens_from_string': num_tokens_from_string,
        'chunk_text': chunk_text,
        'chat': chat,
        'embed': embed,
        'chat_bedrock': chat_bedrock if STRANDS_AVAILABLE else None,
        'embed_bedrock': embed_bedrock,
        'embed_openai': embed_openai,
        'test_neo4j_connection': test_neo4j_connection,
        'test_all_connections': test_all_connections,
    })
else:
    # Provide fallback functions if utils_strand is not available
    def _not_available(*args, **kwargs):
        raise RuntimeError("utils_strand not available. Please check your installation.")
    
    neo4j_driver = None
    num_tokens_from_string = _not_available
    chunk_text = _not_available
    chat = _not_available
    embed = _not_available
    chat_bedrock = _not_available
    embed_bedrock = _not_available
    embed_openai = _not_available
    test_neo4j_connection = _not_available
    test_all_connections = _not_available

def extract_entities(text: str, entity_types: str = "ORGANIZATION,PERSON,LOCATION,EVENT") -> tuple:
    """
    Extract entities and relationships from text using Bedrock
    
    Args:
        text: Input text to process
        entity_types: Comma-separated entity types
    
    Returns:
        Tuple of (entities, relationships)
    """
    return extract_entities_with_llm(text, entity_types, model="bedrock")


def process_book_chunks(chunked_books: List[List[str]], number_of_books: int = 1, 
                       entity_types: str = "ORGANIZATION,PERSON,LOCATION,EVENT"):
    """
    Process book chunks and import to Neo4j using Bedrock
    
    Args:
        chunked_books: List of books, each containing list of text chunks
        number_of_books: Number of books to process
        entity_types: Entity types to extract
    """
    if not UTILS_AVAILABLE or neo4j_driver is None:
        raise RuntimeError("Neo4j driver not available. Please check utils_strand connection.")
    
    for book_i, book in enumerate(tqdm(chunked_books[:number_of_books], desc="Processing Books")):
        for chunk_i, chunk in enumerate(tqdm(book, desc=f"Book {book_i}", leave=False)):
            try:
                # Extract entities using Bedrock
                nodes, relationships = extract_entities(chunk, entity_types)
                
                # Import nodes
                neo4j_driver.execute_query(
                    import_nodes_query,
                    data=nodes,
                    book_id=book_i,
                    text=chunk,
                    chunk_id=chunk_i,
                )
                
                # Import relationships
                if relationships:
                    neo4j_driver.execute_query(
                        import_relationships_query, 
                        data=relationships
                    )
                
                print(f"Processed book {book_i}, chunk {chunk_i}: {len(nodes)} entities, {len(relationships)} relationships")
                
            except Exception as e:
                print(f"Error processing book {book_i}, chunk {chunk_i}: {e}")
                continue


def bedrock_only_pipeline(chunked_books: List[List[str]], number_of_books: int = 1):
    """
    Complete Bedrock-only processing pipeline
    
    Args:
        chunked_books: List of books with chunks
        number_of_books: Number of books to process
    """
    print("🚀 Starting Bedrock-only Knowledge Graph Pipeline")
    
    # Test connections first
    print("📋 Testing connections...")
    results = test_all_connections()
    
    if results.get("neo4j", {}).get("status") != "connected":
        raise RuntimeError("Neo4j connection failed. Please check your connection.")
    
    if results.get("bedrock_embedding", {}).get("status") != "connected":
        raise RuntimeError("Bedrock embedding failed. Please check your AWS credentials.")
    
    if not STRANDS_AVAILABLE:
        raise RuntimeError("Strands not available. Please install strands package.")
    
    print("✅ All connections successful")
    
    # Process books
    print(f"📚 Processing {number_of_books} books...")
    process_book_chunks(chunked_books, number_of_books)
    
    # Calculate communities
    print("🔗 Calculating communities...")
    community_stats = calculate_communities()
    print(f"Community calculation completed: {community_stats}")
    
    # Get community info
    print("📊 Getting community information...")
    communities = get_community_info()
    print(f"Found {len(communities)} communities")
    
    print("🎉 Pipeline completed successfully!")
    return {
        "community_stats": community_stats,
        "communities": communities,
        "books_processed": number_of_books
    }


def clear_all_graph_data(driver=None):
    """
    Neo4j 그래프의 모든 데이터를 지우는 함수
    모든 노드, 관계, 인덱스, 제약조건을 삭제합니다.
    
    Args:
        driver: Neo4j driver instance (optional, uses global neo4j_driver if not provided)
    
    Returns:
        dict: 삭제된 데이터의 통계 정보
    """
    if driver is None:
        if not UTILS_AVAILABLE or neo4j_driver is None:
            raise RuntimeError("Neo4j driver not available. Please check utils_strand connection.")
        driver = neo4j_driver
    
    print("🗑️  Starting to clear all graph data...")
    
    # 1. 모든 관계 삭제
    print("   Deleting all relationships...")
    result = driver.execute_query("MATCH ()-[r]-() DELETE r RETURN count(r) as deleted_relationships")
    deleted_rels = result[0][0]["deleted_relationships"] if result[0] else 0
    
    # 2. 모든 노드 삭제
    print("   Deleting all nodes...")
    result = driver.execute_query("MATCH (n) DELETE n RETURN count(n) as deleted_nodes")
    deleted_nodes = result[0][0]["deleted_nodes"] if result[0] else 0
    
    # 3. GDS 그래프 삭제 (있다면)
    print("   Cleaning up GDS graphs...")
    try:
        driver.execute_query("CALL gds.graph.drop('entity')")
        print("   - Dropped 'entity' graph")
    except Exception:
        print("   - No 'entity' graph to drop")
    
    # 4. 모든 인덱스 삭제
    print("   Dropping all indexes...")
    indexes_result = driver.execute_query("SHOW INDEXES")
    for record in indexes_result[0]:
        index_name = record.get("name")
        if index_name:
            try:
                driver.execute_query(f"DROP INDEX {index_name}")
                print(f"   - Dropped index: {index_name}")
            except Exception as e:
                print(f"   - Failed to drop index {index_name}: {e}")
    
    # 5. 모든 제약조건 삭제
    print("   Dropping all constraints...")
    constraints_result = driver.execute_query("SHOW CONSTRAINTS")
    for record in constraints_result[0]:
        constraint_name = record.get("name")
        if constraint_name:
            try:
                driver.execute_query(f"DROP CONSTRAINT {constraint_name}")
                print(f"   - Dropped constraint: {constraint_name}")
            except Exception as e:
                print(f"   - Failed to drop constraint {constraint_name}: {e}")
    
    # 6. 최종 확인
    final_check = driver.execute_query("MATCH (n) RETURN count(n) as remaining_nodes")
    remaining_nodes = final_check[0][0]["remaining_nodes"] if final_check[0] else 0
    
    stats = {
        "deleted_relationships": deleted_rels,
        "deleted_nodes": deleted_nodes,
        "remaining_nodes": remaining_nodes,
        "status": "success" if remaining_nodes == 0 else "warning"
    }
    
    print(f"✅ Graph cleanup completed!")
    print(f"   - Deleted {deleted_rels} relationships")
    print(f"   - Deleted {deleted_nodes} nodes")
    print(f"   - Remaining nodes: {remaining_nodes}")
    
    if remaining_nodes > 0:
        print("⚠️  Warning: Some nodes may still remain. You might need to check for system nodes.")
    
    return stats


def clear_specific_labels(labels_to_clear=None, driver=None):
    """
    특정 라벨의 노드들만 삭제하는 함수
    
    Args:
        labels_to_clear: 삭제할 라벨 리스트 (기본값: ['__Entity__', '__Chunk__', '__Community__', 'Book'])
        driver: Neo4j driver instance
    
    Returns:
        dict: 삭제된 데이터의 통계 정보
    """
    if driver is None:
        if not UTILS_AVAILABLE or neo4j_driver is None:
            raise RuntimeError("Neo4j driver not available. Please check utils_strand connection.")
        driver = neo4j_driver
    
    if labels_to_clear is None:
        labels_to_clear = ['__Entity__', '__Chunk__', '__Community__', 'Book']
    
    print(f"🗑️  Clearing nodes with labels: {labels_to_clear}")
    
    total_deleted = 0
    
    for label in labels_to_clear:
        print(f"   Deleting nodes with label '{label}'...")
        
        # 해당 라벨의 노드와 연결된 모든 관계도 함께 삭제
        result = driver.execute_query(f"""
        MATCH (n:{label})
        DETACH DELETE n
        RETURN count(n) as deleted_count
        """)
        
        deleted_count = result[0][0]["deleted_count"] if result[0] else 0
        total_deleted += deleted_count
        print(f"   - Deleted {deleted_count} nodes with label '{label}'")
    
    # GDS 그래프 정리
    try:
        driver.execute_query("CALL gds.graph.drop('entity')")
        print("   - Cleaned up GDS 'entity' graph")
    except Exception:
        pass
    
    stats = {
        "labels_cleared": labels_to_clear,
        "total_deleted_nodes": total_deleted,
        "status": "success"
    }
    
    print(f"✅ Selective cleanup completed! Deleted {total_deleted} nodes total.")
    
    return stats