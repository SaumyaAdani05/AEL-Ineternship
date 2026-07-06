@echo off
echo [*] Stopping existing neo4j container if running...
docker-compose down -v

echo [*] Starting neo4j in background...
docker-compose up -d

echo [*] Waiting 10 seconds for container to initialize...
timeout /t 10 /nobreak

echo [*] Running neo4j-admin import...
docker exec neo4j_attrition bin/neo4j-admin database import full --overwrite-destination --nodes=Employee=import/nodes.csv --relationships=SAME_MANAGER=import/edges_manager.csv --relationships=SAME_ROLE_DEPT=import/edges_role.csv --relationships=SAME_TENURE_COHORT=import/edges_tenure.csv neo4j

echo [*] Restarting neo4j to load the newly imported database...
docker-compose restart neo4j

echo [+] Neo4j Graph Database successfully initialized!
