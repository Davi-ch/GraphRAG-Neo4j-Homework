
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

def retrieve_graph():
    with GraphDatabase.driver(
        URI,
        auth=(USERNAME, PASSWORD)
    ) as driver:

        records, summary, keys = driver.execute_query(
            """
            MATCH (a:Technology)-[r]->(b:Technology)
            RETURN
                a.name AS source,
                type(r) AS relationship,
                b.name AS target
            ORDER BY source
            """,
            database_="neo4j"
        )

        print("===== GraphRAG 知识图谱关系检索 =====")
        print(f"共检索到 {len(records)} 条关系：\n")

        for record in records:
            print(
                f"{record['source']} "
                f"--[{record['relationship']}]--> "
                f"{record['target']}"
            )

if __name__ == "__main__":
    retrieve_graph()
