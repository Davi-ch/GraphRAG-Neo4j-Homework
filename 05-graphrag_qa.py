
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")


def graphrag_qa(keyword):
    with GraphDatabase.driver(
        URI,
        auth=(USERNAME, PASSWORD)
    ) as driver:

        records, _, _ = driver.execute_query(
            """
            CALL db.index.fulltext.queryNodes(
                "technology_fulltext", $keyword
            )
            YIELD node, score

            MATCH (node)-[r]-(related:Technology)

            RETURN
                node.name AS source,
                type(r) AS relation,
                related.name AS target
            LIMIT 10
            """,
            keyword=keyword,
            database_="neo4j"
        )

        if not records:
            return "知识图谱中暂未找到相关信息。"

        context = []
        for record in records:
            context.append(
                f"{record['source']} "
                f"--[{record['relation']}]--> "
                f"{record['target']}"
            )

        answer = f"根据知识图谱，关于 {keyword} 的信息如下：\n"
        answer += "\n".join(context)

        return answer


if __name__ == "__main__":
    question = "GraphRAG"
    answer = graphrag_qa(question)
    print("用户问题：GraphRAG 有哪些相关技术？")
    print("\n系统回答：")
    print(answer)
