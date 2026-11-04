
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# 1. 读取 .env 环境变量
load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

# 2. 检查必要的配置
if not all([URI, USERNAME, PASSWORD]):
    raise ValueError("Neo4j 配置不完整，请检查 .env 文件")

# 3. 连接 Neo4j
try:
    with GraphDatabase.driver(
        URI,
        auth=(USERNAME, PASSWORD)
    ) as driver:

        # 验证数据库连接
        driver.verify_connectivity()
        print("Neo4j 数据库连接成功！")

        # 4. 查询之前创建的知识图谱
        records, summary, keys = driver.execute_query(
            """
            MATCH (n:Technology)
            RETURN n.name AS name
            ORDER BY name
            """,
            database_="neo4j"
        )

        print(f"查询成功！共找到 {len(records)} 个技术节点：")

        for record in records:
            print(record["name"])

except Exception as e:
    print(f"Neo4j 连接或查询失败：{e}")
