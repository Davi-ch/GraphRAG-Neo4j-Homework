# 掌握python如何操作neo4j
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
load_dotenv()

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
# 通过创建 Driver 对象并提供 URL 和身份验证令牌来连接到数据库
# 获取 Driver 实例后，使用 .verify_connectivity() 方法确保可以建立有效的连接
def get_movie(name:str):
    """查询该演员参演的所有电影
    :arg
    name:演员名字
    """
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        # cypher查询
        # 查询“刘德华”同志参演的所有电影
        cypher = "MATCH (:Person{name:$name})-[:ACTED_IN]-(m:Movie) return m.title as title, m.year as year, m.rating as rating"
        records, _, _ = driver.execute_query(
            query_=cypher,
            parameters_={"name": name},
        )

        # print(f"records: {records}")
        # print(f"summary: {summary}")
        # print(f"keys: {keys}")
        result = []
        for record in records:
            result.append((record["title"], record["year"], record["rating"]))
        return result

if __name__ == "__main__":
    name = "巩俐"
    movies = get_movie(name)
    print(f"{name}参演的所有电影：")
    for movie in movies:
        print(f"电影名：{movie[0]}, 上映年份：{movie[1]}, 评分：{movie[2]}")


