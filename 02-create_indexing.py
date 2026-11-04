from neo4j import GraphDatabase
from dotenv import load_dotenv
from neo4j_graphrag.indexes import (
    create_vector_index,
    upsert_vectors,
    create_fulltext_index,
)
from sentence_transformers import SentenceTransformer
import os
import re
load_dotenv()
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
import jieba
logger = logging.getLogger(__name__)

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
embedding_model = SentenceTransformer("./model/bge-base-zh")

# 给图谱中所有需要查询的节点，添加索引
# 后续需要使用混合检索：embedding和关键词索引

def clear_index(driver):
    """
    清除所有索引
    :return:
    """
    # 1.删除约束
    records1, _, _ = driver.execute_query("show constraints")
    for record in records1:
        driver.execute_query(f"DROP CONSTRAINT {record['name']} if exists")
    # 2.删除索引
    records2, _, _ = driver.execute_query("show indexes")
    for record in records2:
        if not record["owningConstraint"]:
            driver.execute_query(f"DROP INDEX {record['name']} if exists")

def vector_index(driver, label, property ):
    """
    添加向量索引
    :param label: 节点类型
    :param property: 节点属性
    :return:
    """
    # 1.创建向量索引
    create_vector_index(
        driver,
        f"{label.lower()}_vector",
        label=label,
        embedding_property="embedding",
        dimensions=768,
        similarity_fn="cosine"
    )


    # 2. 生成向量
    records, _, _ = driver.execute_query(f"""
        match (n:{label}) where n.embedding is null return elementId(n) as id, n.{property} as text
    """)

    if not records:
        logger.info(f"No records found for label {label} and property {property}")
        return
    record_tuple_list = [(record["id"], record["text"]) for record in records]
    all_ids, all_texts = zip(*record_tuple_list)
    all_ids = list(all_ids)
    all_texts = list(all_texts)
    vectors = embedding_model.encode(
        all_texts,
        batch_size=32,
        normalize_embeddings=True
    )

    # 3. 写入embedding属性中
    upsert_vectors(
        driver,
        ids=all_ids,
        embedding_property="embedding",
        embeddings=vectors
    )

def fulltext_index(driver, label, property):
    """
    添加全文索引
    :param label: 节点的类型
    :param property: 节点的属性
    :return:
    """
    # 1.创建向量索引
    create_fulltext_index(
        driver,
        name=f"{label.lower()}_fulltext",
        label=label,
        node_properties=["fulltext"]
    )

    records, _, _ = driver.execute_query(f"""
            match (n:{label}) where n.fulltext is null return elementId(n) as id, n.{property} as text
        """)

    if not records:
        logger.info(f"No records found for label {label} and property {property}")
        return

    record_tuple_list = [(record["id"], record["text"]) for record in records]
    pattern = re.compile(r"[a-zA-Z0-9\u4e00-\u9fa5]+")  # 匹配英文字母、数字和中文字符
    fulltext_tuple_list = [
        (
            id,
            " ".join([
                word.strip() for word in jieba.lcut(text) if pattern.fullmatch(word.strip())
            ])
        )
        for id, text in record_tuple_list
    ]


    all_ids, all_texts = zip(*fulltext_tuple_list)
    all_ids = list(all_ids)
    all_texts = list(all_texts)

    # 3. 写入全文索引中
    batch_size = 500
    for i in range(0, len(fulltext_tuple_list), batch_size):
        batch_rows = [
            {"id": id, "fulltext": fulltext} for id, fulltext in zip(all_ids[i:i + batch_size], all_texts[i:i + batch_size])
        ]

        driver.execute_query(
            f"""unwind $rows as row
                    match (n) where elementId(n) = row.id
                    set n.fulltext = row.fulltext""",
            {"rows": batch_rows}
        )


if __name__ == '__main__':
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        clear_index(driver)

        vector_index(driver, "SKU", "sku_name")
        vector_index(driver, "SPU", "spu_name")
        vector_index(driver, "Attr", "attr_value")
        vector_index(driver, "Category3", "category3_name")
        vector_index(driver, "Category2", "category2_name")
        vector_index(driver, "Category1", "category1_name")
        vector_index(driver, "Trademark", "trademark_name")

        fulltext_index(driver, "SKU", "sku_name")
        fulltext_index(driver, "SPU", "spu_name")
        fulltext_index(driver, "Attr", "attr_value")
        fulltext_index(driver, "Category3", "category3_name")
        fulltext_index(driver, "Category2", "category2_name")
        fulltext_index(driver, "Category1", "category1_name")
        fulltext_index(driver, "Trademark", "trademark_name")
        print("索引创建完成")
