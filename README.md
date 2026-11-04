# GraphRAG + Neo4j 实践项目

## 一、项目介绍

本项目基于 Neo4j 图数据库，实现了知识图谱创建、节点关系查询、全文索引检索、关联关系扩展检索以及基于图谱事实的简单问答。

## 二、GraphRAG 基本原理

GraphRAG（Graph Retrieval-Augmented Generation，图检索增强生成）是在传统 RAG 基础上引入知识图谱的技术。

传统 RAG 主要通过向量相似度检索文本片段，而 GraphRAG 可以利用实体之间的关系进行多跳查询，获得更丰富的上下文信息。

## 三、GraphRAG 典型工作流程

1. **文档加载**：读取 PDF、Markdown、数据库等知识来源。
2. **实体关系抽取**：识别文本中的实体及实体之间的关系。
3. **构建知识图谱**：将实体、属性和关系写入 Neo4j。
4. **创建检索索引**：建立全文索引和向量索引，提高检索效率。
5. **知识检索**：根据用户问题进行关键词检索、向量检索或混合检索。
6. **关联关系扩展**：查询匹配节点周围的相关实体与关系。
7. **上下文构建**：将检索到的图谱信息整理成文本。
8. **答案生成**：将用户问题与图谱上下文交给大语言模型生成回答。

## 四、本项目已实现的功能

- 创建包含五个技术节点和五条关系的示例知识图谱。
- 使用 Cypher 查询节点与关系。
- 使用 Python Neo4j Driver 连接图数据库。
- 创建 Technology 节点的全文索引。
- 根据 GraphRAG 关键词检索相关节点。
- 扩展检索 GraphRAG 的关联节点。
- 将图谱事实整理为模板问答结果。

**实现范围说明：** 本项目目前是 GraphRAG 基础演示，尚未实现自动实体关系抽取、向量混合检索和大语言模型生成回答。

## 五、技术栈

- Python 3.11
- Neo4j 5.26 Community
- Neo4j Python Driver
- Docker
- Cypher
- python-dotenv

## 六、运行说明

1. 启动 Neo4j Docker 容器。
2. 配置 `.env` 中的 `NEO4J_URI`、`NEO4J_USERNAME`、`NEO4J_PASSWORD`。
3. 安装 `neo4j`、`python-dotenv`。
4. 创建示例节点和关系，并创建全文索引。
5. 运行 `test_connection.py` 验证连接。
6. 运行 `04-graph_retrieval.py` 查询图谱关系。
7. 运行 `05-graphrag_qa.py` 验证简单问答。

## 七、实验总结

本次实验熟悉了 Neo4j 图数据库的节点、关系与 Cypher 查询，理解了 GraphRAG 利用知识图谱进行关联知识检索的基本过程，并完成了基于全文索引和图关系扩展的简化问答实践。
