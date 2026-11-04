# 根据用户的原始问题-》生成Cypher语句检索neo4j

"""
实现思路：
    先确定查询知识图谱的入口节点，根据入口节点和用户意图生成Cypher语句进行查询
    1.LLM根据用户输入，确定需要的入口节点类型以及实体
    2.根据提供的入口节点类型和实体，使用混合检索获取候选入口节点信息
    3.LLM根据用户输入和入口节点信息生成Cypher查询语句
    4.LLM验证生成的Cypher语法、逻辑是否正确，罗列出错误信息
    5.LLM根据用户输入、入口节点、错误信息、先前Cypher语句来生成更正后的Cypher语句
    6.执行Cypher查询，返回查询结果
工作流程:
    用户提问：用户输入查询问题
    标签路由：LLM识别问题涉及的节点类型和实体
    节点检索：使用混合检索找到候选入口节点
    Cypher生成：LLM根据入口节点和问题生成Cypher查询语句
    语句验证：验证Cypher语句的正确性
    语句校正：如有错误则进行校正
    执行查询：在Neo4j中执行Cypher查询
    返回结果：将查询结果返回给用户
"""
import warnings

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=UserWarning, module="jieba") # 忽略jieba库的用户警告，保持日志清洁

import os
import re
import json
import dotenv
import jieba
import logging
import asyncio
from typing import Any, Text
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from neo4j.exceptions import CypherSyntaxError
from neo4j_graphrag.retrievers import HybridRetriever
from langchain_openai.chat_models import ChatOpenAI
#from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_neo4j import Neo4jGraph
from neo4j_graphrag.retrievers.text2cypher import extract_cypher
from langchain_community.chains.graph_qa.cypher import CypherQueryCorrector, Schema
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

# 配置控制台日志
logger = logging.getLogger("retrieval")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("[%(levelname)s]%(asctime)s: %(message)s")
    handler = logging.StreamHandler() #创建处理器（StreamHandler）：将日志输出到控制台（标准输出 stdout）
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# 路由输出定义,使用Pydantic定义数据模型
class RouteItem(BaseModel):
    label: str = Field(..., description='节点类型，比如"SKU","Attr"')  # 节点类型，如 "SKU"、"Trademark"
    entity: str = Field(..., description="实体文本")# 实体文本，如 "华为Mate 40"

class RouteOutput(BaseModel):
    outputs: list[RouteItem] = Field(..., description="需要检索知识图谱的入口节点")
    need_graph_search: bool = Field(..., description="是否需要检索知识图谱") # 走Neo4j
    need_local_search: bool = Field(..., description="是否需要本地检索") # 走本地的RAG
    answer: str|None = Field(None, description="不需要图谱检索时直接回答的问题")

class BgeEmbedding(Embeddings):
    """定义嵌入模型"""

    def __init__(self):
        # 将相对路径转为绝对路径，避免被当作 HuggingFace repo id
        model_path = os.path.join(os.path.dirname(__file__), ".", "model/bge-base-zh")
        model_path = os.path.abspath(model_path)
        self.model = SentenceTransformer(model_path)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts, batch_size=64, normalize_embeddings=True
        )
        return [list(map(float, emb)) for emb in embeddings]


def get_chat_history( chat_events: list[dict[str, Any]] | None,max_messages: int = 6,) -> str:
    """将最近的历史对话格式化为文本。"""
    if not chat_events:
        return ""

    normalized_events: list[tuple[str, str]] = []

    for event in chat_events:
        if not isinstance(event, dict):
            continue
        role = str(event.get("role") or "").strip().lower()
        content = str(event.get("content") or "").strip()
        normalized_events.append((role, content))

    # 使用偶数条消息，尽量保留完整问答轮次
    max_messages = max(2, max_messages)
    if max_messages % 2 != 0:
        max_messages += 1

    recent_events = normalized_events[-max_messages:]

    return "\n".join(
        f"{role}: {content}"
        for role, content in recent_events
    )

class GraphRAG:
    """GraphRAG 知识图谱检索类"""

    def __init__(self, embedding):
        self.embedding = embedding
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
        self.driver = GraphDatabase.driver(URI, auth=AUTH)
        neo4j_graph = Neo4jGraph(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD"),
            enhanced_schema=True
        )
        self.schema = neo4j_graph.schema
        # 入口节点可选标签
        self.optional_label = (
            '- Category1:   一级分类，如"食品饮料"、"家用电器"、"手机"'
            '- Category2:   二级分类，如"大家电"、"香水彩妆"'
            '- Category3:   三级分类，如"手机"、"香水"、"笔记本"'
            '- Trademark:   品牌，如"华为"、"索芙特"、"金沙河"'
            '- SPU:         商品名称，如"华为Mate 40 pro"'
            '- SKU:         单品名称，如"联想（Lenovo） 拯救者Y9000P 2022 16英寸游戏笔记本电脑 i9-12900H RTX3070Ti 钛晶灰"'
            '- Attr:        商品属性值，如"70英寸"、"蓝色"、"非有机食品"、"运行内存"'
            '- User:        用户ID，如"176"'
        )

        self.relationship_rules= """
            - (:User)-[:View]->(:SKU)
            - (:SKU)-[:Belong]->(:SPU)
            - (:SKU)-[:Have]->(:Attr)
            - (:SPU)-[:Belong]->(:Trademark)
            - (:SPU)-[:Belong]->(:Category3)
            - (:Category3)-[:Belong]->(:Category2)
            - (:Category2)-[:Belong]->(:Category1)
            
            必须遵守：
            1. 通过品牌查询商品，只能通过(:SPU)-[:Belong]->(:Trademark)查询，注意，方向不能反了。
            2. 查询属性只能通过(:SKU)-[:Have]->(:Attr)查询。
            3. 禁止生成 (:SPU)-[:Have]->(:Attr)
            4. SKU同时连接SPU和Attr时，这是分支结构，必须拆分多个Match:
                MATCH (:SKU)-[:Have]->(:Attr)
                MATCH (:SKU)-[:Belong]->(:SPU)
            5. 禁止生成 MATCH (:SKU)-[:Belong]->(:SPU)-[:Have]->(:Attr)
            6. 只有用户行为或者个性化问题才使用User节点。
            7. 商品查询等基本查询不使用User节点。
        """


        # 1. 路由标签识别
        self.route_label_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    """
                        你是商品知识图谱系统的多轮对话路由助手。
                        你必须始终使用唯一的 RouteOutput 结构返回结果。
                        不要把任何字段值当作工具名称。

                        处理规则：
                        1. 商品知识图谱问题
                         如果问题涉及商品、SKU、SPU、品牌、分类、属性或者用户行为关系：
                          - need_graph_search 返回True
                          - need_local_search 返回False
                          - outputs 返回图谱检索入口节点
                          - answer 返回null

                        2. 政策、手册、维修、售后等知识库问题
                          - need_local_search 返回True
                          - need_graph_search 返回False
                          - outputs 返回空列表
                          - answer 返回知识库检索结果

                        3. 不需要知识图谱和本地知识的问题
                        例如：
                         - 普通问候
                         - 闲聊
                         - 感谢
                         - 询问前面的聊天内容
                         处理要求：
                         - need_graph_search 返回False
                         - need_local_search 返回False
                         - outputs 返回空列表
                         - 根据问题直接生成 answer

                        可选知识图谱节点：
                        {optional_label}

                        再次强调：
                        无论问题属于哪种情况，都必须返回 RouteOutput

                    """

                ),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )
        # Cypher 生成 prompt
        # 2. Cypher 生成
        self.generate_cypher_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    """你是一个Cypher专家，请根据入口节点、用户问题和Schema,生成准确的 Cypher 查询语句，
                        {relationship_rules}

                        要求：
                        1.每个关系的起始标签、关系类型、结束标签，必须按照Schema中的定义填写。
                        2.查询结果必须直接回答用户问题。
                        3.不要返回 embedding 等无效属性。
                        4.只返回Cypher语句，不要输出解释或者Markdown代码块。

                        完整的Schema
                        {schema}

                    """
                ),
                HumanMessagePromptTemplate.from_template(
                    "入口节点:\n{entry_nodes}\n\n用户输入:\n{query}\n\nCypher语句:"
                ),
            ]
        )
        # Cypher 验证 prompt
        # 3. Cypher 验证
        self.validate_cypher_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "你是一位Cypher专家，正在审查一位初级开发人员编写的Cypher语句。你需要根据schema和用户输入，检查如下内容：\n"
                    "* Cypher语句中是否需要包含用户信息作为过滤条件？\n"
                    "* Cypher语句中是否有任何语法错误？\n"
                    "* Cypher语句中的关系方向是否符合schema中的定义？\n"
                    "* Cypher语句中是否漏定义了变量或使用了未定义的变量？\n"
                    "* Cypher语句检索出的内容能否用于回答用户的问题？\n"
                    '以严格列表格式输出错误信息，比如"["错误1", "错误2"]"，始终解释schema与Cypher语句之间的差异。'
                    "**注意**：如果确认没有问题，不需要返回任何内容，空即可！。\n"
                    "schema:\n{schema}"
                ),
                HumanMessagePromptTemplate.from_template(
                    "入口节点:\n{entry_nodes}\n\n"
                    "用户输入:\n{query}\n\n"
                    "待验证的Cypher语句:\n{cypher}"
                ),
            ]
        )
        # Cypher 校正 prompt
        # 4. Cypher 校正
        self.correct_cypher_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    """
                    你是一位Cypher专家，正在审查一位初级开发人员编写的Cypher语句
                    重新生成一个完整且合法的Cypher语句。
                    关系：{relationship_rules}

                    注意：
                      1. 不只是修改箭头方向，还要检测关系两端的节点标签。
                      2. 如果原查询把分支错误的写成线性路径，必须拆分多个Match。
                      3. 只输出 cypher语句，不要输出解释或者Markdown代码块。

                    完整 schema:
                    {schema}
                    """
                ),
                HumanMessagePromptTemplate.from_template(
                    "入口节点:\n{entry_nodes}\n\n"
                    "用户输入:\n{query}\n\n"
                    "错误信息:\n{errors}\n\n"
                    "待更正的Cypher语句:\n{cypher}"
                ),
            ]
        )

    async def _route_label(self, query):
		# 1. 根据用户的问题是节点类型和实体
        query = query.strip()
        if not  query:
            return RouteOutput(
                outputs=[],
                need_graph_search=False,
                need_local_search=False,
                answer="请输入要查询的问题",
            )
        # 2. 构建提示词
        prompt = self.route_label_prompt.format(
            query=query,
            optional_label= self.optional_label
        )
        # 3. 调用模型
        route_result = self.llm.with_structured_output(RouteOutput).invoke(prompt)

        if not isinstance(route_result, RouteOutput):
            raise TypeError(f"route_result must be RouteOutput, but got {type(route_result)}")
        """
        outputs=[RouteOutput(label="Trademark", entity="苹果")]
        """
        return route_result

    async def _node_retrieval(self, route_res, top_k):
        """获取具体的入口节点：根据标签和实体，检索出具体的节点
            [RouteItem(label='Category3', entity='手机'), RouteItem(label='Trademark', entity='苹果')]
         1. 区分节点类型，如果是用户节点则直接检索，其他类型则混合检索额
         2. 混合检索
            向量检索和全文检索
             retriever = HybridRetriever(
                driver,
                vector_index_name,
                fulltext_index_name
            )
            retriever.get_search_results(query_text, query_vector,top_k,effective_search_ratio)
        3. 拼接检索结果：
        {
            "Category3":[
                {"category_name":"手机", score=0.9}
            ],
            "Trademark: [
                {"trademark_name":"苹果", score=0.9},
                {"trademark_name":"华为", score=0.8},
            ],
            "User":[
                {"user_id":"123", score=0.9}
            ]
        }
        """

        paris = []
        retrieved_nodes = {}
        # 1. 判断节点是否为用户节点，如果是则直接检索
        for i in route_res:
            if i.label == "User":
                cypher = "MATCH (n:`User`) WHERE n.user_id=$user_id RETURN n"
                user_node, _, _ = self.driver.execute_query(
                    cypher,
                    parameters_={"user_id": int(i.entity)}
                )

                retrieved_nodes.setdefault(i.label, []).append(user_node[0])
            else:
                paris.append((i.label, i.entity))
        if not paris:
            return retrieved_nodes

        # 2. 非用户节点需要进行混合检索
        labels,entities = zip(*paris)
        labels = list(labels)
        entities = list(entities)

        query_vectors = self.embedding.embed_documents(entities) # 生成每个实体的向量
        pattern = re.compile(r"[a-zA-Z0-9\u4e00-\u9fa5]+")  # 匹配英文字母、数字和中文字符
        # ["我们都是中国人","xxx","yyy"]  =》 ["我们 or 都是 or 中国  or 人", "", ""]
        query_texts = [
            " or ".join([
                word.strip() for word in jieba.lcut(text) if pattern.fullmatch(word.strip())
            ])
            for text in entities
        ] # 关键词

        tasks = []
        for label, query_text, query_vector in zip(labels, query_texts, query_vectors):
            retriever = HybridRetriever(
                self.driver,
                vector_index_name=f"{label.lower()}_vector",
                fulltext_index_name=f"{label.lower()}_fulltext",
            )
            #并行检索
            tasks.append(
                asyncio.to_thread(
                    retriever.get_search_results,
                    query_text,
                    query_vector,
                    top_k,
                    effective_search_ratio=2
                )
            )

        # 并发执行所有的检索任务，并等待返回所有的结果
        results = await asyncio.gather(*tasks)
        """
        {
            "Category3":[
                {"category_name":"手机", score=0.9}
            ],
            "Trademark: [
                {"trademark_name":"苹果", score=0.9},
                {"trademark_name":"华为", score=0.8},
            ],
            "User":[
                {"user_id":"123", score=0.9}
            ], 
            "Attr":[
                {"attr_value":"蓝色", score=0.9},
                {"attr_value":"红色", score=0.8},
            ]
        }
        
        """
        for label, result in zip(labels, results):
            retrieved_nodes.setdefault(label, []).extend(
                [
                    {
                        f"{label.lower()}_name": i["node"][f"{label.lower()}_name"],
                        "score": i["score"]
                    }
                    for i in result.records
                ]
                if label !="Attr"
                else [
                    {
                        f"{label.lower()}_value": i["node"][f"{label.lower()}_value"],
                        "score": i["score"]
                    }
                    for i in result.records
                ]
            )

        return retrieved_nodes

    async def _generate_cypher(self, query, entry_nodes):
        """根据查询的具体节点生成语句"""
        prompt = self.generate_cypher_prompt.format_prompt(
            relationship_rules=self.relationship_rules,
            entry_nodes=entry_nodes,
            query=query,
            schema=self.schema
        )
        llm_output = await self.llm.ainvoke(prompt)
        cypher = extract_cypher(llm_output.content)
        return cypher

    async def _validate_cypher(self, query, entry_nodes, cypher):
        errors = []
        # 1. 通过explain进行语法验证
        try:
            self.driver.execute_query(f"explain {cypher}")
        except Exception as e:
            logger.error(f"cypher: {cypher} is invalid")
            errors.append(f"cypher语法错误：{e}")

        # 2. 通过LLM进行逻辑校验
        prompt = self.validate_cypher_prompt.format_prompt(
            cypher=cypher,
            query=query,
            entry_nodes=entry_nodes,
            schema=self.schema
        )
        llm_output = await self.llm.ainvoke(prompt)
        errors.extend(json.loads(llm_output.content.strip()))
        return errors

    async def _correct_cypher(self, query, entry_nodes, cypher, errors):
        # LLM-as-judge :让模型来帮我们纠错，选择一个比较强的coder模型
        prompt = self.correct_cypher_prompt.format_prompt(
            cypher=cypher,
            query=query,
            entry_nodes=entry_nodes,
            schema=self.schema,
            errors=errors,
            relationship_rules=self.relationship_rules
        )
        llm_output = await self.llm.ainvoke(prompt)
        cypher = extract_cypher(llm_output.content)
        return cypher

    async def search(
            self,
            query: Text,
            user_id: str | None = None,
            chat_events: list[dict[str, Any]] | None = None,
    ) -> list[Document]:
        route_result = await self._route_label(query)
        logger.info(f"1. route_result: {route_result}")

        # 判断是否需要走图谱
        if not route_result.need_graph_search:
            # TODO:判断是否需要走本地
            return []

        entry_nodes = await self._node_retrieval(route_result.outputs, top_k=10)
        logger.info(f"2. retrieved_nodes: {entry_nodes}")

        cypher = await self._generate_cypher(query, entry_nodes)
        logger.info(f"3. cypher: {cypher}")

        i=0;
        while True:
            try:
                i+=1
                errors =  await self._validate_cypher(query, entry_nodes, cypher)
                logger.info(f"4. 第{i}次校验结果: {errors}")
                if errors:
                    # 矫正之后有可能还会出现问题
                    cypher = await self._correct_cypher(query, entry_nodes, cypher, errors)
                    logger.info(f"5. 修正后的cypher: {cypher}")
                else:
                    break
                if i>=3:
                    return [Document(page_content="抱歉，没有找到相关内容")]
            except Exception as e:
                logger.error(f"cypher: {cypher} is invalid")
                break


        results = self.driver.execute_query(cypher)
        if not results.records:
            docs = [Document(page_content="抱歉，没有找到相关内容")]
        docs = [Document(page_content=str(dict(result))) for result in results.records]
        logger.info(f"6. 查询结果: {docs}")
        return docs

async  def test():
    embedding = BgeEmbedding()

    graph_rag = GraphRAG(embedding)
    #query = "我是用户13，请问苹果手机都有哪些系列？"
    #query = "hello"
    #query = "Apple12的运行内存这个属性是多少？"
    #query = "苹果手机都有哪些系列？"
    query = "白色256GB的手机有哪些？"
    result = await graph_rag.search(query)

    logger.info(f"最终检索结果为: {result}")


def test_scheam():
    # 使用neo4jgraph获取图谱结构关系
    neo4j_graph = Neo4jGraph(
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD"),
        enhanced_schema=True
    )
    # 获取图谱结构
    structured_schema = neo4j_graph.structured_schema
    #print(structured_schema)

    schema = neo4j_graph.schema
    print(schema)

if __name__ == "__main__":
    # 测试
    asyncio.run(test())

    #test_scheam()