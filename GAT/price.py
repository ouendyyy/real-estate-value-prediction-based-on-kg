from neo4j import GraphDatabase
import pandas as pd
import torch


def get_price():
    # Neo4j
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "thywdy123"

    # Cypher
    query1 = """
    MATCH (house:House)-[r1]->(property:Property)-[r2]->(Price)
    WITH house, AVG(Price.price) AS avg_price
    RETURN  avg_price

    """

    query2 = """
    MATCH (street:Street)-[r3]-(house:House)-[r1]->(property:Property)-[r2]->(Price)
    WITH street, AVG(Price.price) AS avg_price
    RETURN  avg_price

    """

    query3 = """
    MATCH (region:Region)-[r4]-(street:Street)-[r3]-(house:House)-[r1]->(property:Property)-[r2]->(Price)
    WITH region, AVG(Price.price) AS avg_price
    RETURN  avg_price

    """

    # 连接到 Neo4j 数据库
    driver = GraphDatabase.driver(uri, auth=(username, password))

    # 执行查询
    with driver.session() as session:
        result1 = session.run(query1)
        result2 = session.run(query2)
        result3 = session.run(query3)
        # 将查询结果存储到 DataFrame 中
        df1 = pd.DataFrame([r.data() for r in result1])
        df2 = pd.DataFrame([r.data() for r in result2])
        df3 = pd.DataFrame([r.data() for r in result3])

    # 关闭数据库连接
    driver.close()
    tensor1 = torch.squeeze(torch.tensor(df1.values, dtype=torch.float) / 1000000)
    tensor2 = torch.squeeze(torch.tensor(df2.values, dtype=torch.float) / 1000000)
    tensor3 = torch.squeeze(torch.tensor(df3.values, dtype=torch.float) / 1000000)

    return tensor1, tensor2, tensor3


get_price()
