from py2neo import Graph
graph = Graph('http://localhost:7474',user="neo4j",password="liaobowen421")
'''可以每一条数据导入'''
"""
#测试
cql='''
MATCH (p:Person)
where p.name="张柏芝"
return p
'''
#清空数据库
#data = graph.run('MATCH (n) OPTIONAL MATCH (n)-[r]-() DELETE n,r')
data = graph.run(cql)
print(list(data)[0]['p']["biography"])
"""

#导入节点 电影类型  == 注意类型转换
cql='''
LOAD CSV WITH HEADERS  FROM "file:///data/genre.csv" AS line
MERGE (p:Genre{gid:toInteger(line.gid),name:line.gname})
'''

result = graph.run(cql)
print(result,"电影类型 存储成功")

#导入节点 演员信息
cql='''
LOAD CSV WITH HEADERS FROM 'file:///data/person.csv' AS line
MERGE (p:Person { pid:toInteger(line.pid),birth:line.birth,
death:line.death,name:line.name,
biography:line.biography,
birthplace:line.birthplace})
'''
result = graph.run(cql)
print(result,"演员信息 存储成功")

#导入节点 电影信息
cql='''
LOAD CSV WITH HEADERS  FROM "file:///data/movie.csv" AS line
MERGE (p:Movie{mid:toInteger(line.mid),title:line.title,introduction:line.introduction,
rating:toFloat(line.rating),releasedate:line.releasedate})
'''
result = graph.run(cql)
print(result,"电影信息 存储成功")

#导入关系 actedin  电影是谁参演的 1对多
cql='''
LOAD CSV WITH HEADERS FROM "file:///data/person_to_movie.csv" AS line
match (from:Person{pid:toInteger(line.pid)}),(to:Movie{mid:toInteger(line.mid)})
merge (from)-[r:actedin{pid:toInteger(line.pid),mid:toInteger(line.mid)}]->(to)
'''
result = graph.run(cql)
print(result,"电影信息<-->演员信息 存储成功")

#导入关系 is 电影是什么类型 == 1对多
cql='''
LOAD CSV WITH HEADERS FROM "file:///data/movie_to_genre.csv" AS line
match (from:Movie{mid:toInteger(line.mid)}),(to:Genre{gid:toInteger(line.gid)})
merge (from)-[r:is{mid:toInteger(line.mid),gid:toInteger(line.gid)}]->(to)
'''
result = graph.run(cql)
print(result,"电影信息<-->电影类型 存储成功")