# 对关系最好的描述是图，这里使用的图数据库，最常用的图数据库的事noe4j,通过cypher语句可以操作图数据库


from py2neo import Node, Relationship
a = Node('Person',name='Alice')
b = Node('Person',name='Bob')
r = Relationship(a,'Knows',b)
print(a,b,r)

# 赋值属性，如果原本存在这个属性在赋予默认属性没用的
a['Location'] = 'beijin'
a.setdefault('Location','北京')
print(a)

# 通过update来进行属性的批量更新
params = {
    'name':'Liao',
    'age':21,
    'Person':True
}
a.update(params)
print(a)

# subgraph 子图，最简单的构建子图的方式是使用关系运算符 |
s = a | b | r
print(s)


from py2neo import Node, Relationship, Graph

graph = Graph(
    'http://localhost:7474',
    username='neo4j',
    password='liaobowen421',
)

lst = (['廖', 'liao'], ['罗', 'luo'])

for v in lst:
    a = Node('Company',name=v[0])
    b = Node('Company',name=v[1])
    r = Relationship(a, 'INVEST', b)
    s = a | b | r
    graph.create(s)
    r = Relationship(b, 'INVEST', a)
    s = a | b | r
    graph.create(s)

result = graph.run('match data=(n:Person{name:"廖"})-[*1..3]-(m:Person{name:"丹"}) return data').data()
print(result)