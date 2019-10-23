from py2neo import Node, Relationship, Graph
import pandas as pd

def get_date(path='/media/liao/Data/temp_data/relation.txt'):
    with open(path, encoding='utf-8') as f:
        data = pd.read_csv(f,encoding='utf-8',header=None)
        data = data.to_dict(orient='records')
        print(data)
    return data

def into_graph(data):
    graph = Graph(
        'http://localhost:7474',
        username='neo4j',
        password='liaobowen421'
    )
    graph.run('match (n)-[l]-(m) delete m,n,l')
    for example in data:
        graph.run('merge  (n:Person{name:"%s",home:"%s"}) return 1'%(example[0], example[3]))
        graph.run('merge  (n:Person{name:"%s",home:"%s"}) return 1'%(example[1], example[4]))
        graph.run('merge  (n:Place{name:"%s"}) return 1'%(example[3]))
        graph.run('merge  (n:Place{name:"%s"}) return 1'%(example[4]))
        graph.run('match  (p1:Person{name:"%s"}), (p2:Person{name:"%s"}) merge  (p1)-[:%s]->(p2) return 1'%(example[0], example[1], example[2]))
        graph.run('match  (p1:Person{name:"%s"}), (p2:Place{name:"%s"}) merge  (p1)-[:生于]->(p2) return 1'%(example[0], example[3]))
        graph.run('match  (p1:Person{name:"%s"}), (p2:Place{name:"%s"}) merge  (p1)-[:生于]->(p2) return 1'%(example[1], example[4]))




if __name__ == '__main__':
    # data = get_date()
    # into_graph(data=data)
    graph = Graph(
        'http://localhost:7474',
        username='neo4j',
        password='liaobowen421'
    )
    result = graph.run('match (n)-[l]-(m) where n.name="王熙凤" return n.name,l,m.home limit 3')
    # for meg in result:
    #     print(meg['n.name'], meg['m.home'])
    print(list(result)[0][2])