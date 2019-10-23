import pandas as pd
from py2neo import Graph, Relationship, Node
import re

class To_db:
    def __init__(self):
        self.graph = Graph('http://localhost:7474',
                           username='neo4j',
                           password='liaobowen421')

    def run(self,cypher):
        self.graph.run(cypher)

    def convert_person(self):
        with open('data/person.txt', 'r', encoding='utf-8') as f:
            data = pd.read_csv(f)
            data['birth'] = data['birth'].apply(lambda x: {r'\N':"UNK"}.get(x, x))
            data['death'] = data['death'].apply(lambda x: {r'\N':"UNK"}.get(x, x))
            data['biography'] = data['biography'].apply(lambda x: {r'\N':"UNK"}.get(x, x))
            data['birthplace'] = data['birthplace'].apply(lambda x: {r'\N':"UNK"}.get(x, x))
            print(data)

            data.to_csv('data/person.csv')

            for meg in data.itertuples():
                self.run('merge (n:Person{name:"%s", birth:"%s", pid:"%s", biography:"%s", birthplace:"%s", death:"%s"})'%(meg.name,
                                                                                                                meg.birth,
                                                                                                                meg.pid,
                                                                                                                meg.biography,
                                                                                                                meg.birthplace,
                                                                                                                meg.death))


    def convert_genre(self):
        with open('data/genre.txt', 'r', encoding='utf-8') as f:
            data = pd.read_csv(f)
            data.to_csv('data/penre.csv')
            for msg in data.itertuples():
                self.run('merge (n:Genre{gid:"%s", name:"%s"})'%(msg.gid, msg.gname))


    def convert_movies(self):
        with open('data/movie.txt', 'r', encoding='utf-8') as f:
            data = pd.read_csv(f,names=['mid','title','introduction','rating','releasedate','5','6','7','8','9','10','11'], header=0)
            data.drop(['5','6','7','8','9','10','11'], axis=1,inplace=True)
            data['introduction'].fillna('UNK',inplace=True)
            data['rating'].fillna(0,inplace=True)
            data['rating'] = data['rating'].apply(float)
            data['releasedate'].fillna('UNK',inplace=True)

            data['introduction'] = data['introduction'].apply(lambda x: re.sub(r'[\'\"bfnrt\\]', '', x))
            data['releasedate'] = data['releasedate'].apply(lambda x: re.sub(r'[\'\"bfnrt\\]', '', x))

            data.to_csv('data/movie.csv')

            for msg in data.itertuples():
                self.run('merge (n:Movie{mid:"%s", title:"%s", introduction:"%s", rating:"%s", releasedate:"%s"})'%(msg.mid,
                                                                                                                    msg.title,
                                                                                                                    msg.introduction,
                                                                                                                    msg.rating,
                                                                                                                    msg.releasedate))

    def convert_person_to_movie(self):
        with open('data/person_to_movie.txt', 'r', encoding='utf-8') as f:
            data = pd.read_csv(f)
        data.to_csv('data/person_to_movie.csv')

        for meg in data.itertuples():
            self.run('match (n:Person{pid:"%s"}), (m:Movie{mid:"%s"}) merge (n)-[r:acted{pid:"%s", mid:"%s"}]-(m)'%(meg.pid, meg.mid, meg.pid, meg.mid))


    def convert_movie_person(self):
        with open('data/movie_to_genre.txt','r', encoding='utf-8') as f:
            data = pd.read_csv(f)
        data.to_csv('data/movie_to_person.csv')

        for meg in data.itertuples():
            self.run('match (g:Genre{gid:"%s"}), (m:Movie{mid:"%s"}) merge (m)-[r:is{mid:"%s", gid:"%s"}]-(g)'%(meg.gid, meg.mid, meg.mid, meg.gid))


if __name__ == '__main__':
    to_db = To_db()
    to_db.convert_person()
    to_db.convert_genre()
    to_db.convert_movies()
    to_db.convert_person_to_movie()
    to_db.convert_movie_person()