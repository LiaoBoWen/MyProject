import re
from py2neo import Graph


class Question_template:
    def __init__(self):
        self.graph = Graph('http://localhost:7474',
                           username='neo4j',
                           password='liaobowen421')

        self.template_list = (
            self.get_movies_rating,
            self.get_movie_releasedate,
            self.get_movie_types,
            self.get_movie_introduce,
            self.get_movie_actor_list,
            self.get_actor_info,
            self.get_actor_act_type_movie,
            self.get_actor_act_movie,
            self.get_movie_rating_bigger,
            self.get_movie_rating_smaller,
            self.get_actor_act_type,
            self.get_cooperation_movie_list,
            self.get_actor_movie_count,
            self.get_actor_birthday
        )


    def run(self, cypher):
        result = self.graph.run(cypher)
        return result


    def answer_question(self, question_pos_lst, template):
        '''得到相应的模板id, 之后进入对应的模板func, 得到answer'''
        self.template_id = template
        # self.template_list_ = template[1].split()

        # 预处理问题
        self.question_word, self.question_flag = [], []
        for question_pos in question_pos_lst:
            word, flag = question_pos.split('/')
            self.question_word.append(word.strip())
            self.question_flag.append(flag.strip())
        assert len(self.question_word) == len(self.question_flag)
        answer = self.template_list[self.template_id]()

        return answer


    def get_movie_name(self):
        tag_index = self.question_flag.index('nm')
        movie_name = self.question_word[tag_index]
        return movie_name


    def get_name(self,type_str):
        name_count = self.question_flag.count(type_str)
        if name_count == 1:
            tag_index = self.question_flag.index(type_str)
            actor_name = self.question_word[tag_index]
            return actor_name
        else:
            name_lst = []
            for i, name in enumerate(self.question_flag):
                if name == type_str:
                    name_lst.append(self.question_word[i])
            return name_lst


    def get_num_str(self):
        num_str = re.sub(r'\D', '', ''.join(self.question_word))
        return num_str


    def get_movies_rating(self):
        '''查询电影评分'''
        movie_name = self.get_movie_name()
        cypher = 'match (m:Movie)-[]->() where m.title="{}" return m.rating'.format(movie_name)
        query_result = self.run(cypher)
        query_result = round(float(list(query_result)[0]['m.rating']), 2)
        final_result = "{}的评分为: {}".format(movie_name, query_result)
        return  final_result


    def get_movie_releasedate(self):
        '''查询电影上映时间'''
        movie_name = self.get_movie_name()
        cypher = 'match (m:Movie)-[]->() where m.title="{}" return m.releasedate'.format(movie_name)
        query_result = self.run(cypher)
        query_result = list(query_result)[0]['m.releasedate']
        final_result = "{}的上映时间为: {}".format(movie_name, query_result)

        return final_result


    def get_movie_types(self):
        movie_name = self.get_movie_name()
        cypher = 'match (m:Movie)-[r:is]->(b) where m.title="{}" return b.name'.format(movie_name)
        query_result = self.run(cypher)
        result = []
        for type_result in query_result:
            result.append(type_result['b.name'])
        query_result = '、'.join(result)
        final_result = '{}, 是一部{}类型的电影'.format(movie_name, query_result)

        return final_result


    def get_movie_introduce(self):
        movie_name = self.get_movie_name()
        cypher = 'match (m:Movie)-[]->() where m.title="{}" return m.introduction'.format(movie_name)
        query_result = self.run(cypher)
        result = list(query_result)[0]['m.introduction']
        final_result = "{}简介: {}".format(movie_name, result)

        return final_result


    def get_movie_actor_list(self):
        movie_name = self.get_movie_name()
        cypher = 'match (n:Person)-[r:acted]->(m:Movie) where m.title="{}" return n.name'.format(movie_name)
        query_result = self.run(cypher)
        result = []
        for actor_result in query_result:
            result.append(actor_result['n.name'])
        query_result = '、'.join(result)
        final_result = "参与拍摄{}的演员有：{}".format(movie_name, query_result)
        return final_result


    def get_actor_info(self):
        # todo 这里需要注意一个问题，当返回的actor_name是一个列表的话，查询是会出现问题的
        actor_name = self.get_name('nr')
        cypher = 'match (n:Person)-[]->() where n.name="{}" return n.biography'.format(actor_name)
        query_result = self.run(cypher)
        final_answer = list(query_result)[0]['n.biography']

        return final_answer


    def get_actor_act_type_movie(self):
        actor_name = self.get_name('nr')
        movie_type = self.get_name('ng')
        cypher = 'match (n:Person)-[]->(m:Movie) where n.name="{}" return m.title'.format(actor_name)
        query_result = self.run(cypher)
        result = []
        for movie in query_result:
            try:
                cypher = 'match (m:Movie)-[r:is]->(n) where m.title="{}" return n.name'.format(movie['m.title'])
                temp_query_result = self.run(cypher)
                temp_lst = []
                for type_ in temp_query_result:
                    temp_lst.append(type_['n.name'])
                if len(temp_lst):
                    continue
                if movie_type in temp_lst:
                    result.append(movie)
            except:
                pass
        answer = '、'.join(result)
        final_answer = '{}参演过的{}类型电影有{}'.format(actor_name, movie_type, answer)
        return final_answer


    def get_actor_act_movie(self):
        actor_name = self.get_name('nr')
        cypher = 'match (m:Peron)-[]->(n:Movie) where m.name="{}" return n.title'.format(actor_name)
        result = []
        for movie in self.run(cypher):
            result.append(movie['n.title'])

        query_result = '、'.join(result)
        final_query = '{}参与过的电影有{}'.format(actor_name, query_result)

        return final_query


    def get_movie_rating_bigger(self):
        actor_name = self.get_name('nr')
        rateing = self.get_num_str()
        cypher = 'match (n)-[]->(m:Movie) where n.name="{}" and m.rating>={} return m'.format(actor_name,rateing)
        query_result = self.run(cypher)
        result = []
        for movie in query_result:
            result.append(movie['m.title'])
        query_result = '、'.join(result)
        final_result = "{}参演的评分高于{}的电影有{}".format(actor_name, rateing, query_result)

        return final_result


    def get_movie_rating_smaller(self):
        actor_name = self.get_name('nr')
        rateing = self.get_num_str()
        cypher = 'match (n)-[]->(m:Movie) where n.name="{}" and m.rating<={} return m'.format(actor_name,rateing)
        query_result = self.run(cypher)
        result = []
        for movie in query_result:
            result.append(movie['m.titile'])
        query_result = '、'.join(result)
        final_result = "{}参演的评分低于{}的电影有{}".format(actor_name, rateing, query_result)

        return final_result


    def get_actor_act_type(self):
        actor_name = self.get_name('nr')
        cypher = 'match (n:Person)-[]->(m:Movie)-[r:is]->(t) where n.name="{}" return t.name'.format(actor_name)
        result = []
        for type_ in self.run(cypher):
            result.append(type_['t.name'])
        result = '、'.join(set(result))
        final_result = '{}参演过{}类型的电影'.format(actor_name, result)

        return final_result


    def get_cooperation_movie_list(self):
        actor_name_list = self.get_name('nr')
        cypher = 'match (n:Person)-[]->(m:Movie) where n.name="{}" return m'.format(actor_name_list[0])
        movies_1 = []
        for movie in self.run(cypher):
            movies_1.append(movie['m.title'])
        movies_1 = set(movies_1)
        cypher = 'match (n:Person)-[]->(m:Movie) where n.name="{}" return m'.format(actor_name_list[0])
        movies_2 = []
        for movie in self.run(cypher):
            movies_2.append(movie['m.title'])
        movies_2 = set(movies_2)
        result = movies_1 & movies_2
        result = '、'.join(result)
        final_result = '{}、{}共同参演的电影有：{}'.format(actor_name_list[0], actor_name_list[1], result)

        return final_result


    def get_actor_movie_count(self):
        actor_name = self.get_name('nr')
        cypher = 'match (n:Person)-[]->(m:Movie) where n.name="{}" return m'.format(actor_name)
        movies = []
        # for movie in self.run(cypher):
        #     movies.append(movie[])
        len_ = len(set(movies))
        final_result = '{}参演了{}部电影'.format(movies, len_)

        return final_result


    def get_actor_birthday(self):
        actor_name = self.get_name('nr')
        cypher = 'match (n:Person) where n.name="{}" return n.birthday'.format(actor_name)
        result = self.run(cypher)
        query_result = list(result)[0]['n.birthday']

        final_result = '{}生日是:{}'.format(actor_name, query_result)

        return final_result


if __name__ == '__main__':
    pass