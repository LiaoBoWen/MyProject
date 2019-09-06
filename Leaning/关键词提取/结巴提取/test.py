from jieba.analyse import extract_tags
import jieba

def rank():
    text = ''.join(['This is the first document.',
          'This is the second document.',
          'And the third one',
          'Is this the first document?'])
    result = extract_tags(text,topK=4,withWeight=True)
    print(result)

if __name__ == '__main__':
    rank()