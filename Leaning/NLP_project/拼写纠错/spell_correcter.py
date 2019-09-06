import re,collections
from string import ascii_lowercase

def tokens(text):
    '''获取所有的文本'''
    return re.findall(r'[a-z]+',text.lower())

with open('./data/big.txt','r') as f:
    WORDS = tokens(f.read())

WORD_COUNTS = collections.Counter(WORDS)

# top 10 wors in corpus
# print(WORD_COUNTS.most_common(10))

def know(words):
    return {w for w in words if w in WORD_COUNTS}

def edits0(word):
    return {word}

def edits1(word):
    # alphabet = ''.join([chr(ord('a') + i) for i in range(26)])
    alphabet = ascii_lowercase

    def splits(word):
        return [(word[:i],word[i:]) for i in range(len(word))]

    pairs = splits(word)
    deletes = [a + b[1:] for (a,b) in pairs if b]
    transposes = [a + b[1] + b[0] + b[2:] for (a,b) in pairs if len(b) > 1]
    replaces = [a + c + b[1:] for (a,b) in pairs for c in alphabet if b]
    inserts = [a + c + b for (a,b) in pairs for c in alphabet]

    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

def correct(word):
    candidates = (
        know(edits0(word)) or
        know(edits1(word)) or
        know(edits2(word)) or
        {word}
    )
    return max(candidates,key=WORD_COUNTS.get)

def correct_match(match):
    word = match.group()
    def case_of(text):
        return str.upper if text.isupper() else \
                str.lower if text.islower() else \
                str.title if text.istitle() else str

    return case_of(word)(correct(word.lower()))

def correct_text_generic(text):
    return re.sub(r'[a-zA-Z]+',correct_match,text)