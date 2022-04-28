from nltk.corpus import wordnet
import binascii
from nltk.corpus import conll2000


# 导入华尔街日报语料库（WSJ）的数据的语料库conll2000
x = (conll2000.tagged_sents())
for i in range(3):
    print(x[i], '\n')


# 同义词反义词
synonyms = []
antonyms = []

for syn in wordnet.synsets("friend"):
    for lm in syn.lemmas():
        synonyms.append(lm.name())
        if lm.antonyms():
            antonyms.append(lm.antonyms()[0].name())
print(set(synonyms))
print(set(antonyms))


# 编码转换
text = b"Simply Easy Learning"

data_b2a = binascii.b2a_uu(text)

data_a2b = binascii.a2b_uu(data_b2a)

print(data_b2a, data_a2b)
