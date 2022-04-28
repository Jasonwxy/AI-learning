from nltk.corpus import stopwords
import nltk,string
from translation import trans_text
from textwrap3 import wrap

en_stops = stopwords.words()

with open('./file.txt', 'r') as f:
    lines_in_file = f.read()
    # nltk_lines = wrap(lines_in_file,400)
    # for line in nltk_lines:
    #     print(trans_text('en', 'zh', line))
    nltk_tokens = nltk.word_tokenize(lines_in_file)
    print(nltk_tokens, len(nltk_tokens))
    new_tokens = []
    for word in nltk_tokens:
        if word not in en_stops and word not in string.punctuation:
            new_tokens.append(word)
    print(nltk_tokens, len(nltk_tokens))



