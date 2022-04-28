from file_read_backwards import FileReadBackwards
import nltk

with FileReadBackwards('./file.txt') as f:
    for line in f:
        nltk_tokens = nltk.word_tokenize(line)
        nltk_tokens.reverse()
        print(nltk_tokens)
