import nltk

file_name = "./file.txt"

with open(file_name, 'r') as f:
    lines_in_file = f.read()

    nltk_tokens = nltk.word_tokenize(lines_in_file)
    print(nltk_tokens, '\n', 'Number of Words :', len(nltk_tokens))

with open(file_name, 'r')as f:
    lines_in_file = f.read()

    print(lines_in_file.split(), '\n', 'Number of Words', len(lines_in_file.split()))

sentence_data = "The First sentence is about Python. The Second: about Django. You can learn Python,Django and Data Ananlysis here. "

nltk_tokens = nltk.sent_tokenize(sentence_data)
print(nltk_tokens)
