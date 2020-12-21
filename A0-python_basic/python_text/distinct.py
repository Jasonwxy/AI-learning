import nltk

word_data = "The Sky is blue also the ocean is blue also Rainbow has a blue colour."

nltk_tokens = nltk.word_tokenize(word_data)

no_order = list(set(nltk_tokens))
print(no_order)

order_tokens = set()
in_order = []

for word in nltk_tokens:
    if word not in order_tokens:
        order_tokens.add(word)
        in_order.append(word)
print(in_order)
