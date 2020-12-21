from nltk.corpus import gutenberg
import nltk


# fields = gutenberg.fileids()  # 获取多文本文档集合组

def pre_process(text):
    sentences = nltk.sent_tokenize(text)  # 将文本划分为不同的行     1. 分句

    words = [nltk.word_tokenize(sent) for sent in sentences]  # 将文本拆分成单词 2. 分词

    tagged_words = [nltk.pos_tag(sent) for sent in words]  # 标记单词语法分类  3. 标记词性

    return tagged_words


def blocking(grammar, words):
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(words)
    result.draw()
    return result


if __name__ == "__main__":
    # sample = gutenberg.raw("blake-poems.txt")  # 选取其中一个文件，访问原始文本
    sample = [("The", "DT"), ("small", "JJ"), ("red", "JJ"), ("flower", "NN"), ("flew", "VBD"), ("through", "IN"),
              ("the", "DT"), ("window", "NN")]
    # grammar = """NP:{<DT><JJ>*<NN>}{<DT>}"""
    grammar = """NP:{<.*>+}
    }<JJ>*<NN>+{"""
    blocking(grammar, sample)
    # word_list = pre_process(sample)
    # res = blocking(word_list[3])
    # print(res)
