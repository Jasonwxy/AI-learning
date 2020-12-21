from translate import Translator


def trans_text(from_lang, to_lang, text):
    translator = Translator(from_lang=from_lang, to_lang=to_lang)
    return translator.translate(text)
    # print(translation)


def trans_en_to_zh(text):
    return trans_text('en', 'zh', text)


def trans_zh_to_en(text):
    return trans_text('zh', 'en', text)


if __name__ == "__main__":
    print(trans_zh_to_en('分块'))
