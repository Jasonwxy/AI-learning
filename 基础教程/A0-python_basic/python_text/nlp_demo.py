from jieba import posseg
from pyhanlp import HanLP
# from common import get_run_time

content = '现如今，机器学习和深度学习带动人工智能飞速的发展，并在图片处理、语音识别领域取得巨大成功。'


# def jieba_demo():
#     seg2 = posseg.lcut(content)
#     print([x.word + '/' + x.fl ag for x in seg2])


def hanlp_demo():
    seg3 = HanLP.segment(content)
    print(seg3)


if __name__ == "__main__":
    hanlp_demo()