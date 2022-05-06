import re


def get_emails(string):
    emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z0-9]+", string)
    print(emails)


def get_url(string):
    urls = re.findall("https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+", string)
    print(urls)


if __name__ == "__main__":
    text = "Please contact us at contact@codingdict.com for further information.You can also give feedbacl at " \
           "feedback@tp.com"
    url_text = """Now a days you can learn almost anything by just visiting http://www.google.com. But if you are 
        completely new to computers or internet then first you need to leanr those fundamentals. Nextyou can 
        visit a good e-learning site like - https://www.codingdict.com to learn further on a variety of subjects.
        https://192.168.11.200 """
    get_emails(text)
    get_url(url_text)
