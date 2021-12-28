from numpy import dot
from numpy.linalg import norm


def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def read(file_path, encoding="utf8"):
    with open(file_path, mode="r", encoding=encoding) as f:
        text = f.read()
    return text

def write_text(text, file_path, encoding="utf8"):
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(text)

def write_append(text, file_path, encoding="utf8"):
    with open(file_path, 'a', encoding=encoding) as f:
        f.write(text)

def split_text(text: str):
    text = text.split("\n", 1)
    title = text[0].split("Title:")[1].strip()
    content = text[1].split("Content:")[1].strip()
    return title, content

def log(err: str):
    f = open("/content/log.txt", "a")
    f.write(err)
    f.close()