from jieba.analyse.textrank import TextRank


def load_data(file_path):
    """
    :param file_path: 树洞数据文件路径
    :return: 返回一行一行的树洞数据, 格式为list
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    texts = []
    for line in lines:
        if "#p" not in line and "#c" not in line and '[' not in line:
            texts.append(line.replace('\n', ' '))
    return texts


def get_text_rank(texts, stopwords_path="/Users/yanjin/PycharmProjects/NLP/corpus/stopwords.txt"):
    """
    :param texts: 之前拆分好的, 一行一行的树洞数据, 格式为list
    :param stopwords_path: 分词时使用的停止词的目录
    :return: 一个list, 元素为tuple, 存有词语和其textrank值
    """
    text = ""
    for t in texts:
        text += t
    trk = TextRank()
    trk.set_stop_words(stopwords_path)
    ranks = trk.textrank(text, topK=300, withWeight=True)
    return ranks


if __name__ == '__main__':
    data = load_data('/Users/yanjin/Desktop/pkuhole20181031.txt')
    ranks = get_text_rank(data)
    for i in ranks:
        print(i[1]*100000, i[0])

