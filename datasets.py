


def load_pos_corpus(is_train=True):
    file_name = "./dataset/POStrutf.txt"
    if not is_train:
        file_name = "./dataset/POSteutf.txt"
    corpus = []
    sentence = []
    with open(file_name, "r") as file:
        for line in file:
            line = line.split()
            sentence.append((line[0], line[1]))
            if line[0] == ".":
                corpus.append(sentence)
                sentence = []
    return corpus

def load_ner_corpus(is_train=True):
    file_name = "./dataset/NERtr.txt"
    if not is_train:
        file_name = "./dataset/NERte.txt"
    corpus = []
    sentence = []
    with open(file_name, "r") as file:
        for line in file:
            line = line.split()
            if len(line) == 0:
                continue
            sentence.append((line[0], line[1]))
            if line[0] == ".":
                corpus.append(sentence)
                sentence = []
    return corpus