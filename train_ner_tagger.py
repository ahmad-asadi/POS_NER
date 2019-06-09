import nltk
import pprint

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from datasets import load_ner_corpus
from gensim.sklearn_api import W2VTransformer
import numpy as np



# nltk.download("treebank")
# nltk.download("punkt")

tagged_sentences = load_ner_corpus()
test_tagged_sentences = load_ner_corpus(is_train=False)

def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }


def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]

training_sentences = tagged_sentences
test_sentences = test_tagged_sentences

print(len(training_sentences))  # 2935
print(len(test_sentences))  # 979


def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])

    return X, y


X, y = transform_to_dataset(training_sentences)

clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
    # ('SVM', SVC())
])

clf.fit(X[:10000],
        y[:10000])  # Use only the first 10K samples if you're running it multiple times. It takes a fair bit :)

print('Training completed')

X_test, y_test = transform_to_dataset(test_sentences)

print("Accuracy:", clf.score(X_test, y_test))

y_pred = clf.predict(X=X_test)

cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

cm = np.array(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
print(cm)

cm = np.nan_to_num(cm)
np.fill_diagonal(cm, 0)
print(cm)
index = np.argmax(cm)
i = int(index/len(cm))
j = int(index % len(cm))
print("Maximum error: ", i, j)

def pos_tag(sentence):
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return zip(sentence, tags)


print(zip(pos_tag(nltk.word_tokenize('This is my friend, John.'))))
