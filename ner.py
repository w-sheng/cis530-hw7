from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support
import re
import pickle

# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.


def getfeats(word, o, pos):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    
    o = str(o)
    wordlength = str(len(word))
    wordshape = getwordshape(word)
    shortwordshape = getshortwordshape(wordshape)

    features = [
        (o + 'word', word)
        (o + 'length', wordlength),
        (o + 'wordshape', wordshape),
        # (o + 'shortwordshape', shortwordshape),
        (o + 'pos', pos)
    ]

    o = int(o)
    if o == 0:
        prefix1 = word[:1]
        prefix2 = word[:2]
        prefix3 = word[:3]
        prefix4 = word[:4]
        suffix1 = word[-1]
        suffix2 = word[-2:]
        suffix3 = word[-3:]
        suffix4 = word[-4:]
        allupper = word.isupper()
        startupper = word[:1].isupper()
        has_hyphen = '-' in word
        has_accent = hasaccent(word)

        # features.append(('prefix1', prefix1))
        # features.append(('prefix2', prefix2))
        features.append(('prefix3', prefix3))
        features.append(('prefix4', prefix4))
        # features.append(('suffix1', suffix1))
        # features.append(('suffix2', suffix2))
        # features.append(('suffix3', suffix3))
        features.append(('suffix4', suffix4))
        features.append(('allupper', allupper))
        features.append(('startupper', startupper))
        features.append(('has_hyphen', has_hyphen))
        features.append(('has_accent', has_accent))

    return features

def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    for o in [-1,0,1]:
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            pos = sent[i+o][1]
            featlist = getfeats(word, o, pos)
            features.extend(featlist)

    return dict(features)

def getwordshape(word):
    word = re.sub('[A-Z]|[À-Ú]', 'X', word)
    word = re.sub('[a-z]|[à-ú]', 'x', word)
    return re.sub('[0-9]', 'd', word)

def getshortwordshape(word):
    return ''.join(sorted(set(word), key=word.index))

def hasaccent(word):
    return bool(re.search('[À-Úà-ú]', word))

if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

    train_feats = []
    train_labels = []

    for sent in train_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    # TODO: play with other models
    # model = Perceptron(verbose=1)
    # model = SGDClassifier(max_iter=1000, tol=1e-3)
    model = LinearSVC(random_state=0, tol=1e-5)
    # model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    model.fit(X_train, train_labels)
    # pickle.dump(model, open('model', 'wb'))

    test_feats = []
    test_labels = []

    # switch to test_sents for your final results
    for sent in train_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)

    j = 0
    print("Writing to results.txt")
    # format is: word gold pred
    with open("train_results.txt", "w") as out:
        for sent in train_sents: 
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python conlleval.py results.txt")






