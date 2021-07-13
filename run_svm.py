import sklearn.svm as svm
import multiprocessing as mp
from ws_utils import WSUtils
from ws_features import WSCountVectorizer
from helper import remove_space, f1
import time
import datetime
import pickle
import joblib
import argparse


cpu_count = mp.cpu_count()


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--lang',  type=str, help='name of languages')
parser.add_argument('--name',  type=str, help='save as file name')

parser.add_argument('--score_test', action="store_true", help = "mode Evaluation")

args = parser.parse_args()

def load_model(model_path, feature_path):
    model_WS = joblib.load(model_path)
    # Loading features
    vectorizer_WS = pickle.load(open(feature_path, "rb"))
    ws = {'model': model_WS, 'vectorizer': vectorizer_WS}
    return ws

if not args.score_test:
    if args.lang == "vi":
        name = "RDR_VNDict"
    if args.lang == "zhsimp":
        name = "ZHDict"
    print(args.lang)
    train_file = "./data/" + args.lang + "/"+ args.lang  +"_train.txt"
    w_utils               = WSUtils(name) # Utils object for Vietnamese Word Segmentation
    predict_funct         = w_utils.predict_list_of_sentence_ws # Function for prediction after training
    training_sentences    = w_utils.read_ws_corpus(train_file) # Read sentences for training

    X_syls_train, Y_train = w_utils.extract_training_pairs(training_sentences) # Extract training window of syllables
    ratios                = w_utils.compute_ratios(training_sentences) # Extract separable syllables
    seen_words, seen_sfx  = w_utils.pop_seen_words_sfx(training_sentences, ratios) # Extract know words and unknow words containing suffixes


    vectorizer_WS         = WSCountVectorizer(utils=w_utils, ratios=ratios, feature_set=['base', 'long', 'sep', 'sfx'])
    X_train               = vectorizer_WS.fit_transform(X_syls_train) # Transformation function

    model_WS              = svm.LinearSVC(C=0.1) # Linear SVM model
    model_WS.fit(X_train, Y_train) # Train Vietnamese Word segmentation model
    ws                    = {'model': model_WS, 'vectorizer': vectorizer_WS}

    #save model
    model_path = 'models/' + args.lang + "_" + args.name + "_model.pkl"

    print("Saving model...")
    joblib.dump(model_WS,  model_path)

    feature_path = 'models/' + args.lang + "_" + args.name + "_feature.pkl"
    with open(feature_path, 'wb') as fw:
        pickle.dump(vectorizer_WS, fw)

    print('Done saving model!')
else:
    if args.lang == "vi":
        name = "RDR_VNDict"
    if args.lang == "zhsimp":
        name = "ZHDict"

    test_file = "./data/" + args.lang  + "/" + args.lang + "_test.txt"
    w_utils = WSUtils(name)  # Utils object for Vietnamese Word Segmentation
    predict_funct = w_utils.predict_list_of_sentence_ws  # Function for prediction after training
    test_sentences = w_utils.read_ws_corpus(test_file)  # Read sentences for testing

    model_path = 'models/' + args.lang + "_" + args.name + "_model.pkl"
    feature_path = 'models/' + args.lang + "_" + args.name + "_feature.pkl"

    ws = load_model(model_path, feature_path)

    pred_stns_test = predict_funct(ws, cpu_count, test_sentences, False, has_underscore=True)  # Predict

    ###STANZA
    predicted_removed = pred_stns_test[0]
    label_removed = test_sentences

    predicted_list = []
    label = []

    for idx, sentence in enumerate(predicted_removed):
        for idx_char, char in enumerate(sentence):
            test_p = False
            test_l = False
            if char == " ":
                predicted_list.append(1)
                test_p = True
            elif char == "_":
                predicted_list.append(0)
                test_p = True
            elif char == "~":
                predicted_list.append(2)
                test_p = True

            if label_removed[idx][idx_char] == " ":
                label.append(1)
                test_l = True

            elif label_removed[idx][idx_char] == "_":
                label.append(0)
                test_l = True
            elif label_removed[idx][idx_char] == "~":
                label.append(2)
                test_l = True

            if (label_removed[idx][idx_char] == "_" and not char == "_"):
                print("false negative: [", sentence[idx_char - 20:idx_char + 20] + "]",
                      "[" + label_removed[idx][idx_char - 20:idx_char + 20])
                print("\n")

            if (not label_removed[idx][idx_char] == "_" and char == "_"):
                print("false positive: [", sentence[idx_char - 20:idx_char + 20] + "]",
                      "[" + label_removed[idx][idx_char - 20:idx_char + 20])
                print("\n")

            if test_l != test_p:
                print("~~~" + sentence + "~~~", "~~~" + label_removed[idx] + "~~~")
    f1tok = f1(predicted_list, label, {0: 0, 1: 1, 2: 1, 3: 0, 4: 0})
    f1sent = f1(predicted_list, label, {0: 0, 1: 0, 2: 1, 3: 0, 4: 0})

    print("Stanza score word: ", f1tok)
    print("Stanza score sentence: ", f1sent)
    print("F1 score: ", pred_stns_test[1])



