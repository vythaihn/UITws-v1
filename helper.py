def remove_space(file):
    new_file = []
    def startWith(word):
        if word.startswith(".~") or word.startswith("”~") or word.startswith("…~") or word.startswith("...~")  or word.startswith(")~") \
                or word.startswith("?~") or word.startswith("'~")   or word.startswith("!~") :
            return True
        else:
            return False

    for line in file:
        line_pos = line.replace('\n', '')
        '''
        The "underscore" character has the task to concatenate continuous tokens into a work
        The "space" character has the task to segment work (to mark the boundary of two words)
        "Two lines" of code bellow has the task to fix the little errors in the VLSP 2013 for Word Segmentation dataset
        These errors occur only "four times" in the "testing set" of the VLSP 2013 for Word Segmentation dataset
        Therefore, that errors will be not affected on all results because of it very very very very very small than total
        '''
        ######################################
        line_pos = line_pos.replace('_ ', ' ')
        line_pos = line_pos.replace(' _', ' ')
        sentence = line_pos.split(" ")
        spaces = []
        odd_quotes = False

        for word_idx, word in enumerate(sentence):
            space = True
            if word_idx < len(sentence) - 1:
                if sentence[word_idx + 1] in (',', '.', '!', '?', ')', ':', ';', '”', '…', '...'):
                    space = False
                if startWith(sentence[word_idx + 1]):
                    space = False

            if word in ('(', '“'):
                space = False
            #print(word_idx)
            if word == '"' or "\"" in word:
                if odd_quotes:
                    # already saw one quote.  put this one at the end of the PREVIOUS word
                    # note that we know there must be at least one word already
                    odd_quotes = False
                    spaces[word_idx - 1] = False
                else:
                    odd_quotes = True
                    space = False
            spaces.append(space)
        new_sentence = ""
        spaces[-1] = False
        for word_idx, word in enumerate(sentence):
            if spaces[word_idx]:
                new_sentence = new_sentence + word + " "
            else:
                new_sentence = new_sentence + word
        new_file.append(new_sentence)
    return new_file


def f1(pred, gold, mapping):
    pred = [mapping[p] for p in pred]
    gold = [mapping[g] for g in gold]

    lastp = -1;
    lastg = -1
    tp = 0;
    fp = 0;
    fn = 0
    for i, (p, g) in enumerate(zip(pred, gold)):
        if p == g > 0 and lastp == lastg:
            lastp = i
            lastg = i
            tp += 1
        elif p > 0 and g > 0:
            lastp = i
            lastg = i
            fp += 1
            fn += 1
        elif p > 0:
            # and g == 0
            lastp = i
            fp += 1
        elif g > 0:
            lastg = i
            fn += 1

    print("TP:", tp, "FP:", fp, "FN", fn)
    if tp == 0:
        return 0
    else:
        return 2 * tp / (2 * tp + fp + fn)

all_preds = [0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0]
labels = [0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0]

