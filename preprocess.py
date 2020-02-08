import pandas as pd
from transformers import BertTokenizer
import config
import numpy as np

# train data
stories = pd.read_csv('../data/val.csv', encoding='ANSI')

backgrounds = []
endings = []
labels = []
data = []

length = len(stories)

for i in range(length):
    background = stories['InputSentence1'][i] + stories['InputSentence2'][i] + \
        stories['InputSentence3'][i] + stories['InputSentence4'][i]

    backgrounds.append(background)
    backgrounds.append(background)

    endings.append(stories['RandomFifthSentenceQuiz1'][i])
    endings.append(stories['RandomFifthSentenceQuiz2'][i])

    right_answer = stories['AnswerRightEnding'][i]

    if right_answer == 1:
        labels.append(1)
        labels.append(0)
    else:
        labels.append(0)
        labels.append(1)

tokenizer = BertTokenizer.from_pretrained('bert-large-cased')

length = len(labels)

for i in range(length):
    data.append(tokenizer.encode_plus(text=backgrounds[i], text_pair=endings[i], max_length=config.max_words,
                                      truncation_strategy='do_not_truncate', pad_to_max_length=True)['input_ids'])

data = np.array(data, dtype=np.int)
labels = np.array(labels, dtype=np.int)

np.save('../data/train_data.npy', data)
np.save('../data/train_labels.npy', labels)

# test data
test_stories = pd.read_csv('../data/test.csv', encoding='ansi')

backgrounds = []
endings = []
labels = []
data = []

length = len(test_stories)

for i in range(length):
    background = test_stories['InputSentence1'][i] + test_stories['InputSentence2'][i] + \
        test_stories['InputSentence3'][i] + test_stories['InputSentence4'][i]

    backgrounds.append(background)
    backgrounds.append(background)

    endings.append(test_stories['RandomFifthSentenceQuiz1'][i])
    endings.append(test_stories['RandomFifthSentenceQuiz2'][i])

length = len(backgrounds)

for i in range(length):
    data.append(tokenizer.encode_plus(text=backgrounds[i], text_pair=endings[i], max_length=config.max_words,
                                      truncation_strategy='do_not_truncate', pad_to_max_length=True)['input_ids'])

data = np.array(data, dtype=np.int)

np.save('../data/test_data.npy', data)

# validation data
val_stories = pd.read_csv('../data/train.csv')

backgrounds = []
endings = []
labels = []
data = []

length = 5000

for i in range(length):
    background = val_stories['sentence1'][i] + val_stories['sentence2'][i] + \
        val_stories['sentence3'][i] + val_stories['sentence4'][i]

    backgrounds.append(background)

    endings.append(val_stories['sentence5'][i])

    labels.append(1)

length = len(labels)

for i in range(length):
    data.append(tokenizer.encode_plus(text=backgrounds[i], text_pair=endings[i], max_length=config.max_words,
                                      truncation_strategy='do_not_truncate', pad_to_max_length=True)['input_ids'])

data = np.array(data, dtype=np.int)
labels = np.array(labels, dtype=np.int)

print(data[0])

np.save('../data/val_from_train_data.npy', data)
np.save('../data/val_from_train_labels.npy', labels)
