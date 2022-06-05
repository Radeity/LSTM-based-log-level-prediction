import os
from parse import load_dumped_data
import pandas as pd
import numpy as np
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, SpatialDropout1D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

ordinal_label_level = {
    'trace': '10000',
    'debug': '11000',
    'info': '11100',
    'warn': '11110',
    'error': '11111'
}
normal_label_level = {
    'trace': 0,
    'debug': 1,
    'info': 2,
    'warn': 3,
    'error': 4
}

# 设置最频繁使用的500个词
MAX_NB_WORDS = 500
# 每条cut_review最大的长度
MAX_SEQUENCE_LENGTH = 200
# 可以忽略的单词上限
MAX_TRUNC_LEN = 50
# 设置Embeddingceng层的维度
EMBEDDING_DIM = 100
epochs = 100
batch_size = 24
LOG_LEVEL_TYPE = 5


def build_word_dict(ast):
    df_list = []
    for block in ast:
        l = [normal_label_level[block.level], ordinal_label_level[block.level], " ".join(block.combine_feature)]
        df_list.append(l)
    df = pd.DataFrame(df_list, columns=['id', 'label', 'input'])
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['input'].values)
    word_index = tokenizer.word_index
    list_values = list(word_index.values())
    print('共有 %s 个不相同的词语.' % len(word_index))
    X = tokenizer.texts_to_sequences(df['input'].values)
    # 填充X,让X的各个列的长度统一
    X_idx = -1
    for x in X:
        X_idx += 1
        if len(x) <= MAX_SEQUENCE_LENGTH:
            continue
        idx = len(list_values) - 1
        trunc_sum = 0
        while len(x) > MAX_SEQUENCE_LENGTH and trunc_sum < MAX_TRUNC_LEN:
            if list_values[idx] in x:
                last_len = len(x)
                x = list(filter(lambda t: t != list_values[idx], x))
                cur_len = len(x)
                trunc_sum += last_len - cur_len
            idx -= 1
        X[X_idx] = x

    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
    Y = pd.get_dummies(df['id']).values
    print(X.shape)
    print(Y.shape)
    return X, Y


def stratified_random_sampling(X, Y):
    X_train_list = []
    X_test_list = []
    Y_train_list = []
    Y_test_list = []
    for i in range(LOG_LEVEL_TYPE):
        type_idx = np.where(Y[:, i] == 1)
        X_type = X[type_idx]
        Y_type = Y[type_idx]
        X_train, X_test, Y_train, Y_test = train_test_split(X_type, Y_type, test_size=0.20, random_state=42)
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        Y_train_list.append(Y_train)
        Y_test_list.append(Y_test)

    for i in range(0, LOG_LEVEL_TYPE - 1):
        X_train = np.vstack([X_train, X_train_list[i]])
        X_test = np.vstack([X_test, X_test_list[i]])
        Y_train = np.vstack([Y_train, Y_train_list[i]])
        Y_test = np.vstack([Y_test, Y_test_list[i]])

    return X_train, X_test, Y_train, Y_test


if __name__ == '__main__':
    if os.path.exists("./Data/ast/ast-kafka.pkl"):
        ast = load_dumped_data("ast", "kafka")

    X, Y = build_word_dict(ast)

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    X_train, X_test, Y_train, Y_test = stratified_random_sampling(X, Y)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    print("end!")
