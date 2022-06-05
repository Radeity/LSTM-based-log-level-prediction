from torch import nn

class LSTMModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        # vacab_size是使用的字典的长度
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM模块使用word_embeddings作为输入，输出的维度为hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # nn.Linear将LSTM模块的输出映射到目标向量空间，即线性空间
        self.linear = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        linear_out = self.linear(lstm_out.view(len(sentence), -1))
