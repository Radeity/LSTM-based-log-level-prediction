import copy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


class Block:
    def __init__(self, method, level, constant):
        self.method = method
        self.syntactic_feature = []
        self.log_message_feature = []
        self.combine_feature = []
        self.level = level
        self.constant = constant
        self.gen_log_message_feature(constant)

    def gen_log_message_feature(self, constant):
        tokenizer = RegexpTokenizer(r'\w+')
        # split the words using space and remove the punctuation
        split_words = tokenizer.tokenize(constant)
        # filter the common English words
        filtered_words = [w for w in split_words if not w in stop_words]
        # apply stemming on the filtered words
        ps = PorterStemmer()
        stemmed_words = [ps.stem(w) for w in filtered_words]
        self.log_message_feature = stemmed_words
        pass

    def gen_combine_feature(self):
        visit_log_statement = False
        for feature in self.syntactic_feature[::-1]:
            self.combine_feature.insert(0, feature)
            if feature == "LogStatement" and not visit_log_statement:
                visit_log_statement = True
                self.combine_feature.insert(0, "#MsgEnd#")
                self.combine_feature = copy.deepcopy(self.log_message_feature) + self.combine_feature
                self.combine_feature.insert(0, "#MsgStart#")
