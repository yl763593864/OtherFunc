class Word():
    def __init__(self, word):
        self.word = word
        self.sentence = []
        self.similar = []
        self.synonymous = []
        self.antonym = []

    def change_word(self, new_word):
        self.word = new_word

    def add_sentence(self, sentence):
        self.sentence.append(sentence)

    def add_similar(self, similar):
        self.similar.append(similar)

    def get_sql_string(self):
        my_dic = {'word': self.word, 'sentence': self.sentence,
                  'similar': self.similar, 'synonymous': self.synonymous,
                  'antonym': self.antonym}
        return my_dic
