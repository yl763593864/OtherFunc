import pickle
from EnglishWordsMongo.MyCode.word_type import Word
import pymongo
import random


word_list = []
client = pymongo.MongoClient(host='localhost', port=27017)
db = client.english
words_col = db.words

def read_words():
    with open('../MyData/words') as fs:
        lines = fs.readlines()
        is_new = True
        new_word = None
        for line in lines:
            # print(len(line), line)
            if is_new and len(line.split(' ')) == 1 and line != '\n':
                is_new = False
                new_word = Word(line.strip())
                word_list.append(new_word)
            elif is_new == False and (line == '\n' or len(line) == 2):
                is_new = True
            elif is_new == False and len(line.split(' ')) >= 2 and line != '\n':
                new_word.add_sentence(line.strip())
            elif is_new == False and len(line.split(' ')) == 1 and line != '\n':
                new_word.add_similar(line.strip())


def save_words(word, dic):
    s = {}
    s['word'] = word
    my_words = words_col.find(s)
    is_exist = False
    for x in my_words:
        is_exist = True

    if is_exist==False:
        words_col.insert_one(dic)


def random_word():
    x = words_col.aggregate([{'$sample': {'size': 1}}])
    for w in x:
        print('word:\t', w['word'])
        for sentence in w['sentence']:
            print('sentence:\t', sentence)
        for similar in w['similar']:
            print('similar:\t', similar)
        for synonymous in w['synonymous']:
            print('synonymous:\t', synonymous)
        for antonym in w['antonym']:
            print('antonym:\t', antonym)


def find_word(word):
    s = {}
    s['word'] = word
    my_words = words_col.find(s)
    for w in my_words:
        print('word:\t', w['word'])
        for sentence in w['sentence']:
            print('sentence:\t', sentence)
        for similar in w['similar']:
            print('similar:\t', similar)
        for synonymous in w['synonymous']:
            print('synonymous:\t', synonymous)
        for antonym in w['antonym']:
            print('antonym:\t', antonym)


def update_word(word, new_word=None, new_sentence=None, new_similar=None,
                new_synonymous=None, new_antonym=None, replace=False):
    s = {}
    s['word'] = word
    my_words = words_col.find(s)
    if new_word is not None:
        new_change = {}
        new_char = {}
        new_char['word'] = new_word
        new_change['$set'] = new_char
        words_col.update_one(s, new_change)

    # update word's sentence
    if new_sentence is not None:
        new_sen = []
        if replace is False:
            for w in my_words:
                for sentence in w['sentence']:
                    new_sen.append(sentence)

        for sentence in new_sentence:
            new_sen.append(sentence)

        new_change = {}
        new_char = {}
        new_char['sentence'] = new_sen
        new_change['$set'] = new_char
        words_col.update_one(s, new_change)

    # update word's similar
    if new_similar is not None:
        new_sim = []
        if replace is False:
            for w in my_words:
                for similar in w['similar']:
                    new_sim.append(similar)

        for similar in new_similar:
            new_sim.append(similar)

        new_change = {}
        new_char = {}
        new_char['similar'] = new_sim
        new_change['$set'] = new_char
        words_col.update_one(s, new_change)



def find_sentence(word):
    s = {}
    s['word'] = word
    my_words = words_col.find(s)
    new_sentence = []
    for w in my_words:
        # print('word:\t', w['word'])
        # for sentence in w['sentence']:
        #     new_sentence.append(sentence)
        print(w)

    return new_sentence



# read_words()
#
# for word in word_list:
#     save_words(word.word, word.get_sql_string())

#
# new_change = ['revenge']
# update_word('reserve', new_similar=new_change, replace=True)
# find_word('reserve')

# random_word()
find_word('presumably')


