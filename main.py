from janome.tokenizer import Tokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#python3.9.4だと動かなかった

'''
Japanese is really hard, especially reading and writing.
１．正解と自分の日本語訳の比較
２．正解と間違い日本語訳の比較
'''
def correctAndCorrect(model, answer_translation_1, answer_translation_2):
    t = Tokenizer()
    word_1 = list(t.tokenize(answer_translation_1, wakati=True))
    word_2 = list(t.tokenize(answer_translation_2, wakati=True))
    sim_value = model.docvecs.similarity_unseen_docs(model, word_1, word_2)
    return sim_value

def correctAndOwn(model, answer_translation, my_translation):
    t = Tokenizer()
    word_1 = list(t.tokenize(answer_translation, wakati=True))
    word_2 = list(t.tokenize(my_translation, wakati=True))
    sim_value = model.docvecs.similarity_unseen_docs(model, word_1, word_2)
    return sim_value

def correctAndWrong(model, answer_translation, wrong_translation):
    t = Tokenizer()
    word_1 = list(t.tokenize(answer_translation, wakati=True))
    word_2 = list(t.tokenize(wrong_translation, wakati=True))
    sim_value = model.docvecs.similarity_unseen_docs(model, word_1, word_2)
    return sim_value

if __name__ == '__main__':
    model = Doc2Vec.load("./models/jawiki.doc2vec.dbow300d.model")
    answer_translation = "日本語は本当に難しいね。特に読み書きのことだけど。"
    my_translation = "日本語は、特に読み書きが難しいです。"
    wrong_translation = "日本人はとても固く、読み書きのスペシャリストです。"
    print('問：次の英文を日本語に訳してください．')
    print('Japanese is really hard, especially reading and writing.')
    print('#-----------------------------------------------------------')
    print('Correct answer => ', answer_translation)
    print('My answer => ', my_translation)
    print('Baka answer => ', wrong_translation)
    print('Agreement rate between correct and correct answers => ', correctAndCorrect(model, answer_translation, answer_translation))
    print('Agreement rate between correct and my answers => ', correctAndOwn(model, answer_translation, my_translation))
    print('Agreement rate between correct and baka answers => ', correctAndWrong(model, answer_translation, wrong_translation))
