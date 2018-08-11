import numpy as np

words = list()
sentences = list()
vocab_size = 0

def To_One_Hot(word_index):
    global vocab_size    
    temp = np.zeros(vocab_size)
    temp[word_index] = 1
    return temp

def Process(data):
    
    global words
    global sentences    
    raw_sentences = data.split('.')
    for sentence in raw_sentences:
        sentences.append(sentence.strip().split(' '))
        for word in sentence.strip().split(' '):
            words.append(word)
            
def LoadData():
    for i in range(1, 2):
        with open(str(i) + '.txt', 'r', encoding = 'latin-1') as myfile:
            data = myfile.read().replace('\n', '').lower()
            new_data = ''
            for c in data:
                if c in ' qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM,.:;?#!)(':
                    new_data += c
            for c in ',:,;?#)(!':
                new_data = new_data.replace(c, '')
            new_data = new_data.replace('\"', '')
            Process(new_data)
        
def GetData():   
    LoadData()
    
    global words
    global sentences
    global vocab_size
    
    for sentence in sentences:
        while '' in sentence:
            sentence.remove('')

    for sentence in sentences:
        if len(sentence) == 0:
            sentences.remove(sentence)
     
    words = list(set(words))
    for word in words:
        if word == '':
            words.remove('')
        
    vocab_size = len(words)
    word_to_int = {}
    int_to_word = {}
    for i, word in enumerate(words):
        word_to_int[word] = i
        int_to_word[i] = word
    
    x_train = []
    y_train = []
    data = []
    
    WINDOW_SIZE = 4
    for sentence in sentences:
        for word_index, target_word in enumerate(sentence):
            for context_word in sentence[max(word_index-WINDOW_SIZE, 0) : min(word_index+WINDOW_SIZE, len(sentence)+1)]:
                if context_word != target_word:
                    data.append([target_word, context_word])
    
    for data_word in data:
        x_train.append(To_One_Hot(word_to_int[ data_word[0] ]))
        y_train.append(To_One_Hot(word_to_int[ data_word[1] ]))
    
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    return (x_train, y_train, vocab_size, int_to_word, word_to_int)