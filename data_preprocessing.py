#Biblioteca de pré-processamento de dados de texto
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# para stemizar palavras
from nltk.stem import PorterStemmer

# crie um objeto/instância da classe PorterStemmer()


# importando a biblioteca json
import json
import pickle
import numpy as np

words=[] #lista de palavras raízes únicas nos dados
classes = [] #lista de tags únicas nos dados
pattern_word_tags_list = [] #lista do par de (['palavras', 'da', 'frase'], 'tags')

# palavras a serem ignoradas durante a criação do conjunto de dados
ignore_words = ['?', '!',',','.', "'s", "'m"]

# abrindo o arquivo JSON, lendo os dados dele e o fechando.
train_data_file = open('intents.json')
data = json.load(train_data_file)
train_data_file.close()

# criando função para stemizar palavras
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:

        # escreva o algoritmo de stemização:
      stemmer = PorterStemmer()

    # Exemplo de uso:
    palavra = "running"
    stem_palavra = stemmer.stem(palavra)

    print(f"Palavra original: {palavra}")
    print(f"Palavra stemizada: {stem_palavra}")
        # Adicione o código aqui #        

    return stem_words
class StemmerExample:
    def __init__(self):
        # Crie uma instância da classe PorterStemmer
        self.stemmer = PorterStemmer()

    def get_stem_words(self, words_to_stem, words_to_ignore):
        stem_words = []

        for word in words_to_stem:
            # Verifique se a palavra não está na lista de palavras a serem ignoradas
            if word not in words_to_ignore:
                # Stemize a palavra e adicione à lista de palavras stemizadas
                stem_words.append(self.stemmer.stem(word))
            else:
                # Se a palavra estiver na lista de palavras a serem ignoradas, mantenha-a inalterada
                stem_words.append(word)

        return stem_words

    if __name__ == "__main__":
        stemmer_example = StemmerExample()

        # Lista de palavras a serem stemizadas
        words_to_stem = ["running", "jumps", "played", "running"]

        # Lista de palavras a serem ignoradas
        words_to_ignore = ["played"]

        # Obtém a lista de palavras stemizadas
        stem_words_result = stemmer_example.get_stem_words(words_to_stem, words_to_ignore)

        # Exibe o resultado
        print("Palavras a serem stemizadas:", words_to_stem)
        print("Palavras a serem ignoradas:", words_to_ignore)
        print("Palavras stemizadas resultantes:", stem_words_result)

'''
Lista de palavras-tronco ordenadas para nosso conjunto de dados : 

['tudo', 'alg', 'algue', 'são', 'incríve', 'ser', 'melhor', 'bluetooth', 'tchau', 'camera', 'pode', 'chat',
'legal', 'poderia', 'dígito', 'fazer', 'para', 'jogo', 'adeu', 'ter', 'fone de ouvid', 'ola', 'ajudar', 'ei',
'oi', 'ola', 'como', 'e', 'depois', 'ultimo', 'eu', 'mais', 'proximo', 'legal', 'telefone', 'favo', 'popular ',
'produto', 'fornec', 'ver', 'vender', 'mostrar', 'smartphon', 'dizer', 'agradecer', 'isso', 'o', 'la',
'ate', 'tempo', 'até', 'tende', 'vídeo', 'que', 'qual', 'voce', 'seu']

'''


# criando uma função para criar o corpus
def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):

    for intent in data['intents']:

        # Adicione todos os padrões e tags a uma lista
        for pattern in intent['patterns']:  

            # tokenize o padrão          
            pattern_words = nltk.word_tokenize(pattern)

            # adicione as palavras tokenizadas à lista words        
                          
            # adicione a 'lista de palavras tokenizadas' junto com a 'tag' à lista pattern_word_tags_list
            
            
        # Adicione todas as tags à lista classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

            
    stem_words = get_stem_words(words, ignore_words) 

    # Remova palavras duplicadas de stem_words

    # ordene a lista de palavras-tronco e a lista classes

    
    # imprima a stem_words
    print('lista de palavras stemizadas: ' , stem_words)

    return stem_words, classes, pattern_word_tags_list


# Conjunto de dados de treinamento: 
# Texto de Entrada----> como Saco de Palavras (Bag Of Words) 
# Tags----------------> como Label

def bag_of_words_encoding(stem_words, pattern_word_tags_list):
    
    bag = []
    for word_tags in pattern_word_tags_list:
        # exemplo: word_tags = (['Ola', 'voce'], 'saudação']

        pattern_words = word_tags[0] # ['Hi' , 'There]
        bag_of_words = []

        # Stemizando palavras padrão antes de criar o saco de palavras
        stemmed_pattern_word = get_stem_words(pattern_words, ignore_words)

        # Codificando dados de entrada 
        '''
        Escreva o algoritmo BOW:
        1) pegue uma palavra da lista stem_words
        2) verifique se essa palavra está em stemmed_pattern_word
        3) anexe 1 no BOW; caso contrário, anexe 0
        '''
        
        bag.append(bag_of_words)
    
    return np.array(bag)

def class_label_encoding(classes, pattern_word_tags_list):
    
    labels = []

    for word_tags in pattern_word_tags_list:

        # Comece com uma lista de 0s
        labels_encoding = list([0]*len(classes))  

        # exemplo: word_tags = (['ola', 'voce'], 'saudação']

        tag = word_tags[1]   # 'saudação'

        tag_index = classes.index(tag)

        # Codificação de etiquetas
        labels_encoding[tag_index] = 1

        labels.append(labels_encoding)
        
    return np.array(labels)

def preprocess_train_data():
  
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words)
    
    # Converta as palavras-tronco e a lista classes para o formato de arquivo Python pickle
    

    train_x = bag_of_words_encoding(stem_words, word_tags_list)
    train_y = class_label_encoding(tag_classes, word_tags_list)
    
    return train_x, train_y

bow_data  , label_data = preprocess_train_data()

# depois de completar o código, remova o comentário das instruções de impressão
print("primeira codificação BOW: " , bow_data[0])
print("primeira codificação Label: " , label_data[0])


