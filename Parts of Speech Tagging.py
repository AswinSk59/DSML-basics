import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize nltk.download('stopwords')
nltk.download('punkt') nltk.download('averaged_perceptron_tagger') stop_words = set(stopwords.words('english'))
txt = "The quick brown fox jumps over the lazy dog. " \
"This is a sample sentence for tokenization and part-of-speech tagging. " \ "NLP is an interesting field that involves natural language understanding."
tokenized = sent_tokenize(txt) for i in tokenized:
wordsList = nltk.word_tokenize(i)
wordsList = [w for w in wordsList if not w in stop_words] tagged = nltk.pos_tag(wordsList)
print(tagged)
