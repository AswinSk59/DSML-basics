import pandas as pd import nltk nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')




colnames = ['Sentiment', 'news']
df = pd.read_csv('/content/all-data - all-data.csv', encoding="ISO-8859-1", names=colnames, header=None)
sentences_for_chunking = df['news'].head(3) def perform_chunking(sentence):
tokens = nltk.word_tokenize(sentence) pos_tags = nltk.pos_tag(tokens) grammar = "NP: {<DT>?<JJ>*<NN>}"
chunk_parser = nltk.RegexpParser(grammar) chunks = chunk_parser.parse(pos_tags)
 
print(chunks)
for sentence in sentences_for_chunking: print("\nOriginal Sentence:", sentence) perform_chunking(sentence)


 
 


