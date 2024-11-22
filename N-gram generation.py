import numpy as np import pandas as pd
import matplotlib.pyplot as plt plt.style.use(style='seaborn') colnames = ['Sentiment', 'news']
df = pd.read_csv('/content/all-data - all-data.csv', encoding="ISO-8859-1", names=colnames, header=None)
print(df.head())




df.info()


df['Sentiment'].value_counts()

y = df['Sentiment'].values y.shape
x = df['news'].values x.shape


from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.4) print(x_train.shape)
print(y_train.shape) print(x_test.shape) print(y_test.shape)


 

df1 = pd.DataFrame(x_train)
df1 = df1.rename(columns={0: 'news'}) df2 = pd.DataFrame(y_train)
df2 = df2.rename(columns={0: 'sentiment'}) df_train = pd.concat([df1, df2], axis=1) print(df_train.head())


df3 = pd.DataFrame(x_test)
df3 = df3.rename(columns={0: 'news'}) df4 = pd.DataFrame(y_test)
df4 = df2.rename(columns={0: 'sentiment'}) df_test = pd.concat([df3, df4], axis=1) print(df_test.head())



import string
def remove_punctuation(text): if type(text) == float:
return text ans = ""
for i in text:
 
if i not in string.punctuation: ans += i
return ans
df_train['news'] = df_train['news'].apply(lambda x: remove_punctuation(x)) df_test['news'] = df_test['news'].apply(lambda x: remove_punctuation(x)) print(df_train.head())





import nltk
from nltk.corpus import stopwords nltk.download('stopwords')



 
def generate_N_grams(text, ngram=1):
words = [word for word in text.split(" ") if word not in set(stopwords.words('english'))]
print("Sentence after removing stopwords:", words) temp = zip(*[words[i:] for i in range(0, ngram)]) ans = [' '.join(ngram) for ngram in temp]
return ans
print(generate_N_grams("The sun rises in the east", 2))





print(generate_N_grams("The sun rises in the east", 3))




print(generate_N_grams("The sun rises in the east", 4))




