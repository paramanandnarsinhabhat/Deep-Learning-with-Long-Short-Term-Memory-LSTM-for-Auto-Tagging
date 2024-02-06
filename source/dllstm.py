
'''
1. Load Data and Import Libraries
2. Text Cleaning
3. Merge Tags with Questions
4. Dataset Prepartion
5. Text Representation
6. Model Building
    1. Define Model Architecture
    2. Train the Model
7. Model Predictions
8. Model Evaluation
9. Inference
'''



# Load Data and Import Libraries
import re 

#reading files
import pandas as pd

#handling html data
from bs4 import BeautifulSoup

#visualization
import matplotlib.pyplot as plt  


pd.set_option('display.max_colwidth', 200)


import zipfile
import os

# Specify the path to the zip file
zip_file_path = 'data/archive (2).zip'

# Specify the directory to extract to
extract_to_dir = 'data/unzipped_contents'

# Create a directory to extract to if it doesn't exist
os.makedirs(extract_to_dir, exist_ok=True)

# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all the contents into the directory
    zip_ref.extractall(extract_to_dir)
    
    # List the contents of the extracted folder
    print(f"Contents of the zip file '{zip_file_path}':")
    for file_name in zip_ref.namelist():
        print(file_name)

# Now you can access files inside the unzipped directory
# For example, to open a file:
# with open(os.path.join(extract_to_dir, 'yourfile.txt'), 'r') as file:
#     print(file.read())
        
# load the stackoverflow questions dataset
questions_df = pd.read_csv('data/unzipped_contents/Questions.csv',encoding='latin-1')

# load the tags dataset
tags_df = pd.read_csv('data/unzipped_contents/Tags.csv')

#print first 5 rows

print(questions_df.head())
        
# Text Cleaning
#Let's define a function to clean the text data.

def cleaner(text):

  text = BeautifulSoup(text).get_text()
  
  # fetch alphabetic characters
  text = re.sub("[^a-zA-Z]", " ", text)

  # convert text to lower case
  text = text.lower()

  # split text into tokens to remove whitespaces
  tokens = text.split()

  return " ".join(tokens)

# call preprocessing function
questions_df['cleaned_text'] = questions_df['Body'].apply(cleaner)

print(questions_df['Body'][1])

print(questions_df['cleaned_text'][1])

# Merge Tags with Questions
print(tags_df.head())

# count of unique tags
len(tags_df['Tag'].unique())

print(len(tags_df['Tag'].unique()))

tags_df['Tag'].value_counts()

print(tags_df['Tag'].value_counts())

# remove "-" from the tags
tags_df['Tag']= tags_df['Tag'].apply(lambda x:re.sub("-"," ",x))

# group tags Id wise
tags_df = tags_df.groupby('Id').apply(lambda x:x['Tag'].values).reset_index(name='tags')
tags_df.head()

print(tags_df.head())

# merge tags and questions
df = pd.merge(questions_df,tags_df,how='inner',on='Id')

df = df[['Id','Body','cleaned_text','tags']]

print(df.head())

print(df.shape)

# Dataset Preparation
# check frequency of occurence of each tag
freq= {}
for i in df['tags']:
  for j in i:
    if j in freq.keys():
      freq[j] = freq[j] + 1
    else:
      freq[j] = 1


#Let's find out the most frequent tags.
# sort the dictionary in descending order
freq = dict(sorted(freq.items(), key=lambda x:x[1],reverse=True))

print(freq.items())

# Top 10 most frequent tags
common_tags = list(freq.keys())[:10]
print(common_tags)

x=[]
y=[]

for i in range(len(df['tags'])):
  
  temp=[]
  for j in df['tags'][i]:
    if j in common_tags:
      temp.append(j)

  if(len(temp)>1):
    x.append(df['cleaned_text'][i])
    y.append(temp)

# number of questions left
len(x)

print(len(x))

y[:10]

print(y[:10])
#Now we will find a suitable sequence length.
#We will the input sequences to our model to the length of 100
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
 
y = mlb.fit_transform(y)
y.shape


print(y[0,:])

print(mlb.classes_)

#We can now split the dataset into training set and validation set. 
from sklearn.model_selection import train_test_split
x_tr,x_val,y_tr,y_val=train_test_split(x, y, test_size=0.2, random_state=0,shuffle=True)

# Text Representation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 

#prepare a tokenizer
x_tokenizer = Tokenizer() 

#prepare vocabulary
x_tokenizer.fit_on_texts(x_tr)

print(x_tokenizer.word_index)

print(len(x_tokenizer.word_index))

'''
There are around 25,000 tokens in the training dataset. Let's see how many tokens appear at least 5 times in the dataset.
'''

thresh = 3

cnt=0
for key,value in x_tokenizer.word_counts.items():
  if value>=thresh:
    cnt=cnt+1

print(cnt)

# prepare the tokenizer again
x_tokenizer = Tokenizer(num_words=cnt,oov_token='unk')

#prepare vocabulary
x_tokenizer.fit_on_texts(x_tr)

#define threshold for maximum length of a setence
max_len=100

#convert text sequences into integer sequences
x_tr_seq = x_tokenizer.texts_to_sequences(x_tr) 
x_val_seq = x_tokenizer.texts_to_sequences(x_val)

#padding up with zero 
x_tr_seq = pad_sequences(x_tr_seq,  padding='post', maxlen=max_len)
x_val_seq = pad_sequences(x_val_seq, padding='post', maxlen=max_len)

#no. of unique words
x_voc_size = x_tokenizer.num_words + 1
x_voc_size

print(x_voc_size)

x_tr_seq[0]

print(x_tr_seq[0])
