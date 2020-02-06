import pandas as pd
import numpy as np
import string
import re
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv('IMDB Dataset.csv')
df.columns = ["reviews" , "sentiments"]

reviews = df["reviews"].values.tolist()
sentiments = df["sentiments"].values.tolist()

clean_reviews = []
clean_reviews = [review.lower() for review in reviews]
clean_reviews = [re.sub("<br />", '', review) for review in clean_reviews]
clean_reviews = [re.sub(r'[0-9]', '', review) for review in clean_reviews]
clean_reviews = [re.sub(r'[^\w]', ' ', review) for review in clean_reviews]

num_reviews = len(clean_reviews)
word_list = ' '.join(clean_reviews)
count_of_words = Counter(word_list.split(" "))
total_no_of_words = len(word_list)
sorted_words = count_of_words.most_common(total_no_of_words)

vocabulary = {word: i+1 for i, (word, c) in enumerate(sorted_words)}

integer_reviews = []

for review in clean_reviews:
    r = [vocabulary[w] for w in review.split(" ")]
    integer_reviews.append(r)

integer_sentiments = [1 if sentiment == 'positive' else 0 for sentiment in sentiments]
review_size = [len(m) for m in integer_reviews]

#remove too short or too long reviews
integer_reviews = [integer_reviews[i] for i, l in enumerate(review_size) if l > 0]
integer_sentiments = [integer_sentiments[i] for i, l in enumerate(review_size) if l > 0]

def adjust_padding(integer_reviews, seq_length):
    attributes = np.zeros((len(integer_reviews), seq_length), dtype=int)
    for i, review in enumerate(integer_reviews):
        review_size = len(review)
        if review_size <= seq_length:
            zeroes = list(np.zeros(seq_length - review_size))
            new = zeroes + review
        elif review_size > seq_length:
            new = review[0:seq_length]
        attributes[i, :] = np.array(new)
    return attributes


integer_reviews = adjust_padding(integer_reviews, 500)

pd.Series(review_size).hist()
plt.xlabel("<-------No of words-------->")
plt.ylabel("<-------No of reviews------->")
plt.show()
pd.Series(review_size).describe()
