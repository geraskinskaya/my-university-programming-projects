#part A
import spacy
import textacy
from collections import Counter
from spacy import displacy
import csv
import statistics as st
from statistics import mean
import numpy as np
import wordfreq
from matplotlib import pyplot
import seaborn as sns


nlp = spacy.load("en_core_web_sm") # or a different available model
#spaCy counts the number of tokens such as
# words, numbers, puntuation marks etc.

file = "data/preprocessed/train/sentences.txt"
with open(file, 'r', encoding='utf-8') as f:
    newline_break =" "
    for readline in f:
        line_strip = readline.rstrip()
        newline_break+= (line_strip + ' ')
print(newline_break)
input = newline_break

doc = nlp(input)

#(1) number of token   FUNCTIONAL
n_words = (len(doc))
print(n_words)
#words = [token.text for token in doc]
#print(words)
##this for loop is to count how many words are in the text
#counter = 0
#for word in words:
#    counter += 1
#print(counter)

#(1) Number of types FUNCTIONAL
out = []
seen = set()
for word in doc:
    if word.text.lower() not in seen:
        out.append(word.text.lower())
    seen.add(word.text)
# now out has "unique" tokens
num_types = len(out)
print(num_types)
print(out)

#(1) avergae words per sentence
sentences = [[]]
ends = set(".?!")

for sentence in (list(doc.sents)):
    words = []
    for word in sentence:
        words.append(word)
    for word in words:
        if word in ends: sentences.append([])
        else: sentences[-1].append(word)

if sentences[0]:
    if not sentences[-1]: sentences.pop()
    print("average sentence length:", sum(len(s) for s in sentences)/len(sentences))


#(1) FUNCTION FOR AVERAGE      NEEDED
def average(lst):
    return sum(lst) / len(lst)


word_frequencies = []
sentence_len = []
word_len = []

for sentence in doc.sents:
    words = []
    for token in sentence:
        # Let's filter out punctuation
        if not token.is_punct:
            words.append(token.text)

    for word in words:
        word_len.append(len(word))

    sentence_len.append(len(words))

print(sum(sentence_len))  # n of words
print(average(sentence_len))  # words per sentence
print(average(word_len))  # word lenght

poss = []
for token in doc:
    poss.append(token.pos_)
# print(poss)

counted = Counter(poss)

most_ten = counted.most_common(10)

for item in most_ten:
    print(item)

#(2) Relative Tag Frequency (%)
sum_overall = sum(counted.values())
print(sum_overall)

for item in most_ten:
    print(round(item[1]/sum_overall,2))

#(2) 3 most common and 1 infrequent
list_of_lists = []
for pos in most_ten:
    list_for_LOL = []
    for token in doc:
        if token.pos_ == pos[0]:
            t = str(token).lower()
            list_for_LOL.append(t)
        else:
            pass
    list_of_lists.append(list_for_LOL)
    Count = Counter(list_for_LOL)
    print(Count.most_common(3))
    print(Count.most_common(50)[-1])

#(3) token bigrams
bigrams = list(textacy.extract.basics.ngrams(doc, 2, filter_stops = False, filter_punct = True, filter_nums = False, min_freq = 0))
#print(bigrams)
list_a = []
for item in bigrams:
    list_a.append(str(item).lower())
c = Counter(list_a)
#print(c)
most_three = c.most_common(3)

print(most_three)

#(3)token trigrams
trigrams = list(textacy.extract.basics.ngrams(doc, 3, filter_stops = False, filter_punct = True, filter_nums = False, min_freq = 0))
list_a = []
for item in trigrams:
    list_a.append(str(item).lower())
c = Counter(list_a)
#print(c)
most_three = c.most_common(3)

print(most_three)


#(3) pos bigrams
new_list = []
for i in range(len(poss)):
    new_item = poss[i]+' ' + poss[i+1]
    new_list.append(new_item)
    if i == len(poss)-2:
        break

print(new_list)

count_bi_pos = Counter(new_list)
most_three = count_bi_pos.most_common(3)

print(most_three)

#(3) pos trigrams
new_list = []
for i in range(len(poss)):
    new_item = poss[i]+' ' + poss[i+1] + ' ' + poss[i+2]
    new_list.append(new_item)
    if i == len(poss)-3:
        break

print(new_list)

count_tri_pos = Counter(new_list)
most_three = count_tri_pos.most_common(3)

print(most_three)

#(4)   COMPLETED below
list_tok_lemma = []
for token in doc:
    list_tok_lemma.append([token.text, token.lemma_])
print(list_tok_lemma)


#(4) Lemma, inflected form
infl_forms = set()
for item in list_tok_lemma:
    if item[1] == 'be':
        infl_forms.add(item[0])
print(infl_forms)


for form in infl_forms:
    for sentence in doc.sents:
        if form in str(sentence).lower():
            print(form + ':')
            print('->' + str(sentence))
            break

# (5)
list_lab = []
for ent in doc.ents:
    if ent.label_ not in list_lab:
        list_lab.append(ent.label_)

print(len(list_lab))  # number of diff entities
print(len(doc.ents))  # number of entities

#names of entities
print(list_lab)

# Displacy provides nice visualizations of spaCy annotations https://spacy.io/usage/visualizers

displacy.render(doc, jupyter=True, style='ent')

#PART B

target = []
binary_label = []
probabilistic_label = []

with open("data/original/english/WikiNews_Train.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        target.append(line[4])
        binary_label.append(line[9])
        probabilistic_label.append(line[10])
"""
print(target)
print(binary_label)
print(probabilistic_label)
"""

token_b=[]
for i in range(len(target)):
    token_b.append(nlp(target[i]))

# (7) Min, max, median, mean, and stdev of the probabilistic label:
for i in range(len(probabilistic_label)):
    probabilistic_label[i] = float(probabilistic_label[i])

print(st.mean(probabilistic_label))  # mean
print(max(probabilistic_label))  # max
print(np.std(probabilistic_label))  # std
print(min(probabilistic_label))  # min

# (7) instances with 0 and 1

counter0 = 0
counter1 = 1

for i in binary_label:
    if i == '0':
        counter0 += 1
    else:
        counter1 += 1

print(counter0)  # instances 0
print(counter1)  # instances 1


#(7) count instances with more than one token
    # also max num of tokens per instance

counter = 0
counter_of_counts = []
for i in token_b:
    if len(i) > 1:
        counter+=1
    counter_of_counts.append(len(i))

print(counter) #more than one
#print(counter_of_counts) #tokens per instance
print(max(counter_of_counts)) #max tokens per instance

token_length = []
token_freq = []
simple_list = []
pos_new = []
new_pl = []
new_target = []
for i in range(len(target)):
    if binary_label[i] == '1' and len(token_b[i]) == 1:
        simple_list.append(i)
        for token in token_b[i]:
            pos_new.append(token.pos_)
        new_pl.append(float(probabilistic_label[i]))
        new_target.append(target[i])
    else:
        pass

print(simple_list)

for i in simple_list:
    token_length.append(len(target[i]))
    token_freq.append(wordfreq.word_frequency(target[i], 'en'))

print(token_freq) #frequency
print(pos_new)
print(token_length) #token lenght as char
print(new_pl)

pear_pl_tl = np.corrcoef(new_pl, token_length)
pear_pl_tf = np.corrcoef(new_pl, token_freq)

print(pear_pl_tl)
print(pear_pl_tf)


#(8) lenght and complexity
sns.scatterplot(x=token_length,y=new_pl,)

#(8) frequency and complexity
sns.scatterplot(x=token_freq,y=new_pl,)

#(8) pos and complexity
sns.scatterplot(x=pos_new,y=new_pl,)












