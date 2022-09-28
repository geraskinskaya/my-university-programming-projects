# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class
import csv

rows = []
with open(r"C:\Users\kgera\Downloads\intro2nlp_assignment1_code (1)\intro2nlp_assignment1_code\experiments\base_model\model_output.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for row in tsv_file:
        rows.append(row)
print(rows)

def prec(TP, FP):
    try:
        prec = TP/(TP+FP)
    except ZeroDivisionError:
        prec = 0
    return prec

def recall(TP, FN):
    try:
        recall = TP/(TP+FN)
    except ZeroDivisionError:
        recall = 0
    return recall

def f1_mes(prec, recall):
    try:
        f1 = 2*prec*recall/(prec+recall)
    except ZeroDivisionError:
        f1 = 0
    return f1

def weighted_f1(n_c, f1_c, n_n, f1_n):
    try:
        w_f1 = (n_c*f1_c + n_n*f1_n)/(n_c+n_n)
    except ZeroDivisionError:
        w_f1 = 0
    return w_f1
correct_c = 0
correct_n = 0
false_c = 0
false_n = 0
for row in rows:
    if len(row)<3:
        continue
    if row[1] == row[2]:
        if row[1] == 'C':
            correct_c +=1
        else:
            correct_n += 1
    else:
        if row[1] == 'C':
            false_n += 1
        else:
            false_c += 1

prec_c = prec(correct_c, false_c)
prec_n = prec(correct_n,false_n)
recall_c = recall(correct_c, false_n)
recall_n = recall(correct_n,false_c)
f1_c = f1_mes(prec_c, recall_c)
f1_n = f1_mes(prec_n, recall_n)
w_f1 = weighted_f1(correct_c+false_c, f1_c, correct_n+false_n, f1_n)
table = [prec_c, prec_n, recall_c, recall_n, f1_c, f1_n, w_f1]
print(table)

