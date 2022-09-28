import matplotlib.pyplot as plt
import csv

epoch_value = [10, 15, 20, 25, 30, 35, 40, 45, 50, 100]
weighted_f1 = [0.8598244160826585, 0.8598244160826585, 0.8685809016619308, 0.8732183384550146, 0.8732183384550146, 0.8732183384550146, 0.8732183384550146, 0.8732183384550146, 0.8732183384550146, 0.8732183384550146]

plt.plot(epoch_value, weighted_f1)
plt.title('Weighted F1 change with increase of number of epochs ')
plt.xlabel('n of epochs')
plt.ylabel('weighted f1')
plt.show()


rows10 = []
with open(r"C:\Users\kgera\Downloads\intro2nlp_assignment1_code (1)\intro2nlp_assignment1_code\experiments\base_model\model_output10.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for row in tsv_file:
        rows10.append(row)

rows15 = []
with open(r"C:\Users\kgera\Downloads\intro2nlp_assignment1_code (1)\intro2nlp_assignment1_code\experiments\base_model\model_output15.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for row in tsv_file:
        rows15.append(row)

rows20 = []
with open(r"C:\Users\kgera\Downloads\intro2nlp_assignment1_code (1)\intro2nlp_assignment1_code\experiments\base_model\model_output20.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for row in tsv_file:
        rows20.append(row)


for r in range(len(rows10)):
    if len(rows10[r]) != 3:
        continue

    if rows10[r][2] != rows20[r][2] or rows10[r][2] != rows15[r][2]:
        print(rows10[r])
        print(rows15[r])
        print(rows20[r])





