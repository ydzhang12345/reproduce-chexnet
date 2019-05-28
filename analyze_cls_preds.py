import csv

info = []
nih_correct = 0
nih_total = 0
cheX_correct = 0
cheX_total = 0

with open("preds.csv") as file:
	reader = csv.reader(file, quotechar='|')
	count = 0
	for row in reader:
		if count==0:
			count += 1 
			continue
		if row[0].startswith('NIH'):
			nih_total += 1
			if float(row[1]) > 0.5:
				nih_correct += 1
		if row[0].startswith('CheX'):
			cheX_total += 1
			if float(row[1]) < 0.5:
				cheX_correct += 1

print("NIH: ", nih_correct, nih_total)
print ("NIH: ", nih_correct / nih_total)
print("cheX: ", cheX_correct, cheX_total)
print("cheX: ", cheX_correct / cheX_total )
