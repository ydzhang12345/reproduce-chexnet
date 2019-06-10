import csv

bank = []
with open('./cheXpert_labels.csv') as file:
	reader = csv.reader(file, quotechar='|')
	count = 0
	for row in reader:
		if count==0:
			count += 1
			continue
		bank.append([row[0], 1, row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11]])


with open('./nih_labels.csv') as file:
	reader = csv.reader(file, quotechar='|')
	count = 0
	for row in reader:
		if count==0:
			count += 1
			continue
		bank.append(['NIH/images/' + row[0].replace('.png', '.jpg'), 0, row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11]])


with open('test.csv', 'w', newline='') as file:
	writer = csv.writer(file, quotechar='|', quoting=csv.QUOTE_MINIMAL)
	writer.writerow(['Image Index', 'Dataset ID', 'Patient Age', 'Patient Gender', 'View Position', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'fold'])
	for i in range(len(bank)):
		writer.writerow(bank[i])