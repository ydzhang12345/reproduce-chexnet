import csv
import pdb
import numpy as np

rows = []




'''
### first lets remove Lateral data  -> left with 191028 images
## for train.csv
with open('mimic_valid.csv') as file:
	reader = csv.reader(file, quotechar='|')
	count = 0
	for row in reader:
		if count==0:
			rows.append(row)
			count +=1 
			continue
		if row[1]=='lateral' or row[1][0]=='L':
			continue
		else:
			rows.append(row)

with open('mimic_valid_fr.csv', 'w', newline='') as file:
	writer = csv.writer(file, quotechar='|', quoting=csv.QUOTE_MINIMAL)
	for i in range(len(rows)):
		writer.writerow(rows[i])

pdb.set_trace()
'''


'''
### first lets remove Lateral data 
## for valid.csv
with open('valid.csv') as file:
	reader = csv.reader(file, quotechar='|')
	count = 0
	for row in reader:
		if count==0:
			rows.append(row)
			count +=1 
			continue
		if row[3]=='Lateral' or row[3][0]=='L':
			continue
		else:
			rows.append(row)

with open('valid_fr.csv', 'w', newline='') as file:
	writer = csv.writer(file, quotechar='|', quoting=csv.QUOTE_MINIMAL)
	for i in range(len(rows)):
		writer.writerow(rows[i])
'''


# --------------------------------------------
# next sample patients from train set so as to get ~10% data for validation
patient_list = []
patient_dict = {}
patient_count = 0
total_imgs = 0
with open('mimic_train_fr.csv') as file:
	reader = csv.reader(file, quotechar='|')
	count = 0
	for row in reader:
		if count==0:
			rows.append(row)
			count +=1 
			continue
		patient_id = row[0].split('/')[1]
		if patient_id not in patient_list:
			patient_list.append(patient_id)
			patient_count += 1
			patient_dict[patient_id] = 1
		else:
			patient_dict[patient_id] += 1
		total_imgs += 1


np.random.seed(2019)
sample_patient = (np.random.sample(int(patient_count*0.36)) * patient_count).astype(np.int32)

# count validation samples
valid_patient = []
valid_count = 0
test_patient = []
test_count = 0

for i in sample_patient:
	patient_id = patient_list[i]
	if np.random.random() < 0.33:
		valid_count += patient_dict[patient_id]
		valid_patient.append(patient_id)
	else:
		test_count += patient_dict[patient_id]
		test_patient.append(patient_id)

#pdb.set_trace()

# now build nih-like labels
info = {}
entry = []
no_finding_count = 0
total = 0
with open('mimic_train_fr.csv') as file:
	reader = csv.reader(file, quotechar='|')
	count = 0
	for row in reader:
		if count==0:
			count +=1 
			continue
		# Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion
		temp = [row[10], row[4], row[8], row[7], row[12]]
		temp_vec = []
		for t in temp:
			if t=='':
				temp_vec.append(0)
			else:
				temp_vec.append(int(float(t)))
		disease_vec = np.zeros([len(temp_vec)])
		if temp_vec[0]==-1 or temp_vec[0]==1:
			disease_vec[0] = 1
		else:
			disease_vec[0] = 0
		if temp_vec[1]==-1 or temp_vec[1]==0:
			disease_vec[1] = 0
		else:
			disease_vec[1] = 1
		if temp_vec[2]==-1 or temp_vec[2]==0:
			disease_vec[2] = 0
		else:
			disease_vec[2] = 1
		if temp_vec[3]==-1 or temp_vec[3]==1:
			disease_vec[3] = 1
		else:
			disease_vec[3] = 0
		if temp_vec[4]==-1 or temp_vec[4]==0:
			disease_vec[4] = 0
		else:
			disease_vec[4] = 1
		if np.sum(disease_vec)==0:
			no_finding_count += 1
		info[row[0]] = {'Patient Gender': 'N/A', 'Patient Age': 'N/A', 'disease_vec': disease_vec,  
						'Image Index':row[0], 'Patient ID': row[0].split('/')[1], 'view': row[1]}
		entry.append(row[0])
		total += 1



'''
# deal with gt validation labels
info_valid = {}
entry_valid = []
with open('valid_fr.csv') as file:
	reader = csv.reader(file, quotechar='|')
	count = 0
	for row in reader:
		if count==0:
			count +=1 
			continue
		# Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion
		temp = [row[13], row[7], row[11], row[10], row[15]]
		temp_vec = []
		for t in temp:
			if t=='':
				temp_vec.append(0)
			else:
				temp_vec.append(int(float(t)))
		disease_vec = np.zeros([len(temp_vec)])
		if temp_vec[0]==-1 or temp_vec[0]==1:
			disease_vec[0] = 1
		else:
			disease_vec[0] = 0
		if temp_vec[1]==-1 or temp_vec[1]==0:
			disease_vec[1] = 0
		else:
			disease_vec[1] = 1
		if temp_vec[2]==-1 or temp_vec[2]==0:
			disease_vec[2] = 0
		else:
			disease_vec[2] = 1
		if temp_vec[3]==-1 or temp_vec[3]==1:
			disease_vec[3] = 1
		else:
			disease_vec[3] = 0
		if temp_vec[4]==-1 or temp_vec[4]==0:
			disease_vec[4] = 0
		else:
			disease_vec[4] = 1
		info_valid[row[0]] = {'gender': row[1], 'age': row[2], 'disease_vec': disease_vec, 'Follow-up #':'N/A', 
						'Image Index':row[0], 'Patient ID': row[0].split('/')[2], 'view': row[4]}
		entry_valid.append(row[0])
'''


# write new csv file
with open('mimic_full_train.csv', 'w', newline='') as file:
	writer = csv.writer(file, quotechar='|', quoting=csv.QUOTE_MINIMAL)
	writer.writerow(['Image Index', 'Dataset ID', 'Patient Age', 'Patient Gender', 'View Position', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'fold'])
	for e in entry:
		row = [info[e]['Image Index'], 2, info[e]['Patient Age'], info[e]['Patient Gender'], info[e]['view']]
		for i in info[e]['disease_vec']:
			row.append(int(i))
		
		if e.split('/')[1] in valid_patient:
			row.append('val')
		elif e.split('/')[1] in test_patient:
			row.append('test')
		else:
			row.append('train')
		
		#row.append('train')
		writer.writerow(row)
		#pdb.set_trace()

	'''
	for e in entry_valid:
		row = [info_valid[e]['Image Index'], info_valid[e]['Follow-up #'], info_valid[e]['Patient ID'], info_valid[e]['age'], info_valid[e]['gender'], info_valid[e]['view']]
		for i in info_valid[e]['disease_vec']:
			row.append(int(i))
		row.append('val')
		writer.writerow(row)
	'''



pdb.set_trace()
a = 1		






