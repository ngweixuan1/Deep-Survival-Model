import os
import pickle
import random
random.seed(1)
import pandas as pd
import math
import random
import numpy as np

'''
Input: CSVs of labels, events, and cohort files from Bigquery
Output: Train, test, validation split for labels and features
'''

events_filepath = "../data/patient_chartevents_final.csv"
cohort = pd.read_csv('../data/patients_data_final.csv')
labels = pd.read_csv("../data/label.csv")

def read_events_file(events_filepath):
  '''
  Output: pivoted table with keys as subject_id, record date in order for input into dataloader
  '''
  events = pd.read_csv(events_filepath, index_col=0)
  # read the events file and pivot it based on feature names and its value
  events_pivoted = events.pivot(index=['subject_id', 'record_date'], columns=['feature_name'], values=['value']).sort_values( 'record_date').fillna(0)
  events_pivoted.columns = events_pivoted.columns.droplevel()
  events_pivoted = events_pivoted.reset_index()
  id_list = list(events_pivoted['subject_id'].unique())
  temp_events_list = []

  for id in id_list:
    temp_events = events_pivoted[events_pivoted['subject_id']==id]
    # Period column enables identification of sequences. We ignore ignore date but are concerned about the sequence of readings taken
    temp_events['period'] = list(range(temp_events.shape[0]))
    temp_events_list.append(temp_events)
  final_events = pd.concat(temp_events_list)
  return final_events

final_events =  read_events_file(events_filepath)

# Filter based on cohort query
final_events_filtered = final_events.merge(cohort[['subject_id', 'first_admittime', 'end_time_icu']], how ='inner', on='subject_id')
final_events2 = final_events_filtered.copy()

# Merging labels with the dataset
final_events3 = final_events2.merge(labels, left_on = 'subject_id', right_on = 'SUBJECT_ID', how = 'left')
#Convert labels from dates to time to death, measured by end time of icu
final_events3['DOD'] = pd.to_datetime(final_events3['DOD'], errors='coerce')
final_events3['end_time_icu'] = pd.to_datetime(final_events3['end_time_icu'],  errors='coerce')
final_events3['num_days1'] = (final_events3['DOD'] - final_events3['end_time_icu']).dt.days +1
max_days= final_events3['num_days1'].max()
# Label features with right censoring
final_events3['num_days'] = final_events3['num_days1'].apply(lambda x: max_days if ((x>max_days) or math.isnan(x)) else x)
final_events3['mortality'] = final_events3['num_days'].apply(lambda x: 0.0 if x==max_days else 1.0)
final_events3 = final_events3[final_events3['num_days']>=0]

# Train test split based on IDs
id_list = list(final_events3['subject_id'].unique())
train_list = random.sample(id_list, k=int(0.7*len(id_list)))
test_vali_list = list(set(id_list) - set(train_list))
vali_list = random.sample(test_vali_list, k=int(0.66*len(test_vali_list)))
test_list = list(set(test_vali_list) - set(vali_list))

final_labels = final_events3[['subject_id', 'num_days', 'mortality']].groupby('subject_id').min().reset_index().sort_values(['subject_id'])

# split Labels
train_labels = final_labels[final_labels['subject_id'].isin(train_list)].sort_values(by = ['subject_id'], axis=0)
valid_labels = final_labels[final_labels['subject_id'].isin(vali_list)].sort_values(by = ['subject_id'], axis=0)
test_labels = final_labels[final_labels['subject_id'].isin(test_list)].sort_values(by = ['subject_id'], axis=0)

train_labels.to_csv("../data/train_labels.csv")
valid_labels.to_csv("../data/valid_labels.csv")
test_labels.to_csv("../data/test_labels.csv")

# split features
train_seqs = final_events3[final_events3['subject_id'].isin(train_list)].sort_values(['subject_id', 'period'], axis=0)
valid_seqs = final_events3[final_events3['subject_id'].isin(vali_list)].sort_values(['subject_id', 'period'], axis=0)
test_seqs = final_events3[final_events3['subject_id'].isin(test_list)].sort_values(['subject_id', 'period'], axis=0)
train_seqs.to_csv("../data/train_seqs.csv")
valid_seqs.to_csv("../data/valid_seqs.csv")
test_seqs.to_csv("../data/test_seqs.csv")