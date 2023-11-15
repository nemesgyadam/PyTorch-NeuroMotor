EEGNet for each subject:
train_CrossRunValidation
cross Run validation for a single subject


conditinoing research (train goes to all selected subjects, last session is validation)
# Baseline
train_BaseLine.ipynb 

# Approach 1 
## Concat conditioning
models/cat_cond_eegnet.py
train_ConCat.ipynb     

# Approach 2
## subjectId conditioning
models/attn_cond_eegnet.py
train_Attn_subjectId.ipynb

# Approach 3
## subjectAverage conditioning
models/attn_cond_eegnet_subjectAverages.py
train_Attn_subjectAverages.ipynb

use the average of all trial of each subject
(subject Encoder is also EEGNet)

# Approach 4
## subjectFeatures conditioning
models/attn_cond_eegnet_subjectFeatures.py
train_Attn_subjectFeatures.ipynb

use the average of all trial of each subject
(subject Encoder is also EEGNet)