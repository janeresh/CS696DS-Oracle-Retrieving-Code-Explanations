#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd


# In[15]:


cosqa_deepseek_qe_path = '/work/pi_wenlongzhao_umass_edu/27/janet/query_expansion/results/cosqa/cosqa_queries_expanded_deepseek_temp_0.csv'


# In[16]:


df = pd.read_csv(cosqa_deepseek_qe_path)


# In[48]:


df[['id', 'expanded_query']].head()


#  instead of q1 i will rename it as q1.1 q1.2
#  and then create qrels {}
#  and 

# In[18]:


from coir.data_loader import get_tasks
from coir.evaluation import COIR
from coir.models import YourCustomDEModel
import pandas as pd


# In[19]:


def load_data(dataset_name):
    tasks = get_tasks(tasks=[dataset_name])
    return tasks


# In[20]:


dataset_name = "cosqa"


# In[22]:


tasks = load_data(dataset_name)


# In[24]:


corpus, queries, qrels = tasks["cosqa"]


# In[41]:


len(queries)


# In[36]:


qrels['q20105']


# In[1]:


def expand_queries_and_qrels(expanded_df, original_qrels):
    new_queries = {}
    new_qrels = {}

    grouped = expanded_df.groupby('id')

    for qid, group in grouped:
        for idx, row in enumerate(group.itertuples(index=False), start=1):
            if pd.isna(row.expanded_query) or not str(row.expanded_query).strip():
                continue
            new_qid = f"{qid}.{idx}"
            new_queries[new_qid] = row.expanded_query

            if qid in original_qrels:
                new_qrels[new_qid] = original_qrels[qid]

    return new_queries, new_qrels


# In[38]:


new_queries, new_qrels = expand_queries_and_qrels(df, qrels)


# In[51]:


df.loc[df['id'] == 'q20108']


# In[47]:


new_queries['q20105.5']


# In[52]:


new_qrels['q20105.5']


# In[ ]:




