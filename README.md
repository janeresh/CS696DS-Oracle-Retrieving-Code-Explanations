# CS696DS-Oracle-Retrieving-Code-Explanations  
## Retrieving Code Explanations for Improved Code Search  
### Group 27  

---

## Results Naming Convention  
Results are stored using the following structure:  
```
/{dataset_name}/{llm_name} or "baseline"/{retrieval_method}/{model_name}
```

---

## Running Baseline Model  
To run the baseline model, open [`simple-usage-modularised.ipynb`](coir-main/simple-usage-modularised.ipynb) and follow these steps:  

1. **Run the imports**.  
2. **Load the dataset** by calling:  
   ```python
   load_data(dataset_name)
    ```
3. **Run the pipeline** by calling:  
    ```python
   run(encoder_name, tasks, "baseline", retrieval_method, dataset_name)
    ```
---

## Using Explanations Instead of Code
To run the baseline model, open [`simple-usage-modularised.ipynb`](coir-main/simple-usage-modularised.ipynb) and follow these steps: 
1. **Run the imports**.  
2. **Load the dataset** by calling:  
   ```python
   load_data(dataset_name)
    ```
3. **Add explanations to data** by calling:  
   ```python
   add_expl(tasks, explanations_df_path, col_name)
    ```
4. **Run the pipeline** by calling:  
    ```python
   run(encoder_name, tasks, llm_name, retrieval_method, dataset_name)
    ```

### Running bm25
1. **Modify [`evaluation.py`](coir-main/coir/beir/retrieval/evaluation.py)** : change the ```search``` on line 27 to ```search_bm25 ```
2. **Run the imports**.  
3. **Load the dataset** by calling:  
   ```python
   load_data(dataset_name)
    ```
4. **Add explanations to data** by calling:  
   ```python
   add_expl(tasks, explanations_df_path)
    ```
5. **Run the pipeline** by calling:  
    ```python
   run(encoder_name, tasks, llm_name, "bm25", dataset_name)
    ```
