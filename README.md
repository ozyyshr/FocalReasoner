# FocalReasoner

This is the code for our paper "Fact-driven Logical Reasoning for Machine Reading Comprehension".

#### Environment

- python=3.6
- pytorch=1.6.0
- dgl=0.6.1
- transformers=4.3.2



#### How to run?

Directly run the following code:

```bash
bash scripts/run_roberta_large.sh
```

The accuracies on the dev set are stored in the folder "Checkpoints", with test results stored in "test_pred.npy"



Can change the dataset directory in the scripts to run different tasks. For example, to run logiQA, set 

```BASH
RECLOR_DIR=logiQA_data
TASK_NAME=logiqa
```


