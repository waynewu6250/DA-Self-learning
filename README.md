# DA-Self-learning
Ting-Wei Amazon 2022 intern project

# 1. Related Docs
* [Main project page](https://quip-amazon.com/fbpPAJxiCML4/Project-Overview)
* [Proposal plan](https://quip-amazon.com/6SQQAA7X944a/Proposal-Plan)
* [Sync meeting notes](https://quip-amazon.com/yQ95AKJojUFb/Meeting-notes)
* [Experiment results](https://quip-amazon.com/c594AmOPsDXg/Experiment-results)
* [Experiment discussion](https://quip-amazon.com/qRjhA1JYEqRA/Experiment-discussion)
* [Related work](https://quip-amazon.com/ELNfAL1CCj5j/Paper-survey-Notes)

# 2. Quick Setup

Run
```
sh (your_WS)/src/HyprankModelingShell/hms.sh
sh start.sh
```


# 3. Folder structure

```
DA-Self-Learning/
    └── data/
        ├── dataset.py  (main data pipeline)
    └── model/
    └── trainer/ (train model-based methods)
        ├── trainer.py 
    config.py (configurations)
    augmentor.py (main data generation pipeline)
    evaluator.py (evaluate the baseline system)
    utils.py
    main.py
    start.sh (start at every master/worker node)
    requirements.txt
```

Output folder structure:
```
raw_data/
    └── valid/ (raw data)
    └── wuti_data/ (processed data)
        └── data_name/
            ├── train.gz
            ├── valid.gz
            ├── test.gz
        ...
    vocab.json (raw vocab json)
checkpoints/
    └── model_name/
        └── data_name/
            ├── model.pth
            ├── sample
            ├── vocab.txt 
        ...
    other input json files
```


# 4. Example to run

Train:
>
    python main.py -t train -m autoencoder --num_data=200000
    python main.py -t train -m autoencoder --num_data=1000000 --data-folder 2_5_6_15/ -p _cond
Evaluate:
>
    python main.py -t evaluate -m autoencoder --num_data 200000
Generate:
>
    python main.py -t generate -m autoencoder --num_data 200000 -n 10000
    python main.py -t generate -m autoencoder --num_data 200000 -n 20 -g file
    



Deprecated:
```
trainer_cvae.py
cvae.py
cvae_old.py
```




    
