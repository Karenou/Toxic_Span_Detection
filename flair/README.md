## Flair Input Representation

Finetune Flair `news-forward-fast` and `news-backward-fast` language models on toxic civil comment dataset. The embedding dimension is 1024.

- Required packages and environment
    - flair
    ```
    pip install flair
    ```
    - pytorch

- Dataset
    - Civil comment dataset on [kaggle](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data?select=train.csv)

    We downloaded the train split, which has around 1.8 million comments, and then filtered the toxic comments using criteria: `target > 0`, which gives around 530k comments. We randomly sampled 5k for valid set and 5k for test set, and left the remaining 520k for train set. The split of dataset can be found [here](https://drive.google.com/drive/folders/1-YnBGF5o74YBKQm4_rU8hI2mfUXjIxHY?usp=sharing).

    Download the `corpus` folder and put it under the `flair` folder.

- Language models

    By running the `train_language_model.py`, the finetuned models will be saved under the subfolder `language_model`.

    ```
    python train_language_model.py --base_path="." --model='news-forward-fast'
    ```
- Run baseline + flair input 
    ```
    # forward
    python train_baseline_flair.py --base_path="dataset" --flair_model="flair/language_model/news-forward-fast/best-lm.pt" --early_stopping=1 --pooling="sum" --save_pred_path="pred_results/flair_forward_pred.csv"
    
    # add lstm
    python train_baseline_flair.py --base_path="dataset" --flair_model="flair/language_model/news-forward-fast/best-lm.pt" --early_stopping=2 --pooling="sum" --lstm=True --save_pred_path="pred_results/flair_forward_lstm_pred.csv"

    # add bert avg
    python train_baseline_flair.py --base_path="dataset" --flair_model="flair/language_model/news-forward-fast/best-lm.pt" --early_stopping=2 --pooling="sum" --bert_avg=True --save_pred_path="pred_results/flair_forward_bert_avg_pred.csv"

    # backward    
    python train_baseline_flair.py --base_path="dataset" --flair_model="flair/language_model/news-backward-fast/best-lm.pt" --early_stopping=1 --pooling="mean" --save_pred_path="pred_results/flair_backward_pred.csv"


    ```

 