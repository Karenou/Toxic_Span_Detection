## Flair Input Representation

Finetune Flair `news-forward` and `news-backward` language models on toxic civil comment dataset.

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

    By running the `train_language_model.py`, the finetuned models will be saved under the subfolder `language_model`. Our finetuned models can be found [here](https://drive.google.com/drive/folders/1-3kLleCwLHZAVcvdujOc37zOIC9-aqD2?usp=sharing).

    ```
    python train_language_model.py --base_path="." --model='news-forward'
    ```
    

 