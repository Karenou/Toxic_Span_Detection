## README

- FastText
    - Download word vectors from online, put under `fast_text` folder
    ```
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
    unzip crawl-300d-2M.vec.zip
    ```

    - For out-of-vocabulary tokens, we use the BytePairEmbeddings language model from Flair
    - Run the `get_embedding.py`, the fasttext embeddings for train, trial and test set will be saved in `fast_text` folder in h5 format
    ```
    python fast_text/get_embedding.py --data_path="dataset" --model_path="fast_text/crawl-300d-2M.vec" --save_path="fast_text"
    ```

    - Run baseline + fast text only
    ```
    python train_baseline_ft.py --base_path="dataset" --fast_text_path="fast_text" --pooling="sum" --early_stopping=1 --save_pred_path="pred_results/ft_pred.csv"
    ```
    
    - Run baseline + fast text + flair
    ```
    python train_baseline_ft_flair.py --base_path="dataset" --flair_model="flair/language_model/news-backward-fast/best-lm.pt" --fast_text_path="fast_text" --pooling="mean" --early_stopping=1 --save_pred_path="pred_results/ft_flair_backward_pred.csv"
    ```
