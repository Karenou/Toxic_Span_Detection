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
