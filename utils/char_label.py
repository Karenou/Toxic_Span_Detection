import pandas as pd

def get_char_level_label(data):
    """
    convert token-level label to character-level
    """
    labels = []
    length = data["length"]
    tags = data["tags"][1:length]
    offset = data["offset"]
    words = data["words"]
    sentence = data["sentence"]
    index = 0
    bias = 0

    for i in range(length - 1):
        if offset[i+1][0] == 0:
            subword = words[index][offset[i+1][0]:offset[i+1][1]]
        elif offset[i+1][0] > 0:
            subword = words[index][offset[i+1][0]:offset[i+1][1]]
        if offset[i+1][1] == len(words[index]):
            index = index + 1
        elif offset[i+1][1] > len(words[index]):
            print("error")

        bias = sentence.find(subword) + bias

        if tags[i] == "I":
            pos = bias
            for i in range(len(subword)):
                labels.append(pos)
                pos = pos + 1

        bias = bias + len(subword)
        sentence = sentence[sentence.find(subword)+len(subword):]

    return labels

def get_roberta_char_level_label(data):
    """
    convert token-level label to character-level for roberta
    """
    labels = []
    length = data["length"]
    tags = data["tags"][1:length]
    offset = data["offset"]
    words = data["words"]
    sentence = data["sentence"]
    index = 0
    bias = 0

    for i in range(length - 1):
        try:
            if i != 0 and offset[i + 1][0] == 0:
                continue
            if offset[i + 1][0] == 1:
                subword = words[index][offset[i + 1][0] - 1:offset[i + 1][1]]
            else:
                subword = words[index][offset[i + 1][0]:offset[i + 1][1]]
            if offset[i + 1][1] == len(words[index]):
                index = index + 1
            elif offset[i + 1][1] > len(words[index]):
                print("error")
            bias = sentence.find(subword) + bias

            if tags[i] == "I":
                pos = bias
                for i in range(len(subword)):
                    labels.append(pos)
                    pos = pos + 1

            bias = bias + len(subword)
            sentence = sentence[sentence.find(subword) + len(subword):]
        except Exception as e:
            pass
            # print("excpetion in char:", e)

    return labels

def word_to_char_level_label(data):
    """
    the tags are word-level prediction
    """
    labels = []
    mapping = data["mapping"]
    words = data["words"]
    sentence = data["sentence"]
    tags = data["tags"]
    bias = 0

    for i in range(len(mapping)):
        if tags[i] == "I":
            word = words[i]
            bias = sentence.find(word) + bias

            pos = bias
            for _ in range(len(word)):
                labels.append(pos)
                pos += 1
            
            bias += len(word)
            sentence = sentence[sentence.find(word) + len(word):]
    
    return labels

def save_pred_label(pred_label, save_path):
    """
    save predicted label to csv
    @param perd_label: List[List[int]]
    @param save_path: path of csv
    """
    # convert to list of str
    pred_label_str = ["[" + ", ".join(map(str, label)) + "]" for label in pred_label]
    pred_df = pd.DataFrame(pred_label_str, columns=["pred_label"])
    pred_df.to_csv(save_path, index=False, header=True)
