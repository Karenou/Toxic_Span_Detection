

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