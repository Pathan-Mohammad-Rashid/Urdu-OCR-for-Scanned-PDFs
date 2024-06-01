# import torch
# import torch.nn as nn

# class CTCLabelConverter(object):
#     # Implementation for CTCLabelConverter

# class NormalizePAD(object):
#     # Implementation for NormalizePAD


import torch
import torch.nn as nn

class CTCLabelConverter(object):
    def __init__(self, character):
        self.character = character
        self.dict = {char: i for i, char in enumerate(character)}

    def encode(self, text):
        length = [len(s) for s in text]
        text = ''.join(text)
        return torch.IntTensor([self.dict[char] for char in text]), torch.IntTensor(length)

    def decode(self, preds, length):
        preds = preds.argmax(2).permute(1, 0).contiguous().view(-1)
        return ''.join([self.character[i] for i in preds if i != 0])

class NormalizePAD(object):
    def __init__(self, size, pad_value=0):
        self.size = size
        self.pad_value = pad_value

    def __call__(self, img):
        img = img.resize(self.size)
        img = torch.from_numpy(np.array(img)).float()
        img = img.div(255)
        return img
