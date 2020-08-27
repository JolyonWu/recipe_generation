class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        return self.idx


    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<pad>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def return_word(self,idx):
        return self.idx2word[idx]




def clean_data(text):
    replace_dict = {'and': ['&', "'n"], '': ['%', ',', '.','(',')','+', '#', '[', ']', '!', '?','/','-']}
    for rep, char_list in replace_dict.items():
        for c_ in char_list:
            if c_ in text:
                text = text.replace(c_, rep)
    text=text.strip()
    text=text.lower()
    return text