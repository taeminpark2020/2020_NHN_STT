import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

kor2vec = torch.load('D:\\2020NHN\kor2vec.pth')

class PCMDataset(Dataset):

    def __init__(self, rootdir):
        self.rootdir= rootdir
        f = open(self.rootdir, 'r')
        raw_txt = f.read()
        f.close()
        raw_txt = raw_txt.split("\n")

        dirs=[]
        answers=[]

        for text in raw_txt:
            text = text.split("\t")
            if text[0] != "":
                try:
                    dirs.append("D:" + "\\" + text[0].replace("..", ""))
                    answers.append(text[1].replace(",", "").replace(".", "").replace("'", "").replace("...","").replace("  "," ").replace("~","").replace("\"", ""))
                except:
                    pass

        self.dirs = dirs
        self.answers = answers
        self.len=len(dirs)

        #print(answers)

    def __getitem__(self, index):

        def word_count(text):
            length = len(text)
            count = 0
            for x in range(length):
                if text[x] == ' ':
                    if x != 0:
                        count = count + 1
                # 마지막 단어 카운드!
                elif x == length - 1:
                    count = count + 1
            return count

        try:
            self.input = np.memmap(self.dirs[index], dtype='h', mode='r')
            tmp = str(self.answers[index])
            for k in range(21-word_count(tmp)):
                tmp = tmp+" eos"
            self.output_vec = kor2vec.embedding(tmp)
        except:
            pass
        return self.input, self.output_vec

    def __len__(self):
        return self.len

