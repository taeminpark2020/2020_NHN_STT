import torch
from kor2vec import Kor2Vec

kor2vec = torch.load('kor2vec.pth')

with open('D:\\2020NHN\mapping_9.txt', mode='r+', encoding='utf8') as f:
    raw_txt = f.read()
    raw_txt = raw_txt.split("\n")
    f.close()

tmp = torch.zeros(1802,128)

count = 0
for i in raw_txt:
    try:
        tmp[count,] = kor2vec.embedding(i)[0].float()
        count = count+1
        print(count)

        if count == 1802:
            print("done!")
            torch.save(tmp, 'vectormap_9.pt')
            print("save complete")
            break

    except:
        print("error")


