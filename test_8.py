import torch.nn as nn
import numpy as np
import torch
import stt_model_8
import os

#Please, batch size only one!
batch_size =1

print("batch_size :",batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

speak2embed = stt_model_8.Speak2Embed().to(device)
loss = nn.SmoothL1Loss(reduction='sum')
checkpoint = torch.load("D:\\2020NHN\s2e_train_8_13.pth")
speak2embed.load_state_dict(checkpoint['model_state_dict'])
speak2embed.eval()
print("neural net load complete.")

vectormap = torch.load('vectormap_9.pt')
print("vectormap load complete.",vectormap.size())

with open('D:\\2020NHN\mapping_9.txt', mode='r+', encoding='utf8') as f:
    vector2word = f.read()
    vector2word = vector2word.split("\n")
    f.close()

#root directory of Test folder : 테스트 폴더 이름명에 따라 경로 수정 될 수 있음.
first_folder = 'D:\\TEST'

second_folder = os.listdir(first_folder)

for second_idx in second_folder:
    third_folder = os.listdir(first_folder+"\\"+second_idx)

    for third_idx in third_folder:
        file_names = os.listdir(first_folder+"\\"+second_idx+"\\"+third_idx)

        for file_name in file_names:
            with torch.no_grad():
                test_input = torch.tensor(np.memmap(first_folder + "\\" + second_idx+"\\"+third_idx+"\\"+file_name, dtype='h', mode='r')).view(1, -1)
                test_output = speak2embed(test_input.to(device=device, dtype=torch.float), batch_size)
                test_output = test_output.permute(0, 2, 1)
                pred = test_output.view(-1, 128)

            answer = ""

            try:
                for i in range(pred.size(0)):
                    min_index=0
                    min = loss(pred[i],vectormap[0].cuda())
                    for j in range(1,1802):
                        tmp_min = loss(pred[i],vectormap[j].cuda())
                        if tmp_min<min:
                            min = tmp_min
                            min_index = j
                    if (vector2word[min_index] == "eof"):
                        answer = answer+""
                    else:
                        answer = answer + vector2word[min_index] + " "
            except:
                answer = "문장을 만들지 못했습니다"

            try:
                final_file_name = file_name.replace(".PCM", "")
                os.makedirs("D:\\2020NHN\\test_result\\"+second_idx+"\\"+third_idx)
                f = open("D:\\2020NHN\\test_result\\"+second_idx+"\\"+third_idx+"\\"+final_file_name+".txt", 'w', encoding="UTF8")
                f.write(answer)
                f.close()
                print("D:\\2020NHN\\test_result\\" + second_idx + "\\" + third_idx + "\\" + final_file_name + ".txt",
                      "파일이 생성되었습니다.")
            except:
                print(".txt파일을 생성하지 못했습니다.")