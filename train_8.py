import pcm_dataloader_8
import stt_model_8
import torch.optim as optim
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=250
print(device,",", "batch_size :",batch_size)

def make_batch(samples):
    inputs = [torch.tensor(x) for x, y in samples]
    answer_vec = [y for x, y in samples]
    answers_vec = torch.zeros(batch_size,21,128)
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs,batch_first=True)
    for i in range(batch_size):
        answers_vec[i] = torch.tensor(answer_vec[i])
    return  padded_inputs,answers_vec

speak2embed = stt_model_8.Speak2Embed().to(device)
critierion_s2e = torch.nn.SmoothL1Loss(reduction='sum')
optimizer = optim.Adam(speak2embed.parameters(), lr=0.005)

checkpoint = torch.load("D:\\2020NHN\s2e_train_8_26.pth")
speak2embed.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epochs = checkpoint['epoch']
loss = checkpoint['loss']

speak2embed.train()

for epoch in range(epochs+1,1000):

    running_loss = 0.0
    count =0

#1. 성인 : "D:\\2020_자유대화_Hackarthon_학습DB\\001.일반남녀\\000.PCM2TEXT\\2020_일반남녀_학습DB_PCM2TEXT.list"
    dataset = pcm_dataloader_8.PCMDataset(
        rootdir="D:\\2020_자유대화_Hackarthon_학습DB\\001.일반남녀\\000.PCM2TEXT\\2020_일반남녀_학습DB_PCM2TEXT.list")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=make_batch)

    for batch_input, batch_answer_embeding in dataloader:
        count +=1
        if count == 51:
            break
        optimizer.zero_grad()
        batch_output = speak2embed(batch_input.to(device), batch_size)
        embeding_output = batch_output.permute(0,2,1)
        loss = critierion_s2e(embeding_output, batch_answer_embeding.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print("count:", count, "/200", ", loss:", loss)

#2. 노인 : "D:\\2020_자유대화_Hackarthon_학습DB\\002.노인남녀(시니어)\\000.PCM2TEXT\\2020_시니어_학습DB_PCM2TEXT.list"
    dataset = pcm_dataloader_8.PCMDataset(
        rootdir="D:\\2020_자유대화_Hackarthon_학습DB\\002.노인남녀(시니어)\\000.PCM2TEXT\\2020_시니어_학습DB_PCM2TEXT.list")

    for batch_input, batch_answer_embeding in dataloader:
        count += 1
        if count == 101:
            break
        optimizer.zero_grad()
        batch_output = speak2embed(batch_input.to(device), batch_size)
        embeding_output = batch_output.permute(0, 2, 1)
        loss = critierion_s2e(embeding_output, batch_answer_embeding.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print("count:",count,"/200",", loss:",loss)

#3. 어린이 : "D:\\2020_자유대화_Hackarthon_학습DB\\003.소아남녀\\000.PCM2TEXT\\2020_소아남녀_학습DB_PCM2TEXT.list"
    dataset = pcm_dataloader_8.PCMDataset(
        rootdir="D:\\2020_자유대화_Hackarthon_학습DB\\003.소아남녀\\000.PCM2TEXT\\2020_소아남녀_학습DB_PCM2TEXT.list")

    for batch_input, batch_answer_embeding in dataloader:
        count += 1
        if count == 151:
            break
        optimizer.zero_grad()
        batch_output = speak2embed(batch_input.to(device), batch_size)
        embeding_output = batch_output.permute(0, 2, 1)
        loss = critierion_s2e(embeding_output, batch_answer_embeding.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print("count:", count, "/200", ", loss:", loss)

#4. 외래어 : "D:\\2020_자유대화_Hackarthon_학습DB\\004.외래어\\000.PCM2TEXT\\2020_외래어_학습DB_PCM2TEXT.list"
    dataset = pcm_dataloader_8.PCMDataset(
        rootdir="D:\\2020_자유대화_Hackarthon_학습DB\\004.외래어\\000.PCM2TEXT\\2020_외래어_학습DB_PCM2TEXT.list")

    for batch_input, batch_answer_embeding in dataloader:
        count += 1
        if count == 201:
            break
        optimizer.zero_grad()
        batch_output = speak2embed(batch_input.to(device), batch_size)
        embeding_output = batch_output.permute(0, 2, 1)
        loss = critierion_s2e(embeding_output, batch_answer_embeding.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print("count:", count, "/200", ", loss:", loss)

    #result of 1 epoch
    print("epoch:",epoch,"total_loss:",running_loss/50000)

#save training
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': speak2embed.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, 's2e_train_8_'+str(epoch)+'.pth')
        print("save complete")
    except:
        print("save error!")


