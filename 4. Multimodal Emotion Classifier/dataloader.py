import torch
import gzip
import pickle
import random
from tqdm import tqdm


class KEMDset(torch.utils.data.Dataset):
    def __init__(
            self,
            file,
            tokenizer,
            shuffle: bool,
            balance: bool,
            seed=824
    ):
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.balance = balance

        with gzip.open(file, "rb") as fh:
            self.file = pickle.load(fh)

        data = []

        if self.shuffle:
            self.file = self.file.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            pass

        if self.balance:
            random.seed(seed)
            neutral_index = self.file.index[self.file['Emotion'] == 'neutral'].tolist()
            balance_num = len(neutral_index) - (self.file.shape[0] - len(neutral_index)) // 4
            selected_index = random.sample(neutral_index, balance_num)
            self.file = self.file.drop(selected_index).reset_index(drop=True)

        # 음성인식 / 감정분류 중 원하는 Mode 따라 데이터의 Label 정의
        for i in tqdm(range(self.file.shape[0])):
            audio = self.file.loc[i]['Audio']

            tokenized = tokenizer.encode_plus(self.file.loc[i]['Text'], max_length=64, padding='max_length',
                                              truncation=True, return_tensors='pt')
            input_ids = tokenized['input_ids']
            token_type_ids = tokenized['token_type_ids']
            attention_mask = tokenized['attention_mask']

            arousal = float(self.file.loc[i]['Arousal'])
            valence = float(self.file.loc[i]['Valence'])

            if self.file.loc[i]['Emotion'] == 'neutral':
                emotion = 0
            elif self.file.loc[i]['Emotion'] == 'sad':
                emotion = 1
            elif self.file.loc[i]['Emotion'] == 'happy':
                emotion = 2
            elif self.file.loc[i]['Emotion'] == 'surprise':
                emotion = 3
            elif self.file.loc[i]['Emotion'] == 'angry':
                emotion = 4

            data.append((audio, input_ids, token_type_ids, attention_mask, emotion, arousal, valence))
        else:
            pass

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        result = {'audio': torch.as_tensor(self.data[index][0], dtype=torch.float32),
                  'input_ids': self.data[index][1],
                  'token_type_ids': self.data[index][2],
                  'attention_mask': self.data[index][3],
                  'emotion': torch.as_tensor(self.data[index][4], dtype=torch.int),
                  'arousal': torch.as_tensor(self.data[index][5], dtype=torch.float32),
                  'valence': torch.as_tensor(self.data[index][6], dtype=torch.float32)}

        return result
