import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import requests, json 
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from datasets import load_dataset
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from data.utils import pre_caption

class dataset_caption_train(torch.utils.data.Dataset):
    def __init__(self, processor):        
        filename = 'train.json'
        ann_root = './datasets/celeba/annotation'
        print("loading josn file: ", os.path.join(ann_root, filename))   

        self.annotation = json.load(open(os.path.join(ann_root, filename),'r'))
        self.image_root = './datasets/celeba/images/'
        self.max_words = 55      
        self.prompt = 'a photo of a person where '
        self.processor = processor

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, ann['image'])        
        image = Image.open(image_path).convert('RGB')
        caption = self.prompt + pre_caption(ann['caption'], self.max_words) 

        encoding = self.processor(image, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = caption
        return encoding


class dataset_caption_test(torch.utils.data.Dataset):
    def __init__(self, processor):        
        filename = 'test.json'
        ann_root = './datasets/celeba/annotation'
        print("loading josn file: ", os.path.join(ann_root, filename))   

        self.annotation = json.load(open(os.path.join(ann_root, filename),'r'))
        self.image_root = './datasets/celeba/images/'
        self.max_words = 40      
        self.prompt = 'a photo of a person where '
        self.processor = processor

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, ann['image'])        
        image = Image.open(image_path).convert('RGB')
        encoding = self.processor(image, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding



# Image Encoder and Text Decoder
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", 
                                                      device_map="auto", 
                                                      load_in_8bit=True)

torch.cuda.empty_cache()
torch.manual_seed(42)
from peft import LoraConfig, get_peft_model

# Let's define the LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

def collate_fn_train(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch

def collate_fn_test(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        processed_batch[key] = torch.stack([example[key] for example in batch])
    return processed_batch


train_dataset = dataset_caption_train(processor=processor)
train_dataloader = DataLoader(train_dataset, 
                              shuffle=True, 
                              batch_size=32, 
                              pin_memory=True,
                              collate_fn=collate_fn_train)

test_dataset = dataset_caption_test(processor=processor)
test_dataloader = DataLoader(test_dataset, 
                              batch_size=32, 
                              pin_memory=True,
                              collate_fn=collate_fn_test)
print("Total valid images: ", test_dataset.__len__())

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)

num_epochs = 5
patience = 2
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()

    for idx, batch in zip(tqdm(range(len(train_dataloader)), desc='Training batch: ...'), train_dataloader):
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids)
            
        loss = outputs.loss
        epoch_loss += loss.item()
        optimizer.zero_grad()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    model.eval()
    result = []
    count = 0 
    for idx, batch in zip(tqdm(range(len(test_dataloader)), desc='testing batch: ...'), test_dataloader):
        pixel_values = batch.pop('pixel_values').to(device)

        generated_ids = model.generate(pixel_values=pixel_values, max_length=40)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for cap in generated_caption:
            ann = test_dataset.annotation[count]
            result.append({"image_id": ann['image'], "caption": cap})
            count += 1

    result_file = os.path.join(".", 'caption_result_celeba_%d.json'%(epoch))        
    json.dump(result,open(result_file,'w'))