import warnings
warnings.filterwarnings("ignore")
from models.decoder import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from models.iresnet import get_image_encoder

class FaceCPT_Decoder(nn.Module):
    def __init__(self,                 
                 config = 'configs/decoder_config.json',  
                 image_size = 112,
                 img_encoder = 'arcface',
                 max_length = 45,
                 prompt = 'a photo of a person where ',
                 ):
           
        super().__init__()
        self.visual_encoder, vision_width = get_image_encoder(img_encoder) 

        self.dec_tokenizer = init_dec_tokenizer()   
        decoder_config = BertConfig.from_json_file(config)
        decoder_config.encoder_width = vision_width

        self.text_decoder = BertLMHeadModel(config=decoder_config)
        self.text_decoder.resize_token_embeddings(len(self.dec_tokenizer))     
        
        self.max_length = max_length
        self.prompt = prompt
        self.prompt_length = len(self.dec_tokenizer(self.prompt).input_ids)-1

        
    def forward(self, image, caption):
        
        image_embeds = self.visual_encoder(image).unsqueeze(dim=1) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        text = self.dec_tokenizer(caption, 
                              padding='longest', 
                              truncation=True, 
                              max_length=self.max_length, 
                              return_tensors="pt").to(image.device) 
        
        decoder_input_ids = text.input_ids      
        decoder_input_ids[:,0] = self.dec_tokenizer.bos_token_id
        
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.dec_tokenizer.pad_token_id, -100)         
        decoder_targets[:,:self.prompt_length] = -100
     
        decoder_output, dec_embed = self.text_decoder(decoder_input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True)
                                           #return_emb = True   
                                          
        loss_lm = decoder_output.loss
        return loss_lm
        
    def generate(self, image, 
                 sample=False, 
                 num_beams=3, 
                 max_length=40, 
                 min_length=20, 
                 top_p=0.9, 
                 repetition_penalty=1.0):
        
        image_embeds = self.visual_encoder(image).unsqueeze(dim=1)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        
        prompt = [self.prompt] * image.size(0)
        input_ids = self.dec_tokenizer(prompt, return_tensors="pt").input_ids.to(image.device) 
        input_ids[:,0] = self.dec_tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 

        if sample:
            #nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.dec_tokenizer.sep_token_id,
                                                  pad_token_id=self.dec_tokenizer.pad_token_id, 
                                                  repetition_penalty=1.1,                                            
                                                  **model_kwargs)
        else:
            #beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.dec_tokenizer.sep_token_id,
                                                  pad_token_id=self.dec_tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)            
            
        captions = []    
        for output in outputs:
            caption = self.dec_tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(self.prompt):])
        return captions


def facecpt_caption(pretrained='',**kwargs):
    model = FaceCPT_Decoder(**kwargs)
    if pretrained:
        print("loading checkpoint form: ", pretrained)
        if os.path.isfile(pretrained):        
            checkpoint = torch.load(pretrained, map_location='cpu') 
        else:
            raise RuntimeError('checkpoint path is invalid')
        
        state_dict = checkpoint['model']

        print("missing keys:")
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg.missing_keys)
        return model 
    else:
        print("No pre-trained for finetuned model")


def init_dec_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer