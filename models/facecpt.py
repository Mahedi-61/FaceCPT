import warnings
warnings.filterwarnings("ignore")
from models.decoder import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from models.iresnet import iresnet50, iresnet100


class BLIP_Base(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 112,
                 img_encoder = 'arcface',            
                 ):
           
        super().__init__()
        self.visual_encoder = iresnet50()
        vision_width = 768
        
        if img_encoder =='arcface':
            checkpoint = torch.load("weights/arcface_ir50_ms1mv3.pth", 
                            map_location=torch.device('cpu'), weights_only=True)
            msg = self.visual_encoder.load_state_dict(checkpoint, strict=False)
            print("missing keys in saved_checkpoint")
            print(msg)

        self.dec_tokenizer = init_dec_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)  

        
    def forward(self, image, caption, mode):
        
        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        text = self.dec_tokenizer(caption, return_tensors="pt").to(image.device) 
        
        if mode=='image':    
            # return image features
            image_embeds = self.visual_encoder(image)             
            return image_embeds
        
        elif mode=='text':
            # return text features
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')  
            return text_output.last_hidden_state
        
        elif mode=='multimodal':
            # return multimodel features
            image_embeds = self.visual_encoder(image)    
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)      
            
            text.input_ids[:,0] = self.dec_tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )              
            return output.last_hidden_state


class FaceCPT_Decoder(nn.Module):
    def __init__(self,                 
                 config = 'configs/decoder_config.json',  
                 image_size = 112,
                 img_encoder = 'arcface',
                 prompt = 'a photo of a person where ',
                 ):
           
        super().__init__()
        self.visual_encoder = iresnet50()
        vision_width = 768

        if img_encoder == 'arcface':
            checkpoint = torch.load("weights/arcface_ir50_ms1mv3.pth", 
                            map_location=torch.device('cpu'), weights_only=True)
            msg = self.visual_encoder.load_state_dict(checkpoint, strict=False)
            print("missing keys in saved_checkpoint")
            print(msg)

        self.dec_tokenizer = init_dec_tokenizer()   
        decoder_config = BertConfig.from_json_file(config)
        decoder_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=decoder_config)
        self.text_decoder.resize_token_embeddings(len(self.dec_tokenizer))     
        
        self.prompt = prompt
        self.prompt_length = len(self.dec_tokenizer(self.prompt).input_ids)-1

        
    def forward(self, image, caption):
        
        image_embeds = self.visual_encoder(image).unsqueeze(dim=1) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        text = self.dec_tokenizer(caption, 
                              padding='longest', 
                              truncation=True, 
                              max_length=40, 
                              return_tensors="pt").to(image.device) 
        
        decoder_input_ids = text.input_ids      
        decoder_input_ids[:,0] = self.dec_tokenizer.bos_token_id
        
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.dec_tokenizer.pad_token_id, -100)         
        decoder_targets[:,:self.prompt_length] = -100
     
        decoder_output = self.text_decoder(decoder_input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
        loss_lm = decoder_output.loss
        return loss_lm
        
    def generate(self, image, 
                 sample=False, 
                 num_beams=3, 
                 max_length=40, 
                 min_length=10, 
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


def facecpt_decoder(pretrained='',**kwargs):
    model = FaceCPT_Decoder(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        #assert(len(msg.missing_keys) == 0)
    return model    
    
def blip_feature_extractor(pretrained='',**kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys) == 0)
    return model        


def init_dec_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer
    
"""
def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")
"""

def load_checkpoint(model, filename):
    if os.path.isfile(filename):        
        checkpoint = torch.load(filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint path is invalid')
        
    state_dict = checkpoint['model']

    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]
    
    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % filename)  
    return model, msg
    
