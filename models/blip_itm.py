from models.med import BertConfig, BertModel
import torch
from torch import nn
import torch.nn.functional as F

from models.facecpt import create_vit, init_tokenizer, load_checkpoint

class BLIP_ITM(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                      
                 embed_dim = 256,     
                 ):
            
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)          

        text_width = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.itm_head = nn.Linear(text_width, 2) 

        
        
    def forward(self, image, caption, match_head='itm'):

        image_embeds = self.visual_encoder(image) #b_s x 577 x 768
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)       

        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(image.device) #b_s x 35

        if match_head=='itm':
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      ) #b_s x 35 x 768
            
            #print(output.last_hidden_state.size())
            itm_output = self.itm_head(output.last_hidden_state[:,0,:]) #b_s x 768 --> b_x x 2
            return itm_output
            
        elif match_head=='itc':
            text_output = self.text_encoder(text.input_ids, 
                                            attention_mask = text.attention_mask,                      
                                            return_dict = True, 
                                            mode = 'text') #1, 35, 768

            #print(text_output.last_hidden_state.size())                  
            image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  #b_s x 256
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)  #b_s x 256
            
            sim = image_feat @ text_feat.t() #b_s, b_s
            return sim

        
def blip_itm(pretrained='',**kwargs):
    model = BLIP_ITM(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,  pretrained)
        assert(len(msg.missing_keys)==0)
    return model         


if __name__ == "__main__":                                        
    blip_itm(pretrained='weights/model_base_retrieval_coco.pth',
             image_size=384, vit='base')
