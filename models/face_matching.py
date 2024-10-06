from models.med import BertConfig, BertModel
import torch
from torch import nn
import torch.nn.functional as F

from models.blip import init_tokenizer, load_checkpoint
from models.iresnet import iresnet50, iresnet100


class ITC_ITM(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 112,
                 img_encoder = 'arcface', 
                 max_word = 31,                 
                 embed_dim = 256,     
                 ):
            
        super().__init__()
        
        self.visual_encoder = iresnet50()
        vision_width = 768

        if img_encoder=='arcface':
            checkpoint = torch.load("weights/arcface_ir50_ms1mv3.pth", 
                            map_location=torch.device('cpu'), weights_only=True)
            msg = self.visual_encoder.load_state_dict(checkpoint, strict=False)
            print("missing keys in saved_checkpoint")
            print(msg)

        self.max_word = max_word
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)          

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.itm_head = nn.Linear(text_width, 2) 

        
    def forward(self, image, caption, match_head='itm', return_text=False):
        image_embeds = self.visual_encoder(image).unsqueeze(dim=1)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)       

        text = self.tokenizer(caption, 
                              padding='max_length', 
                              truncation=True, 
                              max_length=self.max_word, 
                              return_tensors="pt").to(image.device) #b_s x 35

        if match_head=='itm':
            text_output = self.text_encoder.bert(text.input_ids, 
                                        attention_mask = text.attention_mask, 
                                        return_dict = True, 
                                        mode = 'text')
        
            text_embeds = text_output.last_hidden_state 
            output = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                attention_mask = text.attention_mask,
                                encoder_hidden_states = image_embeds,
                                encoder_attention_mask = image_atts,      
                                return_dict = True,
                                mode = 'fusion',
                                )  #b_s x max_words x 768


            itm_output = self.itm_head(output.last_hidden_state[:,0,:]) #b_s x 768 --> b_x x 2
            if return_text: return itm_output, text 
            return itm_output
            
        elif match_head=='itc':
            text_output = self.text_encoder.bert(text.input_ids, 
                                            attention_mask = text.attention_mask,                      
                                            return_dict = True, 
                                            mode = 'text') 
               
            image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  #b_s x 256
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)  #b_s x 256
            
            #sim = image_feat @ text_feat.t() #b_s, b_s
            #return sim
            cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)(image_feat, text_feat) #bs x 1
            return cosine_similarity

        
def facecpt_matching(pretrained='',**kwargs):
    model = ITC_ITM(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,  pretrained)
        assert(len(msg.missing_keys)==0)
    return model         


