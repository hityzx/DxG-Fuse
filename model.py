import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, ViTModel, BertConfig, ViTConfig

class CrossModalAttentionBlock(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout_rate):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, _ = self.cross_attn(query, key, value, key_padding_mask=key_padding_mask)
        query = query + self.dropout(attn_output) 
        query = self.norm1(query)
        ffn_output = self.ffn(query)
        query = query + self.dropout(ffn_output) 
        query = self.norm2(query)
        return query

class MultiFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout_rate):
        super().__init__()
        self.text_attends_to_image_block = CrossModalAttentionBlock(
            d_model, nhead, dim_feedforward, dropout_rate
        )

    def forward(self, text_tokens, text_mask, image_patches):
        image_key_padding_mask = torch.zeros(image_patches.shape[0], image_patches.shape[1], dtype=torch.bool, device=image_patches.device)

        img_aware_text_tokens = self.text_attends_to_image_block(
            query=text_tokens,
            key=image_patches,
            value=image_patches,
            key_padding_mask=image_key_padding_mask 
        )

        bert_key_padding_mask = ~text_mask
        img_aware_text_tokens_masked = img_aware_text_tokens.masked_fill(bert_key_padding_mask.unsqueeze(-1), 0.0)
        pooled_img_aware_text = img_aware_text_tokens_masked.sum(dim=1) / text_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        return pooled_img_aware_text


class dxg(nn.Module):
    def __init__(self,
                 bert_model_name='bert-base-uncased',
                 vit_model_name='google/vit-base-patch16-224-in21k',
                 judgment_embed_dim=64,
                 perspective_dim=256,
                 enhanced_img_dim=512, 
                 transformer_d_model=768, 
                 transformer_nhead=8,
                 transformer_num_layers=2, 
                 transformer_dim_feedforward=2048,
                 dropout_rate=0.1):
        super().__init__()

        print("Initializing Model...")
        self.bert_config = BertConfig.from_pretrained(bert_model_name)
        self.vit_config = ViTConfig.from_pretrained(vit_model_name)

        print(f"Loading BERT: {bert_model_name}")
        self.bert = BertModel.from_pretrained(bert_model_name, use_safetensors=False)
        print(f"Loading ViT: {vit_model_name}")
        self.vit = ViTModel.from_pretrained(vit_model_name)

        bert_hidden_size = self.bert_config.hidden_size
        vit_embed_dim = self.vit_config.hidden_size

        if bert_hidden_size != transformer_d_model:
            print(f"Warning: BERT hidden size ({bert_hidden_size}) != Core Fusion d_model ({transformer_d_model}).")
        if vit_embed_dim != transformer_d_model:
             print(f"Warning: ViT hidden size ({vit_embed_dim}) != Core Fusion d_model ({transformer_d_model}).")

        self.judgment_embedding = nn.Embedding(num_embeddings=3, embedding_dim=judgment_embed_dim)

        img_enhance_attn_dim = transformer_d_model
        self.img_enhance_q_proj = nn.Linear(bert_hidden_size, img_enhance_attn_dim)
        self.img_enhance_k_proj = nn.Linear(vit_embed_dim, img_enhance_attn_dim)
        self.img_enhance_v_proj = nn.Linear(vit_embed_dim, img_enhance_attn_dim)
        self.img_enhance_mlp = nn.Sequential(
            nn.Linear(vit_embed_dim + img_enhance_attn_dim, enhanced_img_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(enhanced_img_dim * 2, enhanced_img_dim)
        )

        perspective_input_dim = judgment_embed_dim + bert_hidden_size
        self.mlp_senti = self._build_mlp(perspective_input_dim, perspective_dim, dropout_rate)
        self.mlp_cont = self._build_mlp(perspective_input_dim, perspective_dim, dropout_rate)
        self.mlp_cons = self._build_mlp(perspective_input_dim, perspective_dim, dropout_rate)

        self.multi_fusion_module = MultiFusionModule(
            d_model=bert_hidden_size,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout_rate=dropout_rate
        )

        self.proj_mfm_out = nn.Linear(bert_hidden_size, transformer_d_model) if bert_hidden_size != transformer_d_model else nn.Identity()
        self.proj_i_enhanced = nn.Linear(enhanced_img_dim, transformer_d_model) if enhanced_img_dim != transformer_d_model else nn.Identity()
        self.proj_p_senti = nn.Linear(perspective_dim, transformer_d_model) if perspective_dim != transformer_d_model else nn.Identity()
        self.proj_p_cont = nn.Linear(perspective_dim, transformer_d_model) if perspective_dim != transformer_d_model else nn.Identity()
        self.proj_p_cons = nn.Linear(perspective_dim, transformer_d_model) if perspective_dim != transformer_d_model else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier_primary = nn.Linear(transformer_d_model, 1)
        self.classifier_aux_senti = nn.Linear(perspective_dim, 3)
        self.classifier_aux_cont = nn.Linear(perspective_dim, 3)
        self.classifier_aux_cons = nn.Linear(perspective_dim, 3)

        print("Model Initialization Complete.")

    def _build_mlp(self, input_dim, output_dim, dropout_rate):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self,
                content_ids, content_mask,
                image,
                cs_rationale_ids, cs_rationale_mask, cs_pred,
                td_rationale_ids, td_rationale_mask, td_pred,
                img_rationale_ids, img_rationale_mask, img_pred,
                img_desc_ids, img_desc_mask,
                **kwargs):

        bert_outputs_content = self.bert(input_ids=content_ids, attention_mask=content_mask)
        V_T_orig_tokens = bert_outputs_content.last_hidden_state

        bert_outputs_cs = self.bert(input_ids=cs_rationale_ids, attention_mask=cs_rationale_mask)
        V_R_cs_cls = bert_outputs_cs.pooler_output
        bert_outputs_td = self.bert(input_ids=td_rationale_ids, attention_mask=td_rationale_mask)
        V_R_td_cls = bert_outputs_td.pooler_output
        bert_outputs_img_rat = self.bert(input_ids=img_rationale_ids, attention_mask=img_rationale_mask)
        V_R_img_cls = bert_outputs_img_rat.pooler_output
        bert_outputs_img_desc = self.bert(input_ids=img_desc_ids, attention_mask=img_desc_mask)
        V_T_img_desc_cls = bert_outputs_img_desc.pooler_output

        vit_outputs = self.vit(pixel_values=image)
        V_I_cls = vit_outputs.pooler_output
        V_I_patches = vit_outputs.last_hidden_state[:, 1:, :]

        V_J_senti = self.judgment_embedding(cs_pred)
        V_J_cont = self.judgment_embedding(td_pred)
        V_J_cons = self.judgment_embedding(img_pred)

        query = self.img_enhance_q_proj(V_T_img_desc_cls)
        keys = self.img_enhance_k_proj(V_I_patches)
        values = self.img_enhance_v_proj(V_I_patches)
        attn_scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))
        attn_scores = attn_scores / (keys.size(-1) ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        V_I_attended = torch.bmm(attn_probs, values).squeeze(1)
        V_I_combined = torch.cat((V_I_cls, V_I_attended), dim=-1)
        V_I_enhanced = self.img_enhance_mlp(V_I_combined)

        P_cont_input = torch.cat((V_J_cont, V_R_cs_cls), dim=-1)
        P_senti_input = torch.cat((V_J_senti, V_R_td_cls), dim=-1)
        P_cons_input = torch.cat((V_J_cons, V_R_img_cls), dim=-1)
        P_cont = self.mlp_cont(P_cont_input)
        P_senti = self.mlp_senti(P_senti_input)
        P_cons = self.mlp_cons(P_cons_input)

        V_text_img_aware_pooled = self.multi_fusion_module(
            text_tokens=V_T_orig_tokens,
            text_mask=content_mask.bool(),
            image_patches=V_I_patches
        )

        proj_Text = self.proj_mfm_out(V_text_img_aware_pooled)
        proj_Img = self.proj_i_enhanced(V_I_enhanced)
        proj_Senti = self.proj_p_senti(P_senti)
        proj_Cont = self.proj_p_cont(P_cont)
        proj_Cons = self.proj_p_cons(P_cons)

        fusion_input = torch.stack([proj_Text, proj_Img, proj_Senti, proj_Cont, proj_Cons], dim=1)

        transformer_output = self.transformer_encoder(fusion_input)

        V_final = transformer_output[:, 0, :]

        V_final_dropped = self.dropout(V_final)
        logits_primary = self.classifier_primary(V_final_dropped)

        logits_aux_senti = self.classifier_aux_senti(self.dropout(P_senti))
        logits_aux_cont = self.classifier_aux_cont(self.dropout(P_cont))
        logits_aux_cons = self.classifier_aux_cons(self.dropout(P_cons))

        return {
            'logits_primary': logits_primary,
            'logits_aux_senti': logits_aux_senti,
            'logits_aux_cont': logits_aux_cont,
            'logits_aux_cons': logits_aux_cons
        }


