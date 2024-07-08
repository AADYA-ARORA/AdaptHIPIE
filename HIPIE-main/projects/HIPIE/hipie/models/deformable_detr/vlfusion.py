import torch
import torch.nn as nn
import torch.nn.functional as F
from .fuse_helper import BiAttentionBlockForCheckpoint
import torch.utils.checkpoint as checkpoint
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_utils import apply_chunking_to_forward
from .modeling_bert import BertAttention, BertIntermediate, BertOutput

class BertEncoderLayer(BertPreTrainedModel):
    def __init__(self, config, clamp_min_for_underflow=False, clamp_max_for_overflow=False):
        super().__init__(config)
        self.config = config

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.attention = BertAttention(config, clamp_min_for_underflow, clamp_max_for_overflow)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, inputs):
        language_dict_features = inputs["lang"]
        hidden_states = language_dict_features["hidden"]
        attention_mask = language_dict_features["masks"]

        device = hidden_states.device
        input_shape = hidden_states.size()[:-1]
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        self_attention_outputs = self.attention(
            hidden_states,
            extended_attention_mask,
            None,
            output_attentions=False,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        hidden_states = outputs[0]

        language_dict_features["hidden"] = hidden_states

        features_dict = {"visual": inputs["visual"],
                         "lang": language_dict_features}

        return features_dict

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(Adapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, input_dim)
        )

    def forward(self, x):
        return x + self.adapter(x)


class VLFuse(nn.Module):
    def __init__(self, cfg):
        super(VLFuse, self).__init__()
        self.init_configs(cfg)
        self.cfg = cfg
        self.use_checkpoint = cfg.MODEL.VL_FUSION_USE_CHECKPOINT

        self.b_attn = BiAttentionBlockForCheckpoint(
            v_dim=self.img_dim,
            l_dim=self.lang_dim,
            embed_dim=self.embed_dim,
            num_heads=self.n_head,
            dropout=0.1,
            drop_path=.0,
            init_values=1.0 / cfg.MODEL.DDETRS.ENC_LAYERS,
            cfg=cfg
        )

        self.adapter = Adapter(input_dim=self.lang_dim, bottleneck_dim=256)
        self.lang_proj = nn.Linear(self.lang_dim, self.img_dim)  
        self.cosine_sim_mlp = nn.Sequential(
            nn.Linear(self.img_dim + self.img_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def init_configs(self, cfg):
        self.lang_model = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        self.img_dim = cfg.MODEL.DDETRS.HIDDEN_DIM
        self.max_query_len = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN
        self.n_layers = cfg.MODEL.LANGUAGE_BACKBONE.N_LAYERS
        self.n_head = 8
        self.embed_dim = cfg.MODEL.DDETRS.VL_HIDDEN_DIM

        if self.lang_model in ["bert-base-uncased", "roberta-base", "clip"]:
            self.lang_dim = cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM
        else:
            self.lang_dim = 1024

    def forward(self, x, task=None):
        visual_features = x["visual"]
        language_dict_features = x["lang"]

        if self.use_checkpoint:
            fused_visual_features, language_features = checkpoint.checkpoint(
                self.b_attn,
                visual_features, language_dict_features['hidden'], language_dict_features['masks'], task
            )
        else:
            fused_visual_features, language_features = self.b_attn(
                visual_features, language_dict_features['hidden'], language_dict_features['masks'], task
            )

        language_features = self.adapter(language_features)
        language_dict_features['hidden'] = language_features

        proj_lang = self.lang_proj(language_features.mean(dim=1))
        cosine_sim = F.cosine_similarity(proj_lang.unsqueeze(1), visual_features, dim=-1)

        mlp_input = torch.cat([proj_lang.unsqueeze(1).expand(-1, visual_features.size(1), -1), visual_features], dim=-1)
        mlp_input = mlp_input.view(-1, mlp_input.size(-1))  
        mlp_output = self.cosine_sim_mlp(mlp_input)
        mlp_output = mlp_output.view(x["visual"].size(0), x["visual"].size(1), -1)  

        fused_language_dict_features = language_dict_features

        features_dict = {"visual": fused_visual_features,
                         "lang": fused_language_dict_features,
                         "cosine_sim": cosine_sim,
                         "mlp_output": mlp_output}

        return features_dict


