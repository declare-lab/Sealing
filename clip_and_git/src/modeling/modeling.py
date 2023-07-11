"""
Transformer part of ClipBERT
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import CLIPTextModel, CLIPVisionModelWithProjection
from transformers import GitModel, GitPreTrainedModel
from typing import Optional, Union, Tuple, List
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPooling, BaseModelOutputWithPast

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class MyGitModel(GitModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length = input_shape[1]

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        projected_visual_features = None
        if pixel_values is not None:
            if pixel_values.ndim == 4:
                # here we assume pixel_values is of shape (batch_size, num_channels, height, width)
                visual_features = self.image_encoder(pixel_values).last_hidden_state

            elif pixel_values.ndim == 5:
                # here we assume pixel_values is of shape (batch_size, num_frames, num_channels, height, width)
                visual_features = []
                for frame_idx in range(pixel_values.shape[1]):
                    visual_features_frame = self.image_encoder(pixel_values[:, frame_idx, :, :]).last_hidden_state
                    # visual_features_frame += self.img_temperal_embedding[frame_idx]
                    visual_features.append(visual_features_frame)

                # finally, concatenate all features along sequence dimension
                visual_features = torch.cat(visual_features, dim=1)

            else:
                raise ValueError("pixel_values must be of rank 4 or 5")

            projected_visual_features = self.visual_projection(visual_features)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if projected_visual_features is None:
            projected_visual_features = torch.zeros(
                (embedding_output.shape[0], 0, embedding_output.shape[2]),
                dtype=embedding_output.dtype,
                device=embedding_output.device,
            )

        # Repeat visual features to match embedding batch size.
        projected_visual_features = projected_visual_features.repeat(
            embedding_output.size(0) // projected_visual_features.size(0), 1, 1
        )

        # concatenate patch token and text token embeddings
        hidden_states = torch.cat((projected_visual_features, embedding_output), dim=1)

        # By default, an additive causal mask is created
        # for masking the future (one direction).
        tgt_mask = self._generate_future_mask(seq_length, embedding_output.dtype, embedding_output.device)

        # Create an attention mask of shape (batch_size, 1, tgt_seq_len, src_seq_len)
        combined_attention_mask = self.create_attention_mask(
            tgt=embedding_output,
            memory=projected_visual_features,
            tgt_mask=tgt_mask,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is not None:
            # if the user provides an attention mask, we add it to the default one
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, embedding_output.dtype, tgt_len=input_shape[-1]).to(
                embedding_output.device
            )
            if past_key_values_length > 0:
                expanded_attn_mask = expanded_attn_mask[:, :, -past_key_values_length:, :]
            else:
                combined_attention_mask[:, :, -input_shape[1] :, -input_shape[1] :] += expanded_attn_mask

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=combined_attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values_present=pixel_values is not None,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithPast(
            last_hidden_state=sequence_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class MyGitForCausalLM(GitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.git = MyGitModel(config)
        self.output = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.output

    def set_output_embeddings(self, new_embeddings):
        self.output = new_embeddings

    # @add_start_docstrings_to_model_forward(GIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.git(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.output(sequence_output)

        loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            num_image_tokens = self.git.encoder.layer[0].attention.self.image_patch_tokens
            shifted_logits = logits[:, num_image_tokens:-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shifted_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": kwargs.get("pixel_values", None),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

class CLIPBaseModel(nn.Module):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    config keys:
        clip_config: str, text model name, default "openai/clip-vit-base-path-32"
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.txt_model = CLIPTextModel.from_pretrained(config.pretrained_model)
        self.vis_model = CLIPVisionModelWithProjection.from_pretrained(config.pretrained_model)

    def forward(self, txt_inputs, vis_inputs):
        r"""Modified from BertModel
        text_input_ids: (B, Lt)
        visual_inputs: (B * #frame, C, H, W)
        attention_mask: (B, Lt)  with 1 indicates valid, 0 indicates invalid position.
        """
        txt_out = self.txt_model(**txt_inputs)
        vis_out = self.vis_model(**vis_inputs)
        return dict(txt_out=txt_out, vis_out=vis_out, txt_attn_mask=txt_inputs["attention_mask"])

class GITBaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.model = MyGitForCausalLM.from_pretrained(config.pretrained_model, num_image_with_embedding=config.img_len)

    def forward(self, inputs):
        r"""Modified from BertModel
        text_input_ids: (B, Lt)
        visual_inputs: (B * #frame, C, H, W)
        attention_mask: (B, Lt)  with 1 indicates valid, 0 indicates invalid position.
        """
        if self.training:
            out = self.model(**inputs)
        else:
            inputs.pop('labels')
            out = self.model.generate(**inputs, max_length=50)
        return out

def instance_bce_with_logits(logits, labels, reduction="mean"):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(
        logits, labels, reduction=reduction)
    if reduction == "mean":
        loss *= labels.size(1)
    return loss

class CrossAttentionLayer(nn.Module):
    def __init__(self, in_size, dropout, nhead, n_layer=1, attn_type='dec-only', **kwargs):
        super(CrossAttentionLayer, self).__init__()
        self.attn_type = attn_type
        if attn_type == 'enc-dec':
            self.attention = torch.nn.Transformer(
                d_model=in_size,
                nhead=nhead,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dropout=dropout,
                dim_feedforward=4*in_size,
                batch_first=True,
                activation=torch.nn.functional.gelu,
            )
        elif attn_type in ['dec-only', 'dec-cas']:
            dec_layer = torch.nn.TransformerDecoderLayer(
                d_model=in_size,
                nhead=nhead,
                dim_feedforward=4*in_size,
                batch_first=True,
                activation=torch.nn.functional.relu
            )
            self.attention = torch.nn.TransformerDecoder(decoder_layer=dec_layer, num_layers=n_layer)
    
    def forward(self, txt_in, vis_in, txt_attn_mask=None):
        if self.attn_type == 'enc-dec':
            return self.attention(vis_in, txt_in, tgt_key_padding_mask=~txt_attn_mask.bool())
        elif self.attn_type == 'dec-only':
            # trg is the first param
            return self.attention(txt_in, vis_in, tgt_key_padding_mask=~txt_attn_mask.bool())
        elif self.attn_type == 'dec-cas':
            T = vis_in.size(1)  # (B, L, E_v)
            o = txt_in
            for t in range(T):
                o = self.attention(
                    o, vis_in[:, t:t+1], 
                    tgt_key_padding_mask=~txt_attn_mask
                )
            return o
        
        
class CLIPForSeqClassification(nn.Module):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self, config):
        super(CLIPForSeqClassification, self).__init__()
        self.config = config

        if 'clip' in config.pretrained_model.lower():
            self.vlm = CLIPBaseModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
                
        self.attention = CrossAttentionLayer(
            in_size=config.txt_output_size, dropout=0.1, nhead=8, attn_type='dec-only'
        )
        self.classifier = nn.Linear(config.txt_output_size, config.num_labels)

    def forward(self, txt_inputs, vis_inputs, video_start_end, repeat_counts=None, **kwargs):
        outputs = self.vlm(
            txt_inputs=txt_inputs,
            vis_inputs=vis_inputs,
        )
        txt_output, vis_output = outputs['txt_out'], outputs['vis_out']
        txt_attn_mask = outputs['txt_attn_mask']    # (B, L_t)
        
        if 'pooler_output' in vis_output.keys():
            vis_pooled_output = vis_output.pooler_output
        else:
            vis_pooled_output = vis_output.image_embeds    # (\sum L_i, E)
        txt_pooled_output = txt_output.pooler_output    # (B, E_t)

        bsz, e_t = txt_pooled_output.size()
        decoded_tokens = txt_pooled_output.new_zeros(bsz, 1, e_t)
        txt_attn_mask = torch.cat([txt_attn_mask.new_ones(bsz, 1), txt_attn_mask], dim=1)   # (B, L_t + 1)

        # for unequal numbers of video frames
        sample_vis_outputs = []
        if repeat_counts is None:
            for s, e in zip(video_start_end[:-1],video_start_end[1:]):
                # sample_vis_outputs.append(vis_pooled_output[s:e].mean(dim=0, keepdim=True))  # List of (1, E) 
                sample_vis_outputs.append(vis_pooled_output[s:e])  # List of (L, E_v) 
            sample_vis_outputs = torch.stack(sample_vis_outputs)
        else:
            for s, e, rc in zip(video_start_end[:-1], video_start_end[1:], repeat_counts):
                sample_vis_outputs.append(vis_pooled_output[s:e].mean(dim=0).repeat(rc, 1)) # (rc, E_v)
            sample_vis_outputs = torch.cat(sample_vis_outputs, dim=0)

        txt_attn_in = torch.cat([decoded_tokens, txt_output.last_hidden_state], dim=1)
        vis_attn_in = sample_vis_outputs
        
        attn_outputs = self.attention(txt_attn_in, vis_attn_in, txt_attn_mask) # 
        logits = self.classifier(attn_outputs)[:,0,:]   # (b, V)
        return logits



