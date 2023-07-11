from torch import nn
from src.utils.load_save import load_state_dict_with_mismatch
from transformers import CLIPVisionModel, CLIPTextModel
from torch.nn import CrossEntropyLoss, MSELoss
from .modeling import instance_bce_with_logits
from .modeling import CLIPForSeqClassification

# model wrapper
class CLIPModelforFinetune(nn.Module):
    def __init__(self, config, vlm_cls=CLIPForSeqClassification):
        super(CLIPModelforFinetune, self).__init__()
        self.config = config
        self.VLModel = vlm_cls(config)
        
    def forward(self, batch):
        # used to make visual feature copies
        repeat_counts = batch["n_examples_list"]
        # obtain outputs
        if 'clip' in self.config.pretrained_model:
            vis_inputs = {'pixel_values': batch['visual_inputs']}
            
            txt_inputs = {'input_ids': batch['text_input_ids'], \
                            'attention_mask': batch['text_attention_mask']}
            logits = self.VLModel(
                                txt_inputs=txt_inputs,
                                vis_inputs=vis_inputs,
                                video_start_end=batch['video_start_end'],
                                repeat_counts = None if all(rc == 1 for rc in repeat_counts) else repeat_counts
                            )
            logits, loss = self.calc_loss(logits, batch['labels'])
            return dict(logits=logits, loss=loss)

        elif 'git' in self.config.pretrained_model: # generation style
            inputs = {
                'pixel_values': batch['visual_inputs'],
                'input_ids': batch['text_input_ids'],
                'attention_mask': batch['text_attention_mask'],
                'labels': batch['labels']
            }
            loss = generated_ids = None
            out = self.VLModel(inputs)
            if self.training:
                loss = out.loss
            else:
                generated_ids = out
            return dict(generated_ids=generated_ids, loss=loss)

    def calc_loss(self, logits, labels):
        if labels is not None:
            if self.config.num_labels == 1:  # regression
                loss_fct = MSELoss(reduction="none")
                # labels = labels.to(torch.float)
                loss = loss_fct(
                    logits.view(-1), labels.view(-1))
            else:
                if self.config.loss_type == 'bce':  # [VQA]
                    loss = instance_bce_with_logits(
                        logits, labels, reduction="none")
                elif self.config.loss_type == "ce":  # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(
                        logits.view(-1, self.config.num_labels),
                        labels.view(-1))
                else:
                    raise ValueError("Invalid option for config.loss_type")
        else:
            loss = 0
        return logits, loss