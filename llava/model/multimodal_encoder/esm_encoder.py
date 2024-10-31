import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class ESMProteinEncoder(nn.Module):
    def __init__(self, protein_encoder, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.protein_encoder_name = protein_encoder
        self.select_layer = args.mm_protein_select_layer
        self.select_feature = getattr(args, 'mm_protein_select_feature', 'cls')  # Default to 'patch'
        # self.max_length = getattr(args, 'mm_protein_max_length', 512)  # Default to 'patch'

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_protein_encoder', False):
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.protein_encoder_name))
            return

        self.protein_encoder = AutoModel.from_pretrained(self.protein_encoder_name, device_map=device_map)
        self.tokenizer = AutoTokenizer.from_pretrained(self.protein_encoder_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.protein_encoder_name,
        #                                               max_length=self.max_length, padding=True, truncation=True)
        self.protein_encoder.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, sequence_forward_outs):
        sequence_features = sequence_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':  # Default behavior for 'patch'
            sequence_features = sequence_features[:, 1:]  # Assuming 'patch' indicates skipping the CLS token
        elif self.select_feature == 'cls_patch':
            sequence_features = sequence_features  # Keep the entire sequence including CLS
        elif self.select_feature == 'cls': # Keep the CLS only
            sequence_features = sequence_features[:, 0:1]
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return sequence_features

    @torch.no_grad()
    def forward(self, sequences):
        # Tokenize sequences
        inputs = self.tokenizer(sequences, return_tensors='pt', padding=True, truncation=True)


        if isinstance(sequences, list) and all(isinstance(seq, list) for seq in sequences):
            raise ValueError("sequence is a list of list - FixMe")
            sequence_features = []
            for sequence in sequences:
                # sequence_forward_out = self.protein_encoder(**inputs.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                sequence_forward_out = self.protein_encoder(**inputs.to(device=self.device), output_hidden_states=True)
                # sequence_feature = self.feature_select(sequence_forward_out).to(sequence.dtype)
                sequence_feature = self.feature_select(sequence_forward_out).to(self.dtype)
                sequence_features.append(sequence_feature)
        else:
            # sequence_forward_outs = self.protein_encoder(**inputs.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            sequence_forward_outs = self.protein_encoder(**inputs.to(device=self.device), output_hidden_states=True)
            # sequence_features = self.feature_select(sequence_forward_outs).to(sequences.dtype)
            sequence_features = self.feature_select(sequence_forward_outs).to(self.dtype)

        return sequence_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.protein_encoder.dtype

    @property
    def device(self):
        return self.protein_encoder.device

    @property
    def config(self):
        return self.protein_encoder.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    # No equivalent for num_patches in protein sequences, so this can be omitted.

    # @property
    # def num_patches_per_side(self):
    #     return self.config.image_size // self.config.patch_size

    # @property
    # def num_patches(self):
    #     return (self.config.image_size // self.config.patch_size) ** 2
