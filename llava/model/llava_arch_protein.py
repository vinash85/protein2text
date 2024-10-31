#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.



import torch
from torch.cuda import CudaError as CudaRuntimeError
import torch.nn as nn
from abc import ABC, abstractmethod

from .multimodal_encoder.builder_protein import build_protein_encoder  # Using multimodal_encoder to build protein encoder
from .multimodal_projector.builder_protein import build_protein_projector  # Using multimodal_projector to build protein projector

from llava.constants_protein import IGNORE_INDEX, PROTEIN_SEQUENCE_TOKEN_INDEX, DEFAULT_PROTEIN_SEGMENT_TOKEN, DEFAULT_PROT_START_TOKEN, DEFAULT_PROT_END_TOKEN



class LlavaMetaModel:

    def __init__(self, config):
        """
        Initialize the LlavaMetaModel class for handling protein sequences.
        """
        super(LlavaMetaModel, self).__init__(config)

        # Check if the configuration includes a protein encoder and initialize it if present
        if hasattr(config, "mm_protein_encoder"):
            self.protein_encoder = build_protein_encoder(config, delay_load=True)  # Build the protein encoder
            self.mm_projector = build_protein_projector(config)  # Build the protein projector

    def get_protein_encoder(self):
        """
        Retrieve the protein encoder, handling cases where it might be stored as a list (for FSDP compatibility).
        """
        protein_encoder = getattr(self, 'protein_encoder', None)
        if isinstance(protein_encoder, list):
            protein_encoder = protein_encoder[0]
        return protein_encoder

    def initialize_protein_modules(self, model_args, fsdp=None):
        """
        Initialize the protein encoder and projector modules.
        """
        # Assign the protein encoder directly from model_args
        protein_encoder = model_args.protein_encoder  # Use consistent naming for the protein encoder
        mm_protein_select_layer = model_args.mm_protein_select_layer
        mm_protein_select_feature = model_args.mm_protein_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        # Store the protein encoder configuration
        self.config.mm_protein_encoder = protein_encoder

        # If the protein encoder is not yet initialized, build it
        if self.get_protein_encoder() is None:
            protein_encoder = build_protein_encoder(model_args)

            # Handle Fully Sharded Data Parallel (FSDP) configuration
            if fsdp is not None and len(fsdp) > 0:
                self.protein_encoder = [protein_encoder]  # Store as a list for FSDP compatibility
            else:
                self.protein_encoder = protein_encoder
        else:
            # If already initialized, ensure it's properly loaded
            if fsdp is not None and len(fsdp) > 0:
                protein_encoder = self.protein_encoder[0]
            else:
                protein_encoder = self.protein_encoder
            protein_encoder.load_model()  # Load the protein encoder model

        # Configure the model to use the protein projector
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')  # Default to a linear projector
        self.config.mm_hidden_size = protein_encoder.hidden_size  # Set the hidden size based on the encoder
        self.config.mm_protein_select_layer = mm_protein_select_layer  # Specify the layer to select from the protein encoder
        self.config.mm_protein_select_feature = mm_protein_select_feature  # Specify the feature to select from the protein encoder

        # Initialize the protein projector if it hasn't been set yet
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_protein_projector(self.config)
        else:
            # If the projector is frozen (e.g., by LoRA), ensure gradients are enabled
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        # Load pre-trained weights for the protein projector if specified
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))




class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_protein_encoder(self):
        return self.get_model().get_protein_encoder()

    def encode_protein_sequences(self, sequences):
        """
        Encodes protein sequences using the  protein encoder and projects them to a suitable space.
        
        :param sequences: A list of protein sequences (strings).
        :return: The encoded and projected protein sequence features.
        """


        sequence_features = self.get_model().get_protein_encoder()(sequences)
        # print(f"Data type of sequence_features before projection: {sequence_features.dtype}")
        # print(f"shape of sequence_features before projection: {sequence_features.shape}")
        # for layer in self.get_model().mm_projector:
        #     if hasattr(layer, 'weight'):
        #         print(f"Data type of {layer} weights: {layer.weight.dtype}")

        # sequence_features = torch.stack(sequence_features, dim=0)

        #Causes problem during inferencing: Ensure that the data type of the sequence features matches the projector weights
        projector_weight_dtype = self.get_model().mm_projector[0].weight.dtype
        if projector_weight_dtype != sequence_features.dtype:
            print(f"Data type of sequence_features before projection: {sequence_features.dtype}")
            sequence_features = sequence_features.to(projector_weight_dtype)
            print(f"Data type of sequence_features after projection: {sequence_features.dtype}")

        sequence_features = self.get_model().mm_projector(sequence_features)


        # print(f"Data type of sequence_features after projection: {sequence_features.dtype}")


        # print (f"sequence_features after projection:{sequence_features.shape}")
        return sequence_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        sequences #, sequence_lengths=None
    ):
        """
        Prepares inputs and labels for multimodal training, including protein sequences.
        
        :param input_ids: Token IDs of the input text.
        :param position_ids: Position IDs of the input text.
        :param attention_mask: Attention mask for the input text.
        :param past_key_values: Cached key/values for previous steps.
        :param labels: Labels for training.
        :param sequences: Protein sequences to be encoded and integrated.
        :param sequence_lengths: Lengths of the protein sequences (not used).
        :return: Processed inputs, labels, and other relevant data for multimodal training.
        """

        # print(f"Initial input_ids shape: {input_ids.shape}") # torch.Size([32, 30]) 
        # print(f"Initial position_ids shape: {position_ids}") # None 
        # print(f"Initial attention_mask shape: {attention_mask.shape}") # torch.Size([32, 30]) 
        # print(f"Initial past_key_values shape: {past_key_values}") # None
        # print(f"Initial labels shape: {labels.shape}") # torch.Size([32, 30]) 


        protein_encoder = self.get_protein_encoder()
        if protein_encoder is None or sequences is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels        

        # if isinstance(sequences, list):
        if isinstance(sequences, list) and all(isinstance(seq, list) for seq in sequences):
            raise ValueError("sequence is a list of list - FixMe")

            sequence_features = self.encode_protein_sequences(sequences)
            split_sizes = [len(seq) for seq in sequences]

            print("#"*60)
            print(sequences)
            print(len(sequences))
            print(len(sequences[0]))
            print(split_sizes)
            print(sequence_features.shape)
            print("#"*60)

            sequence_features = torch.split(sequence_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')

            if mm_patch_merge_type == 'flat':
                sequence_features = [x.flatten(0, 1) for x in sequence_features]
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:

            try:
                sequence_features = self.encode_protein_sequences(sequences)
            except Exception as e:
                print(f"Length of sequences: {len(sequences)}")
                print("Length of all sequences:", [len(seq) for seq in sequences])
                print(f"Sequences: {sequences}")
                print(f"Error during encoding: {e}")
                raise e

        # print (f"sequence_features:{sequence_features.shape}")
        # print("#"*60)

        # Handling token indices and embedding for protein sequences
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_sequence_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_sequences = (cur_input_ids == PROTEIN_SEQUENCE_TOKEN_INDEX).sum()
            if num_sequences == 0:
                cur_sequence_features = sequence_features[cur_sequence_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_sequence_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_sequence_idx += 1
                continue

            sequence_token_indices = [-1] + torch.where(cur_input_ids == PROTEIN_SEQUENCE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_no_seq = []
            cur_labels = labels[batch_idx]
            cur_labels_no_seq = []

            for i in range(len(sequence_token_indices) - 1):
                cur_input_ids_no_seq.append(cur_input_ids[sequence_token_indices[i] + 1:sequence_token_indices[i + 1]])
                cur_labels_no_seq.append(cur_labels[sequence_token_indices[i] + 1:sequence_token_indices[i + 1]])

            split_sizes = [x.shape[0] for x in cur_labels_no_seq]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_no_seq))
            cur_input_embeds_no_seq = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_sequences + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_seq[i])
                cur_new_labels.append(cur_labels_no_seq[i])
                if i < num_sequences:
                    cur_sequence_features = sequence_features[cur_sequence_idx]
                    cur_sequence_idx += 1
                    cur_new_input_embeds.append(cur_sequence_features)
                    cur_new_labels.append(torch.full((cur_sequence_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as sequence embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # print(f"final position_ids shape: {position_ids}")  # None 
        # print(f"final attention_mask shape: {attention_mask.shape}")  # torch.Size([32, 30]) 
        # print(f"final labels shape: {new_labels.shape}")  # torch.Size([32, 30]) 
        # print(f"final new_input_embeds shape: {new_input_embeds.shape}")  # torch.Size([32, 30]) 
        # print(f"final past_key_values shape: {past_key_values}")  # None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_protein_tokenizer(self, model_args, tokenizer):
        """
        Initialize the tokenizer for handling protein sequences, adding necessary special tokens.
        
        :param model_args: Arguments specific to the model, including tokenizer configurations.
        :param tokenizer: The tokenizer to be initialized.
        """
        # Handle protein segment token
        if model_args.mm_use_protein_segment_token:
            tokenizer.add_tokens([DEFAULT_PROTEIN_SEGMENT_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        # Handle start and end tokens for protein sequences
        if model_args.mm_use_protein_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_PROT_START_TOKEN, DEFAULT_PROT_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Number of new tokens: {num_new_tokens}.")
        
        # Handle the case where only the protein segment token is used without start/end tokens
        elif model_args.mm_use_protein_segment_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
