import torch
from einops import rearrange
from transformers import BertModel, BertTokenizer
from typing import List, Union, Optional, Tuple

# Helper function to check if a value exists (is not None)
def exists(val: Optional[Union[torch.Tensor, any]]) -> bool:
    return val is not None

# Singleton globals to hold preloaded BERT model and tokenizer
MODEL: Optional[BertModel] = None
TOKENIZER: Optional[BertTokenizer] = None
BERT_MODEL_DIM: int = 768  # Dimension size of BERT model output

# Function to retrieve the BERT tokenizer
def get_tokenizer() -> BertTokenizer:
    global TOKENIZER
    if not exists(TOKENIZER):
        TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased')
    return TOKENIZER

# Function to retrieve the BERT model, initializes if not loaded yet
def get_bert() -> BertModel:
    global MODEL
    if not exists(MODEL):
        MODEL = BertModel.from_pretrained('bert-base-cased')
        if torch.cuda.is_available():
            MODEL = MODEL.cuda()  # Move to GPU if available
    return MODEL

# Function to tokenize a list or string of texts
def tokenize(texts: Union[str, List[str], Tuple[str]]) -> torch.Tensor:
    if not isinstance(texts, (list, tuple)):
        texts = [texts]  # Convert a single string to a list

    tokenizer = get_tokenizer()

    # Tokenize the input texts, returning PyTorch tensor of token ids
    encoding = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,  # Add [CLS] and [SEP] tokens
        padding=True,  # Pad to the max sequence length
        return_tensors='pt'  # Return as PyTorch tensor
    )

    return encoding.input_ids

# Function to obtain BERT embeddings for tokenized texts
@torch.no_grad()  # No gradients are calculated in this function
def bert_embed(
    token_ids: torch.Tensor,
    return_cls_repr: bool = False,
    eps: float = 1e-8,
    pad_id: int = 0
) -> torch.Tensor:
    """
    Given tokenized input, returns embeddings from BERT.
    
    Parameters:
        token_ids (torch.Tensor): Tensor of token IDs.
        return_cls_repr (bool): If True, return the [CLS] token representation.
        eps (float): Small epsilon for numerical stability.
        pad_id (int): Token ID to treat as padding (defaults to 0).

    Returns:
        torch.Tensor: BERT embeddings (mean or [CLS] token).
    """
    model = get_bert()
    mask = token_ids != pad_id  # Create a mask to ignore padding tokens

    if torch.cuda.is_available():
        token_ids = token_ids.cuda()
        mask = mask.cuda()

    # Obtain the hidden states from the BERT model
    outputs = model(
        input_ids=token_ids,
        attention_mask=mask,
        output_hidden_states=True  # Get hidden states from all layers
    )

    hidden_state = outputs.hidden_states[-1]  # Get the last layer of hidden states

    # If return_cls_repr is True, return the representation of the [CLS] token
    if return_cls_repr:
        return hidden_state[:, 0]  # [CLS] token is the first token

    # If no mask provided, return the mean of all tokens in the sequence
    if not exists(mask):
        return hidden_state.mean(dim=1)

    # Calculate the mean of non-padding tokens by applying the mask
    mask = mask[:, 1:]  # Ignore the [CLS] token in the mask
    mask = rearrange(mask, 'b n -> b n 1')  # Reshape for broadcasting

    # Weighted sum of token representations (ignoring padding)
    numer = (hidden_state[:, 1:] * mask).sum(dim=1)
    denom = mask.sum(dim=1)
    masked_mean = numer / (denom + eps)  # Normalize by sum of mask (to avoid division by zero)

    return masked_mean
