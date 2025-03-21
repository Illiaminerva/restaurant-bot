"""
Restaurant recommendation chatbot model based on GPT-2.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List, Optional, Union
import os

# Set memory allocation strategy
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class RestaurantChatbot(nn.Module):
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        # Initialize GPT-2 model and tokenizer
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add special tokens
        special_tokens = {
            "additional_special_tokens": [
                "User:",
                "Assistant:",
                "<|endoftext|>"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Add sentiment analysis head
        self.sentiment_head = nn.Linear(
            self.model.config.n_embd,
            1,  # Single score for sentiment
            bias=True
        )
        
        self.to(device)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True  # Always get hidden states for sentiment
        )
        
        # Get sentiment prediction from last hidden state
        last_hidden = outputs.hidden_states[-1]
        sentiment_logits = self.sentiment_head(last_hidden.mean(dim=1))
        
        result = {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "sentiment_logits": sentiment_logits
        }
        
        if output_hidden_states:
            result["hidden_states"] = outputs.hidden_states
            
        return result
    
    def generate_response(
        self,
        user_input: str,
        max_length: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> str:
        """Generate a response to user input."""
        # Format input
        if not user_input.startswith("User:"):
            user_input = f"User: {user_input}"
        
        # Prepare input for generation
        inputs = self.tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode and format response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:", 1)[1].strip()
        
        return response
    
    def get_sentiment(self, text: str) -> float:
        """Get sentiment score for a piece of text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self(
                **inputs,
                output_hidden_states=True
            )
        
        return outputs["sentiment_logits"].item()
    
    def save_pretrained(self, path: str) -> None:
        """Save model and tokenizer to path."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save sentiment head separately
        torch.save(self.sentiment_head.state_dict(), f"{path}/sentiment_head.pt")
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = None) -> "RestaurantChatbot":
        """Load model from path."""
        instance = cls(model_name=path, device=device)
        
        # Load sentiment head if available
        sentiment_path = f"{path}/sentiment_head.pt"
        if os.path.exists(sentiment_path):
            instance.sentiment_head.load_state_dict(
                torch.load(sentiment_path, map_location=instance.device)
            )
        
        return instance 