"""
LoRA Fine-tuning Script
Author: Adham Aboulkheir
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingConfig:
    base_model: str = "mistralai/Mistral-7B-v0.1"
    output_dir: str = "models/lora_finetuned"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    fp16: bool = True
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
    
    def to_peft_config(self) -> dict:
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "lora_dropout": self.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
    
    def to_training_args(self) -> dict:
        return {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler,
            "fp16": self.fp16,
            "logging_steps": 10,
            "save_steps": 100,
            "evaluation_strategy": "steps",
            "eval_steps": 100,
        }


def prepare_instruction_dataset(examples: List[dict]) -> List[dict]:
    """
    Format examples into instruction-tuning format.
    
    Input:  [{"instruction": "...", "input": "...", "output": "..."}]
    Output: [{"text": "<s>[INST] instruction [/INST] output </s>"}]
    """
    formatted = []
    for ex in examples:
        instruction = ex.get("instruction", "")
        inp = ex.get("input", "")
        output = ex.get("output", "")
        
        if inp:
            text = f"<s>[INST] {instruction}\n\nInput: {inp} [/INST] {output} </s>"
        else:
            text = f"<s>[INST] {instruction} [/INST] {output} </s>"
        
        formatted.append({"text": text})
    return formatted


if __name__ == "__main__":
    print("LoRA Fine-tuning Configuration Demo")
    config = TrainingConfig(base_model="mistralai/Mistral-7B-v0.1", lora_r=16)
    
    print(f"Base model: {config.base_model}")
    print(f"Effective batch size: {config.effective_batch_size()}")
    print(f"\nPEFT config: {config.to_peft_config()}")
    
    # Sample dataset
    examples = [
        {"instruction": "Diagnose this error", "input": "Error E-47", 
         "output": "Power supply fault. Check fuse and power cable."},
        {"instruction": "Explain the solution", "input": "LED flashing red",
         "output": "Device overheating. Clear ventilation and allow cooling."},
    ]
    
    formatted = prepare_instruction_dataset(examples)
    print(f"\nFormatted {len(formatted)} training examples:")
    print(f"  Sample: {formatted[0]['text'][:100]}...")
    
    print("\nNote: Actual training requires transformers + peft + trl packages.")
    print("Install: pip install transformers peft trl bitsandbytes")
