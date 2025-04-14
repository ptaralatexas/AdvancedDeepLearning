from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name
    
    # Convert Path to string to avoid the error
    model_path_str = str(model_path)

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path_str).to(llm.device)
    llm.model.eval()

    return llm




class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full




def format_example(prompt: str, answer: str | float) -> dict[str, str]:
    """
    Improved formatting to ensure consistent numeric representation.
    """
    # Handle the case where answer is already a float
    if isinstance(answer, float):
        float_answer = answer
    else:
        # Try to convert string to float
        try:
            float_answer = float(answer.strip() if hasattr(answer, 'strip') else answer)
        except (ValueError, AttributeError):
            # If conversion fails, use the original answer
            return {
                "question": prompt,
                "answer": f"<answer>{answer}</answer>"
            }
    
    # Format the float answer with appropriate precision
    if float_answer == int(float_answer):
        # For whole numbers, display as integers
        formatted_answer = f"<answer>{int(float_answer)}</answer>"
    else:
        # For decimals, use consistent formatting
        formatted_answer = f"<answer>{float_answer:.6f}</answer>"
        # Remove trailing zeros after decimal point
        formatted_answer = formatted_answer.replace('</answer>', '</answer>').rstrip('0').rstrip('.')
        if formatted_answer.endswith('.'):
            formatted_answer += '0'
        formatted_answer += '</answer>'
    
    return {
        "question": prompt,
        "answer": formatted_answer
    }


def train_model(
    output_dir: str = "homework/sft_model",
    **kwargs,
):
    """
    Enhanced fine-tuning with improved parameters for higher accuracy.
    """
    from pathlib import Path
    import os
    from transformers import Trainer, TrainingArguments, default_data_collator, EarlyStoppingCallback
    from peft import get_peft_model, LoraConfig, TaskType
    
    from .base_llm import BaseLLM
    from .data import Dataset
    
    # Create output directory
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the base model and tokenizer
    llm = BaseLLM()
    model = llm.model
    tokenizer = llm.tokenizer
    
    # More conservative LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        r=12,  # Slight increase from 8
        lora_alpha=40,  # 4x the rank
        lora_dropout=0.07,  # Slightly less dropout
    )
    
    # Get the LoRA model
    model = get_peft_model(model, lora_config)
    
    # Enable input require gradients to avoid bug when using GPU
    model.enable_input_require_grads()
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    # Create training and validation datasets
    train_dataset = TokenizedDataset(tokenizer, Dataset("train"), format_example)
    eval_dataset = TokenizedDataset(tokenizer, Dataset("valid"), format_example)
    
    # Enhanced training arguments for better learning
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=16,  # Slightly smaller than 32
        gradient_accumulation_steps=2,  # Effective batch size of 48
        per_device_eval_batch_size=32,
        num_train_epochs=12,  # More epochs
        learning_rate=2e-4,  # Slightly less than original
        warmup_ratio=0.15,  # More warmup
        lr_scheduler_type="cosine_with_restarts",  # Try with restarts
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,  # Keep gradient clipping
        logging_dir=str(output_dir),
        logging_steps=20,
        report_to="tensorboard",
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        # Keep fp16 disabled if it caused issues
    )
    
    # Initialize the Trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop if not improving
    )
    
    # Start training
    trainer.train()
    
    # Save the final model to the specified directory
    trainer.save_model(str(output_dir))
    
    # Test the trained model
    print("Evaluating the fine-tuned model...")
    test_model(str(output_dir))
    
    return output_dir

def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})

