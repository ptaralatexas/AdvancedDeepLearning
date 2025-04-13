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


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair for direct completion.
    No chat template, just direct completion with <answer> tags.
    """
    # The answer should be a float inside the answer tags
    try:
        float_answer = float(answer.strip())
        formatted_answer = f"<answer>{float_answer}</answer>"
    except ValueError:
        # Fallback in case answer cannot be converted to float
        formatted_answer = f"<answer>{answer.strip()}</answer>"
        
    return {
        "question": prompt,
        "answer": formatted_answer
    }


def train_model(
    output_dir: str = "homework/sft_model",
    **kwargs,
):
    """
    Fine-tunes the base model using LoRA adapters following the specific instructions.
    """
    from pathlib import Path
    import os
    from transformers import Trainer, TrainingArguments, default_data_collator
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
    
    # Define LoRA configuration as per instructions
    lora_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM,
      bias="none",
      # Use specific layer names instead of "all-linear"
      target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
      r=8,  # Rank to keep model size below 20MB
      lora_alpha=32,  # About 4 times the rank
      lora_dropout=0.1,
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
    
    # Define training arguments based on instructions
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        learning_rate=5e-5,  # A reasonable learning rate
        gradient_checkpointing=True,  # Save GPU memory
        logging_dir=str(output_dir),
        report_to="tensorboard",  # Create tensorboard logs
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False,  # Important for custom datasets
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
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

