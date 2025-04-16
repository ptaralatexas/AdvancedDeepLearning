def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    """
    Generate a dataset for rejection sampling fine-tuning.
    
    Args:
        output_json: Path to save the generated dataset
        oversample: Number of attempts to generate per question
        temperature: Temperature for generation diversity
    """
    import json
    import os
    from tqdm import tqdm
    from pathlib import Path
    
    from .cot import CoTModel
    from .data import Dataset
    
    # Initialize CoTModel for chain-of-thought generation
    model = CoTModel()
    
    # Get training dataset
    trainset = Dataset("train")
    
    # Initialize dataset
    rft_data = []
    
    # Track statistics
    total_questions = 0
    successful_questions = 0
    
    # For each question in the dataset
    for question, true_answer in tqdm(trainset, desc="Generating CoT examples"):
        total_questions += 1
        
        # Convert true answer to float for comparison
        try:
            correct_answer = float(true_answer)
        except ValueError:
            # Skip if answer can't be converted to float
            continue
        
        # Generate multiple completions for the question
        outputs = model.batched_generate(
            [model.format_prompt(question)],
            num_return_sequences=oversample,  # Generate multiple samples
            temperature=temperature,  # Use specified temperature
        )[0]  # Get list for first (and only) prompt
        
        # Track successful generations for this question
        successful_samples = []
        
        # Process each completion
        for i, output in enumerate(outputs):
            try:
                # Extract answer from completion
                extracted_answer = model.parse_answer(output)
                
                # Check if answer is correct (within tolerance)
                relative_error = abs(extracted_answer - correct_answer) / (1 + abs(correct_answer))
                if relative_error < 0.01:  # 1% tolerance
                    # Save successful sample
                    successful_samples.append({
                        "question": question,
                        "reasoning": output,
                        "answer": true_answer
                    })
            except (ValueError, IndexError):
                # Skip if parsing fails
                continue
        
        # If we found at least one successful sample, count the question as successful
        if successful_samples:
            successful_questions += 1
            rft_data.extend(successful_samples)
    
    # Report statistics
    success_rate = successful_questions / total_questions if total_questions > 0 else 0
    print(f"Generated {len(rft_data)} examples from {successful_questions}/{total_questions} questions")
    print(f"Success rate: {success_rate:.2%}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    # Save dataset to file
    with open(output_json, 'w') as f:
        json.dump(rft_data, f, indent=2)
    
    print(f"Dataset saved to {output_json}")
    #return rft_data


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
