# Imports
import gc
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import transformers
from numba import cuda
import Helpers as helpers
import Prompts as prompts
from tqdm import tqdm
import time
import shutil
import os

def train_model(input_path_training, number_of_reviews, is_Explanation, debug, model_name, new_model, number_of_epochs, max_length_value_training, HUGGINGFACE_TOKEN, SECRET_WAND_TOKEN):
    try:

        print(f"Training the model {new_model}")
        start_time = time.time()

        # Load the training data
        df_reviews = pd.read_json(input_path_training)
        
        # Balance the dataset
        df_reviews = helpers.create_balanced_dataset(df_reviews, number_of_reviews)
        
        # Generate the text prompts with or without explanation
        if is_Explanation:
            if debug: print("Explanation: True")
            df_reviews['text'] = df_reviews.apply(lambda row: prompts.generate_prompt_training_with_Explanation(row), axis=1)
        else:
            if debug: print("Explanation: False")
            df_reviews['text'] = df_reviews.apply(lambda row: prompts.generate_prompt_training(row), axis=1)

        # Convert the DataFrame to a DatasetDict
        train_dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(df_reviews),
        })

        # Hugging Face and WandB login
        os.system(f'huggingface-cli login --token {HUGGINGFACE_TOKEN}')
        wandb.login(key=SECRET_WAND_TOKEN) 
        run = wandb.init(
            project=new_model, 
            job_type="training",
            anonymous="allow"
        )

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # BitsAndBytesConfig for model quantization
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=compute_dtype
        )

        # Load the model with quantization configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model.config.use_cache = False
        
        # LoRA configuration
        lora_alpha = 16
        lora_dropout = 0.1
        lora_r = 64

        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj"]
        )

        # TrainingArguments configuration
        output_dir = "../Models"
        per_device_train_batch_size = 1  # Adjust batch size if necessary
        gradient_accumulation_steps = 1
        optim = "paged_adamw_32bit"
        logging_steps = 500
        learning_rate = 2e-4
        max_grad_norm = 0.3
        warmup_ratio = 0.03
        lr_scheduler_type = "constant"

        training_arguments = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            num_train_epochs=number_of_epochs,
            save_strategy="epoch",
            fp16=True,  # Mixed precision training
            warmup_ratio=warmup_ratio,
            group_by_length=True,
            lr_scheduler_type=lr_scheduler_type,
        )

        max_seq_length = max_length_value_training

        # Initialize the trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset_dict['train'],
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_arguments,
        )

        # Ensure certain modules are in float32
        for name, module in trainer.model.named_modules():
            if "norm" in name:
                module = module.to(torch.float32)
        
        # Start training
        trainer.train()
        trainer.model.save_pretrained(new_model)
        wandb.finish()
        model.eval()

        # Push the model to the Hugging Face Hub
        try:
            trainer.model.push_to_hub(new_model, use_temp_dir=False, private=True)
        except Exception as e:
            print(f"An exception occurred: {e}")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        # Cleanup
        del model, trainer, tokenizer
        
        # Remove model directory
        if os.path.exists(new_model):
            shutil.rmtree(new_model)
        
        # Clear the Hugging Face cache
        hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        if os.path.exists(hf_cache_dir):
            helpers.clear_directory(hf_cache_dir)

        if os.path.exists("../Models"):
            shutil.rmtree("../Models")
            
        # Run garbage collection and clear GPU cache
        gc.collect()
        torch.cuda.empty_cache()
        #print time for training in minutes
        print(f"Training completed in {(time.time() - start_time)/60} minutes")
        time.sleep(10)

# Example usage:
# train_model("path/to/input.json", 1000, True, True, "gpt2", "new_model_name", 3, 512, "your_huggingface_token", "your_wandb_token")

def push_model(HUGGINGFACE_TOKEN, model_name, new_model):
    try:
        #push the model to huggingface
        print(f"Pushing the model {new_model} to Huggingface")

        # Log in to Hugging Face
        os.system(f'huggingface-cli login --token {HUGGINGFACE_TOKEN}')
        
        huggingface_new_model = "sadeep25/" + new_model

        # Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Load the PEFT model
        model = PeftModel.from_pretrained(base_model, huggingface_new_model)
        model = model.merge_and_unload()

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Save the model and tokenizer locally
        save_path = new_model
        model.save_pretrained(save_path) 
        tokenizer.save_pretrained(save_path)

        # Push the model and tokenizer to the Hugging Face Hub
        model.push_to_hub(huggingface_new_model, use_temp_dir=False, private=True)
        tokenizer.push_to_hub(huggingface_new_model, use_temp_dir=False, private=True)
    except Exception as e:
        print(f"An error occurred while pushing the model to Hugging Face: {e}")
        # Ensure cleanup even if an error occurs
    finally:
        # Cleanup resources
        del model, base_model, tokenizer
        
        # Clean up the local save directory
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        # Clear the Hugging Face cache
        hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        if os.path.exists(hf_cache_dir):
            helpers.clear_directory(hf_cache_dir)

        # Run garbage collection and clear GPU cache
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Model {new_model} pushed to Huggingface")
        time.sleep(10)

def inference_model(iterations,HUGGINGFACE_TOKEN, new_model, debug, temperature_value, top_p_value, input_path_validataion, is_Explanation, max_length_value_inference):
    try:

        print(f"Starting inference for model {new_model}")
        start_time = time.time()

        # Log in to Hugging Face
        os.system(f'huggingface-cli login --token {HUGGINGFACE_TOKEN}')

        model_name = "sadeep25/" + new_model
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",
            torch_dtype=torch.float16
        )

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )
        
        for iteration in range(1, (iterations+1)):
            helpers.process_and_save(debug, iteration, new_model, temperature_value, top_p_value, input_path_validataion, "../Output", pipeline, is_Explanation, max_length_value_inference)
        
        helpers.process_files(new_model, "../Output", iterations, temperature_value, top_p_value)
        helpers.process_files_accuracy_measures(new_model, "../Output", iterations, temperature_value, top_p_value)
    
    except Exception as e:
        print(f"An error occurred during inference: {e}")

    finally:
        # Cleanup resources
        del model, tokenizer, pipeline
        
        # Clear the Hugging Face cache
        hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        if os.path.exists(hf_cache_dir):
            helpers.clear_directory(hf_cache_dir)
        
        # Run garbage collection and clear GPU cache
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Inference completed in {(time.time() - start_time)/60} minutes")
