import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from utils import reward_fn, fix_missing_period
import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def build_dataset(config, dataset_name="xsum", input_min_text_length=100, input_max_text_length=900):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="test")
    ds = ds.rename_columns({"document": "doc"})
    
    # only keep those sentences who have less than 1024 words
    ds = ds.filter(lambda x: 100<len(x["doc"].split(' ')) < input_max_text_length, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample['doc']=fix_missing_period(sample['doc'])
        sample['doc']=' '.join(sample['doc'].split('\n'))
        sample['doc']=f'The summary of the news article [{sample["doc"]}] is:'
        sample["input_ids"] = tokenizer.encode(sample["doc"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

config = PPOConfig(
    model_name='gpt2',
    learning_rate=1e-5,
    batch_size=64,
    log_with='wandb'
    )

ref_model_name='EleutherAI/gpt-j-6b'
dataset = build_dataset(config)

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
# ref_tokenizer=AutoTokenizer.from_pretrained(ref_model_name)

tokenizer.pad_token = tokenizer.eos_token

sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 2}

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

    
output_min_length = 16
output_max_length = 32
output_length_sampler = LengthSampler(output_min_length, output_max_length)
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # query=[f'The summary of the news article [{doc}] is:' for doc in batch['query']]
    # query_tensors=[]
    # for doc in query:
    #     tokens = tokenizer(doc, return_tensors="pt")
    #     tokens = {k: v.to(device='cuda') for k, v in tokens.items()}
    #     query_tensors.append(tokens['input_ids'].squeeze())

    #### Get response from gpt2
    response_tensors = []
    rewards=[]
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs).squeeze()[-gen_len:].unsqueeze(0)
        query=query.unsqueeze(0)
        reward=reward_fn(ref_model, query, response, softmax=True, 
                         pad_token_id=tokenizer.pad_token_id, device=device)
        response_tensors.append(response.squeeze()[-gen_len:])
        rewards.append(reward.detach().cpu())
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    #### Compute sentiment score
    # rewards = reward_fn(ref_model, batch["input_ids"], batch["response"], softmax=True)
    # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    # rewards=reward_fn(texts, sent_kwargs)

    #### Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)