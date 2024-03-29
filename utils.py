import torch
dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence
def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."

def reward_fn(reward_model, text, response, tokenizer, softmax, pad_token_id, device=None):
    with torch.no_grad():
        
        input_ids=tokenizer.encode(text, return_tensors='pt').to(device)
        output_ids=tokenizer.encode(response, return_tensors='pt').to(device)

        pad_token_id = 0
        reward_model.eval()
        
        # concat_input_ids=torch.cat((input_ids, output_ids), dim=1)
        outputs = reward_model(input_ids) # input_ids (B, L)
        
        logits = outputs[0] # (B, L, V)
        
        #smoothing
        logits = logits - torch.mean(logits, dim=-1, keepdim=True) # (B, L, V)
        
        # truncation
        logits = logits[:,-len(output_ids[0]):, :]
        
        # print("logits length", logits.shape[1])
        # print("output_ids length", output_ids.shape[1])
        
        selection_value = torch.gather(logits, -1, output_ids[:,:, None]).squeeze(-1) # selection value (B, L) & output_ids[..., None] (B, L, 1)
        
        selection_value.masked_fill_(output_ids == pad_token_id, 0.0)

        next_logits = torch.roll(logits, -1, 1)
        
        if softmax:
            next_state_value = torch.sum(torch.softmax(next_logits, dim=-1) * next_logits, dim=-1)
        else:
            next_state_value = next_logits.max(dim=-1)[0] #(B , L)

        next_state_value.masked_fill_(output_ids == pad_token_id, 0.0)
        next_state_value.masked_fill_(output_ids == reward_model.config.eos_token_id, 0.0)
        
        reward=(selection_value - next_state_value).sum()
        # print(next_state_value)
        return reward
