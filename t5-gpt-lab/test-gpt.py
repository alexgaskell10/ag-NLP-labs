import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).eval()

def greedy_search(model, tokenizer, input_ids, max_len=41):
    max_len=41
    i, culm_logprobs = 0, 0
    new_input_ids = input_ids
    ## YOUR CODE HERE ##
    while True:
        # Model forward pass
        all_logits = model(new_input_ids).logits

        # GPT generates a logit per token We want the last logit
        logits = all_logits[:,-1,:]

        # Compute log probability
        logprobs = F.log_softmax(logits, dim=-1)

        # Greedy search: take most likely
        next_token = torch.argmax(logits, dim=-1).unsqueeze(1)

        # Cumulate log probabilites 
        culm_logprobs += torch.max(logprobs, dim=-1).values

        # Append to input_ids
        new_input_ids = torch.cat([new_input_ids, next_token], dim=1)

        # Print new output
        print(i, tokenizer.batch_decode(new_input_ids, skip_special_tokens=True)[0])

        # Stopping criteria
        if new_input_ids.size(1) >= max_len:
            break

        i += 1
    return new_input_ids, culm_logprobs


tmp = torch.tensor([[ 1135,  2883,  6155,   351,   616, 13779,  3290]])
tmp2 = torch.tensor([[   40,  2883,  6155,   351,   616, 13779,  3290]])
tmp3 = torch.tensor([[   40,  2883,  6155,   351,   500, 13779,  3290]])

    # max_len=41
    # n_beams = 3
def beam_search(model, tokenizer, input_ids, max_len=41, n_beams=3):
    i = 0
    bsz = input_ids.size(0)
    ## YOUR CODE HERE ##
    scores = torch.full((input_ids.size(0), n_beams), -1e9)
    scores[:,0] = 0
    scores = scores.view(-1,1)
    new_input_ids = input_ids.repeat(n_beams, 1) #torch.cat([tmp, tmp2, tmp3], dim=0)
    while True:
        # Model forward pass
        all_logits = model(new_input_ids).logits

        # GPT generates a logit per token. We want the last logit
        logits = all_logits[:,-1,:]

        # Compute log probability
        logprobs = F.log_softmax(logits, dim=-1)
        
        # Add past scores
        bs_logprobs = logprobs + scores.expand_as(scores)

        # Reshape to beam search size
        vocab_sz = logits.size(-1)
        bs_logprobs = bs_logprobs.reshape(bsz, -1)

        # Beam search: keep n_beams most likely candidates
        beam_scores, indices = torch.topk(bs_logprobs, k=n_beams, dim=-1, sorted=True)

        next_tokens = indices % vocab_sz
        prev_seq_ixs = torch.flatten((indices - next_tokens) // vocab_sz)
        prev_seqs = new_input_ids[prev_seq_ixs]

        new_input_ids = torch.cat([prev_seqs, next_tokens.view(-1,1)], dim=1)
        scores = beam_scores.view(*scores.shape)

        # Print new output
        # print(i, tokenizer.batch_decode(new_input_ids, skip_special_tokens=True)[0])

        i += 1
        # Stopping criteria
        if new_input_ids.size(1) >= max_len:
            break
    return new_input_ids[0], scores[0]

    

# encode context the generation is conditioned on
input_ids = tokenizer.encode('We enjoy walking with my cute dog', return_tensors='pt')

# generate text until the output length (which includes the context length) reaches 50
# greedy_output = model.generate(input_ids, max_length=50)
beam_output = model.generate(
    input_ids,  #torch.cat([tmp, tmp2, tmp3], dim=0), #input_ids,  
    max_length=43, 
    num_beams=3, 
    no_repeat_ngram_size=2,
    early_stopping=True,
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
print(tokenizer.batch_decode(beam_output, skip_special_tokens=True))

# my_output,_ = beam_search(model, tokenizer, input_ids, max_len=43, n_beams=2)
# print(beam_output == my_output)
    
# max_len = 43
# n_beams = 2
def beam_search_no_ngram(model, tokenizer, input_ids, max_len=45, n_beams=2, max_n_gram=2):
    i = 0
    bsz = input_ids.size(0)
    n_gram = max_n_gram
    ## YOUR CODE HERE ##
    scores = torch.full((input_ids.size(0), n_beams), -1e9)
    scores[:,0] = 0
    scores = scores.view(-1,1)
    new_input_ids = input_ids.repeat(n_beams, 1)
    while True:
        # Model forward pass
        all_logits = model(new_input_ids).logits

        # GPT generates a logit per token. We want the last logit
        logits = all_logits[:,-1,:]

        # n-gram penalty    
        # if i >= n_gram:
        gens = new_input_ids
        
        # Obtain all generated n_grams so far
        all_gen_n_grams = torch.cat([gens[:,j:j+n_gram] for j in range(gens.size(1)-n_gram+1)], dim=0)
        
        # Filter for uniques
        gen_n_grams = all_gen_n_grams.unique(dim=0)

        # Compute log probability
        logprobs = F.log_softmax(logits, dim=-1)        

        # For each of the previous generations, obtain restricted tokens and mask the logits accordingly
        # Ideally this would be vectorized by for readability I have done this as a for loop
        masking = {}
        prev_gens = gens[:,-1:]
        for n,gen in enumerate(prev_gens):
            mask = gen == gen_n_grams[:,0]
            to_mask = torch.masked_select(gen_n_grams[:,1], mask)
            logprobs[n, to_mask] = -1e9 
            masking[n] = to_mask            # For debugging

        # Add past scores
        bs_logprobs = logprobs + scores.expand_as(scores)

        # Reshape to beam search size
        vocab_sz = logits.size(-1)
        bs_logprobs = bs_logprobs.reshape(bsz, -1)

        # Beam search: keep n_beams most likely candidates
        beam_scores, indices = torch.topk(bs_logprobs, k=n_beams, dim=-1, sorted=True)

        next_tokens = indices % vocab_sz
        prev_seq_ixs = torch.flatten((indices - next_tokens) // vocab_sz)
        prev_seqs = new_input_ids[prev_seq_ixs]

        new_input_ids = torch.cat([prev_seqs, next_tokens.view(-1,1)], dim=1)
        scores = beam_scores.view(*scores.shape)

        # Print new output
        print(i, tokenizer.batch_decode(new_input_ids, skip_special_tokens=True)[0])

        i += 1
        # Stopping criteria
        if new_input_ids.size(1) >= max_len:
            break
    return new_input_ids[0], scores[0]

x = beam_search_no_ngram(model, tokenizer, input_ids, max_len=43, n_beams=3, max_n_gram=2)
y = beam_output
print(x[0])
print(y)

print('A')