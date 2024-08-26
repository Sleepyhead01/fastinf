from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fastapi import FastAPI, Response
from pydantic import BaseModel
import torch
import time

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# start_time = time.time()
# # model = torch.compile(model, mode="max-autotune")
# print(f"Time taken to compile model: {time.time() - start_time}")
app = FastAPI()

class Body(BaseModel):
    text: str

@app.get('/')
def root():
    return Response(f"<h1>A self-documenting API to interact with a GPT2 model and generate text using PyTorch (Running on: {device})</h1>")

@app.post('/generate')
def predict(body: Body):
    input_ids = tokenizer.encode(body.text, return_tensors='pt').to(device)   
    start_time = time.time()

    # Generate text
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    inference_time = time.time() - start_time
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"generated_text": generated_text, "inference_time_seconds": inference_time}

# {
#   "generated_text": "hi, who was born in the city of Kolkata, was a member of the Kolkata-based Kolkata-based Kolkata-based Kolkata-based Kolkata-based Kolkata-based Kolk",
#   "inference_time_seconds": 1.929274320602417
# }

# {
#   "generated_text": "hi, who was born in the city of Kolkata, was a member of the Kolkata-based Kolkata-based Kolkata-based Kolkata-based Kolkata-based Kolkata-based Kolk",
#   "inference_time_seconds": 1.6875169277191162
# }

# {
#   "generated_text": "hi, who was born in the city of Kolkata, was a member of the Kolkata-based Kolkata-based Kolkata-based Kolkata-based Kolkata-based Kolkata-based Kolk",
#   "inference_time_seconds": 1.6436452865600586
# }

