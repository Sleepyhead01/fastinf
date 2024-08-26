from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fastapi import FastAPI, Response
from pydantic import BaseModel
import torch
import time
from vllm import LLM, SamplingParams

# choosing the large language model
llm = LLM(model="gpt2")

# setting the parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.90,max_tokens = 50)

# # Check if CUDA is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Model will be loaded on device: {device}")

# # Load pre-trained model and tokenizer
# model_name = 'gpt2'
# start_time = time.time()
# model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# print(f"Time taken to load model and tokenizer: {time.time() - start_time:.2f} seconds")

# start_time = time.time()
# model = torch.compile(model, mode="max-autotune")
# print(f"Time taken to compile model: {time.time() - start_time:.2f} seconds")

app = FastAPI()

class Body(BaseModel):
    text: str

@app.get('/')
def root():
    return Response(f"<h1>A self-documenting API to interact with a GPT2 model and generate text using vLLM</h1>")

@app.post('/generate')
def predict(body: Body):
    # defining our prompt
    prompt = body.text
    start_time = time.time()
    # generating the answer
    answer = llm.generate(prompt,sampling_params)

    # getting the generated text out from the answer variable
    generated_text = answer[0].outputs[0].text
    inference_time = time.time() - start_time
    return {"generated_text": generated_text, "inference_time_seconds": inference_time}