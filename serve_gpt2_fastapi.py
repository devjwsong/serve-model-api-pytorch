from itertools import chain
from pydantic import BaseModel
from fastapi import FastAPI
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import uvicorn
import torch
import argparse


# Definition of a basic text input.
class TextInput(BaseModel):
    user_id: str
    message: str


# Definition of the main inferencer class.
class Inferencer():
    def __init__(self, args):
        self.chat_history = {}

        # Setting the GPU.
        if torch.cuda.is_available() and isinstance(args.gpu, int):
            self.device = torch.device(f"cuda:{args.gpu}")
        else:
            self.device = torch.device("cpu")

        # Setting the tokenizer.
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
        special_tokens = self.tokenizer.special_tokens_map
        self.bos_token = special_tokens['bos_token']
        self.eos_token = special_tokens['eos_token']
        self.sp1_token = special_tokens['additional_special_tokens'][0]
        self.sp2_token = special_tokens['additional_special_tokens'][1]

        vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(vocab)
        self.bos_id = vocab[self.bos_token]
        self.eos_id = vocab[self.eos_token]
        self.sp1_id = vocab[self.sp1_token]
        self.sp2_id = vocab[self.sp2_token]

        # Setting the model.
        self.model = GPT2LMHeadModel.from_pretrained(args.model_path).to(self.device)
        self.model.eval()

        # Decoding parameters.
        self.max_turns = args.max_turns
        self.max_len = self.model.config.n_ctx
        self.top_p = args.top_p
        self.temperature = args.temperature
        self.num_beams = args.num_beams

    # Adding a new message.
    def add_message(self, user_id, speaker_id, message):
        if user_id not in self.chat_history:
            self.chat_history[user_id] = []

        self.chat_history[user_id].append((speaker_id, message))

    # A single prediction.
    async def predict(self, user_id):
        input_hists = []
        for tup in self.chat_history[user_id]:
            token_ids = [self.sp1_id if tup[0] == 1 else self.sp2_id] + self.tokenizer.encode(tup[1])
            input_hists.append(token_ids)

        # Adjusting the length.
        if len(input_hists) >= self.max_turns:
            num_exceeded = len(input_hists) - self.max_turns + 1
            input_hists = input_hists[num_exceeded:]

        # Setting the input ids and type ids.
        input_ids = [self.bos_id] + list(chain.from_iterable(input_hists)) + [self.sp2_id]
        start_sp_id = input_hists[0][0]
        next_sp_id = self.sp1_id if start_sp_id == self.sp2_id else self.sp2_id
        token_type_ids = [[start_sp_id] * len(hist) if h % 2 == 0 else [next_sp_id] * len(hist) for h, hist in enumerate(input_hists)]
        token_type_ids = [start_sp_id] + list(chain.from_iterable(token_type_ids)) + [self.sp2_id]
        input_len = len(input_ids)

        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
        token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(self.device)

        # Getting the output.
        output_ids = self.model.generate(
            input_ids=input_ids, token_type_ids=token_type_ids, pad_token_id=self.eos_id,
            do_sample=True, top_p=self.top_p, max_length=self.max_len, num_beams=self.num_beams, temperature=self.temperature,
            output_hidden_states=True, output_scores=True, return_dict_in_generate=True
        ).sequences
        output_ids = output_ids[0].tolist()[input_len:]
        res = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        self.chat_history[user_id].append((2, res))

        return res
        

app = FastAPI()


# Default page.
@app.get("/")
def index():
    return {'message': "Welcome to the basic GPT2 chit chat API!"}


# Posting one user message.
@app.post("/infer")
async def infer(data: TextInput):
    data = data.dict()

    user_id = data['user_id']
    message = data['message']

    inferencer.add_message(user_id, 1, message)
    response = await inferencer.predict(user_id)

    return {'message': response}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000, help="The port number.")
    parser.add_argument('--model_path', type=str, required=True, help="The path to the model in HuggingFace Hub.")
    parser.add_argument('--gpu', type=int, default=0, help="The index of GPU to use.")
    parser.add_argument('--max_turns', type=int, default=5, help="The maximum number of dialogue histories to include.")
    parser.add_argument('--top_p', type=float, default=1.0, help="The top p value for nucleus sampling.")
    parser.add_argument('--temperature', type=float, default=1.0, help="The temperature value.")
    parser.add_argument('--num_beams', type=int, default=1, help="The number of beams for beam search.")
              
    args = parser.parse_args()

    # Initializing the inferencer.
    inferencer = Inferencer(args)

    # Running the server.
    uvicorn.run(app, host='127.0.0.1', port=args.port)
