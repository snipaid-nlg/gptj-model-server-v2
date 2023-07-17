from potassium import Potassium, Request, Response

from transformers import GPTJConfig, AutoTokenizer, models
from utils import GPTJBlock, GPTJForCausalLM, add_adapters
import torch

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("patching for 8bit...")
    models.gptj.modeling_gptj.GPTJBlock = GPTJBlock  # monkey-patch GPT-J

    print("loading config...")
    config = GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
    model = GPTJForCausalLM(config=config)
    
    print("adding LoRA adapters...")
    add_adapters(model)

    print("loading model to CPU...")
    checkpoint = torch.hub.load_state_dict_from_url('https://huggingface.co/snipaid/gptj-title-teaser-10k/resolve/main/pytorch_model.pt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    print("done")

    # conditionally load to GPU
    if device == "cuda:0":
        print("loading model to GPU...")
        model.cuda()
        print("done")

    # configure padding tokens
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    config.pad_token_id = config.eos_token_id
    tokenizer.pad_token = config.pad_token_id
   
    # build context to return model and tokenizer
    context = {
        "model": model, 
        "tokenizer": tokenizer
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    model = context.get("model")
    outputs = model(prompt)

    return Response(
        json = {"outputs": outputs[0]}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()