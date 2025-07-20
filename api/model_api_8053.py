from fastapi import FastAPI
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer
import torch

torch.set_num_threads(3)  # Hoặc số luồng bạn muốn
torch.set_num_interop_threads(3)

if True:
    model, tokenizer = FastModel.from_pretrained(
        model_name="../model/",  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=8096,
        load_in_8bit=True,
        load_in_4bit=False,
    )


def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content


def gen_json(model, tokenizer):
    """
    Generate JSON from the cached Markdown file using the tokenizer.
    """
    md = read_txt(
        "../web/cache/md_cached.txt"  # Path to your cached Markdown file
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"<context>{str(md)}</context>\n<requirement>\n Return information extracted from Markdown as JSON for me </requirement>\n\n\n",
                }
            ],
        }
    ]

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # Must add for generation
    )
    outputs = model.generate(
        **tokenizer([text], return_tensors="pt").to("cuda"),
        max_new_tokens=8096,  # Increase for longer outputs!
        # Recommended Gemma-3 settings!
        temperature=0.2,
        top_p=0.95,
        top_k=64,

        num_beams=1,
        do_sample=True,

        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )
    return tokenizer.batch_decode(outputs)


app = FastAPI()

@app.get("/extracted", response_model=dict)
def insert_tieu_chi():
    print("Generating JSON from Markdown...")
    json_output = gen_json(model, tokenizer)
    return {"response": json_output}
