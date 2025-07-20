from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="/home/duckq1u/Documents/obsidian_aio/Notebook/Dự án/OCR anh hiếu/fine-tunning/checkpoint-gemma8bit/cheeckpoint-80",  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=8096,
    load_in_8bit=True,
    load_in_4bit=False,
)


def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content


def json_maker(tokenizer):
    md = read_txt(
        "/home/duckq1u/Documents/obsidian_aio/Notebook/Dự án/OCR anh hiếu/web_deploy/web/cache/md_cached.txt"
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

    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # Must add for generation
    )
    return text


json_maker(tokenizer)
