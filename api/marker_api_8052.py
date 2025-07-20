from fastapi import FastAPI
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
import os

# load marker for convert pdf to markdown
config = {
    "output_format": "markdown",
    "use_llm": False,
    "gemini_api_key": "AIzaSyBkRQB1MTbXbHdDvphv1ID5bkejT7QEDyE",
    "gemini_model_name": "gemini-2.5-flash",
}
config_parser = ConfigParser(config)

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=config_parser.get_llm_service(),
)

app = FastAPI()
path = "../web/cache/"


@app.get("/convertPDF2MD", response_model=dict)
def insert_tieu_chi():
    rendered = converter(os.path.join(path, "pdf_cached.pdf"))
    text, _, images = text_from_rendered(rendered)
    with open(
        os.path.join(path, "md_cached.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(text)
    return {"response": True}
