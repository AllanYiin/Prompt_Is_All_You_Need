import gradio as gr
from prompt4all import context
import copy

cxt = context._context()


def image_api(text_input, image_size, temperature=1.2):
    # 創建與API的對話
    _system_prompt = open("./prompts/dalle3.md", encoding="utf-8").read()
    _parameters = copy.deepcopy(cxt.baseChatGpt.API_PARAMETERS)
    _parameters['temperature'] = temperature
    _parameters['max_tokens'] = 100
    results = []
    conversation = [
        {
            "role": "system",
            "content": _system_prompt
        },
        {
            "role": "user",
            "content": text_input
        }
    ]
    image_prompt = cxt.imageChatGpt.post_and_get_answer(conversation, _parameters)
    if ':' in image_prompt:
        image_prompt = ' '.join(image_prompt.split(':')[1:])
    images_urls = cxt.imageChatGpt.generate_images(image_prompt, text_input, image_size)
    return image_prompt, images_urls


def image_panel():
    with gr.Column(variant="panel") as _panel:
        with gr.Row(variant="compact"):
            image_text = gr.Textbox(
                label="請輸入中文的描述",
                show_label=False,
                max_lines=1,
                placeholder="請輸入中文的描述",
                container=False
            )
        image_btn = gr.Button("設計與生成圖片", scale=1)
        image_prompt = gr.Markdown("")
        image_gallery = gr.Gallery(value=None, preview=False, allow_preview=True, show_label=False, columns=[2],
                                   object_fit="cover",
                                   height="auto")
        with gr.Accordion("超參數", open=False):
            temperature2 = gr.Slider(minimum=-0, maximum=2.0, value=0.7, step=0.1, interactive=True,
                                     label="溫度 (Temperature)", )
            image_size = gr.Radio(["1024x1024", "1792x1024", "1024x1792"], label="圖片尺寸", value="1792x1024")
    image_text.submit(image_api, [image_text, image_size, temperature2], [image_prompt, image_gallery])
    image_btn.click(image_api, [image_text, image_size, temperature2], [image_prompt, image_gallery])
    return _panel
