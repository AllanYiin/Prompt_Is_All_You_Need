import gradio as gr
from prompt4all import context
import copy
cxt = context._context()


def rewrite_api(text_input, style_name):
    # 創建與API的對話

    style_name = style_name.split('(')[0].strip()
    _parameters = copy.deepcopy(cxt.baseChatGpt.API_PARAMETERS)
    _parameters['temperature'] = 1.2
    _parameters['frequency_penalty'] = 0
    _parameters['presence_penalty'] = 0
    results = []
    conversation = [
        {
            "role": "system",
            "content": "#zh-TW 你是一個寫作高手，你擅長使用{0}的語氣來改寫輸入之文字，並依照語氣風格特性適時加入表情符號、emoji與調整文字排版，無須解釋，直接改寫".format(
                style_name)
        },
        {
            "role": "user",
            "content": text_input
        }
    ]
    streaming_answer = cxt.baseChatGpt.post_and_get_streaming_answer(conversation, _parameters, conversation)
    while True:
        try:
            answer, full_history = next(streaming_answer)
            yield answer
        except StopIteration:
            break


def rewrite_panel():
    with gr.Column(elem_id="col_container") as _panel:
        rewrite_dropdown = gr.Dropdown(
            ["抽象 (Abstract)",
             "冒險 (Adventurous)",
             "比喻體 (Allegorical)",
             "曖昧 (Ambiguous)",
             "擬人化 (Anthropomorphic)",
             "對比 (Antithetical)",
             "領悟 (Aphoristic)",
             "思辯 (Argumentative)",
             "聲音式 (Auditory)",
             "喚醒 (Awakening)",
             "無邊際 (Boundless)",
             "突破 (Breakthrough)",
             "古典 (Classical)",
             "口語 (Colloquial)",
             "逆袭 (Comeback)",
             "喜劇 (Comedic)",
             "舒適 (Comforting)",
             "簡潔 (Concise)",
             "自信 (Confident)",
             "體悟 (Contemplative)",
             "反向思考 (Counterintuitive)",
             "勇敢 (Courageous)",
             "創意無限 (Creative)",
             "深奧 (Cryptic)",
             "可愛 (Cute)",
             "飛舞 (Dancing)",
             "燦爛 (Dazzling)",
             "細緻 (Delicate)",
             "描繪 (Descriptive)",
             "冷漠 (Detached)",
             "保持距離 (Distant)",
             "夢幻 (Dreamy)",
             "優雅 (Elegant)",
             "感性 (Emotional)",
             "迷人 (Enchanting)",
             "無盡 (Endless)",
             "隱喻 (Euphemistic)",
             "精緻 (Exquisite)",
             "充滿信念 (Faithful)",
             "無畏 (Fearless)",
             "無懈可擊 (Flawless)",
             "靈活 (Flexible)",
             "正式 (Formal)",
             "自由 (Free Verse)",
             "未來主義 (Futuristic)",
             "天賦異禀 (Gifted)",
             "壯麗 (Grandiose)",
             "溫馨 (Heartwarming)",
             "豪邁 (Heroic)",
             "幽默 (Humorous)",
             "誇張 (Hyperbolic)",
             "個性化 (Idiomatic)",
             "獨立 (Independent)",
             "強烈 (Intense)",
             "問答 (Interrogative)",
             "疑問 (Interrogative)",
             "道出内心 (Introspective)",
             "反諷 (Ironic)",
             "歡樂 (Joyful)",
             "傳奇 (Legendary)",
             "人生哲理 (Life Wisdom)",
             "抒情 (Lyric)",
             "魔幻 (Magical)",
             "隱喻 (Metonymic)",
             "現代 (Modern)",
             "神秘 (Mysterious)",
             "敘事 (Narrative)",
             "自然主義 (Naturalistic)",
             "高貴 (Noble)",
             "懷舊 (Nostalgic)",
             "客觀 (Objective)",
             "原聲 (Onomatopoeic)",
             "充滿激情 (Passionate)",
             "激情 (Passionate)",
             "個人 (Personal)",
             "哲學 (Philosophical)",
             "淺白 (Plain)",
             "俏皮 (Playful)",
             "詩意 (Poetic)",
             "正能量 (Positive)",
             "實用主義 (Pragmatic)",
             "頌揚 (Praising)",
             "亮麗 (Radiant)",
             "叛逆 (Rebellious)",
             "高雅 (Refined)",
             "文藝復興 (Renaissance)",
             "復古 (Retro)",
             "啟示 (Revelatory)",
             "革命 (Revolutionary)",
             "修辭 (Rhetorical)",
             "諷刺 (Satirical)",
             "科幻 (Science Fiction)",
             "魅惑 (Seductive)",
             "聳人聽聞 (Sensational)",
             "感傷 (Sentimental)",
             "銳利 (Sharp)",
             "疑問 (Skeptical)",
             "社會評論 (Social Commentary)",
             "嚴肅 (Solemn)",
             "心靈 (Soulful)",
             "靈性 (Spiritual)",
             "主觀 (Subjective)",
             "奇幻 (Surreal)",
             "懸疑 (Suspenseful)",
             "象徵 (Symbolic)",
             "道家 (Taoist)",
             "格調 (Tone)",
             "傳統 (Traditional)",
             "超凡脫俗 (Transcendent)",
             "過渡 (Transitional)",
             "流行 (Trendy)",
             "從容 (Unhurried)",
             "奔放 (Unrestrained)",
             "充滿活力 (Vibrant)",
             "漫遊式 (Wandering)",
             "溫暖 (Warm)",
             "充滿智慧 (Wise)",
             "俏皮 (Witty)",
             "瑜珈式 (Yogic)",
             "青春 (Youthful)"], value="正式 (Formal)", multiselect=False, label="改寫文字風格形容詞",
            interactive=True)
        gr.Markdown("將文本輸入到下面的方塊中，選取改寫風格後，點選改寫後即可將文字基於選取風格進行改寫")
        with gr.Row():
            with gr.Column(scale=1):
                rewrite_inputs = gr.Textbox(lines=30, placeholder="輸入句子...")
            with gr.Column(scale=1):
                rewrite_output = gr.Text(label="改寫", interactive=True, lines=30, show_copy_button=True)
        rewrite_button = gr.Button("送出")
    rewrite_inputs.submit(rewrite_api, [rewrite_inputs, rewrite_dropdown], rewrite_output)
    rewrite_button.click(rewrite_api, [rewrite_inputs, rewrite_dropdown], rewrite_output)
    return _panel
