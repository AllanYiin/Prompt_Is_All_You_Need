import gradio as gr
from prompt4all import context
from prompt4all.utils.regex_utils import *
import copy
import json

cxt = context._context()


def nlu_api(text_input):
    # 創建與API的對話

    text_inputs = text_input.split('\n')
    text_inputs = [t.strip() for t in text_inputs if len(t.strip()) > 0]
    _parameters = copy.deepcopy(cxt.baseChatGpt.API_PARAMETERS)
    _parameters['temperature'] = 0.1
    results = []
    for txt in text_inputs:
        conversation = [
            {
                "role": "system",
                "content": "請逐一讀取下列句子，每個句子先理解語意後再進行分詞、情感偵測、命名實體偵測以及意圖偵測。分詞結果是指將輸入的文字，先進行語意理解，然後基於語意理解合理性的前提下，將輸入文字進行分詞(tokenize)，若非必要，盡量不要出現超過3個字的詞，然後使用「|」插入至分詞詞彙即構成分詞結果。\n需要偵測的情感類型\n正面情緒(positive_emotions):[自信,快樂,體貼,幸福,信任,喜愛,尊榮,期待,感動,感謝,熱門,獨特,稱讚]\n負面情緒(negative_emotions):[失望,危險,後悔,冷漠,懷疑,恐懼,悲傷,憤怒,擔心,無奈,煩悶,虛假,討厭,貶責,輕視]\n當句子中有符合以上任何情感種類時，請盡可能的將符合的「情感種類」及句子中的那些「觸及到情感種類的內容」成對的列舉出來，一個句子可以觸及不只一種情感。\n需要偵測的實體類型(entities)應該包括但不僅限於[中文人名,中文翻譯人名,外語人名,歌手/樂團/團體名稱,地名/地點,時間,公司機構名/品牌名,商品名,商品規格,化合物名/成分名,歌曲/書籍/作品名稱,其他專有名詞,金額,其他數值]，你可以視情況擴充，\n此外，若是句子中有偵測到符合上述實體類型時，也請盡可能的將符合的「實體類型」及句子中的那些「觸及到實體類型內容｣成對的列舉出來，一個句子可以觸及不只一種實體類型。當你偵測到句子中有要求你代為執行某個任務、或是表達自己想要的事物或是行動、或是想查詢某資訊的意圖(intents)時，根據以意圖最普遍的英文講法之「名詞+動詞-ing」的駝峰式命名形式來組成意圖類別(例如使用者說「請幫我訂今天下午5點去高雄的火車票」其意圖類別為TicketOrdering)，及句子中的那些「觸及到意圖類別的內容」成對的列舉出來，一個句子可以觸及不只一種意圖。以下為「張大帥的人生是一張茶几，上面放滿了杯具。而本身就是杯具」的範例解析結果\n"
                           "{\nsentence:  \"張大帥的人生是一張茶几，上面放滿了杯具。而本身就是杯具\",\nsegmented_sentence:  \"張大帥|的|人生|是|一|張|茶几|，|上面|放滿了|杯具|。|而|本身|就是|杯具\",\npositive_emotions:  [\n0:  {\ntype:  \"煩悶\",\ncontent:  \"放滿了杯具\"\n} ,\n1:  {\ntype:  \"無奈\",\ncontent:  \"本身就是杯具\"\n}\n],\nnegative_emotions:  [\n0:  {\ntype:  \"失望\",\ncontent:  \"上面放滿了杯具\"\n} \n],\nentities:  [\n0:  {\ntype:  \"中文人名\",\ncontent:\"張大帥\"\n}\n]\n}\n\r最後將每個句子的解析結果整合成單一json格式，縮進量為1。"
            },
            {
                "role": "user",
                "content": txt
            }
        ]
        jstrs = json_pattern.findall(cxt.baseChatGpt.post_and_get_answer(conversation, _parameters))
        jstrs = jstrs[0] if len(jstrs) == 1 else '[' + ', '.join(jstrs) + ']'
        output_json = json.loads(jstrs)
        results.append(output_json)

        yield json.dumps(results, ensure_ascii=False, indent=3)


def nlu_panel():
    with gr.Column() as _panel:
        gr.Markdown(
            "將文本輸入到下面的方塊中，按下「送出」按鈕將文本連同上述的prompt發送至OpenAI ChatGPT API，然後將返回的JSON顯示在視覺化界面上。")
        with gr.Row():
            nlu_button = gr.Button("送出")
            nlu_pulse_button = gr.Button("停止")
        with gr.Row(elem_id="nlu_row"):
            nlu_inputs = gr.Textbox(lines=15, placeholder="輸入句子...", scale=1)
            nlu_output = gr.JSON(label="回傳的JSON視覺化", container=True, scale=2)

    cancel_handles = []
    inputs_event = nlu_inputs.submit(nlu_api, nlu_inputs, nlu_output)
    click_event = nlu_button.click(nlu_api, nlu_inputs, nlu_output)
    cancel_handles.append(inputs_event)
    cancel_handles.append(click_event)
    nlu_pulse_button.click(fn=None, inputs=None, outputs=None, cancels=cancel_handles)
    nlu_output.change(scroll_to_output=True)
    return _panel
