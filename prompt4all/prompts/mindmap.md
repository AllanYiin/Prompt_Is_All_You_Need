### 使用Mermaid語法創建心智圖的規則:

* 創建心智圖的語法簡單，依賴於縮進來設置層次結構中的級別。
* 創建心智圖時不應使用箭頭，所有連接都基於縮進。
* 不要在圖表的文本中使用不同類型的括號，除非是定義節點形狀的情況。
* 如果用戶在文本中設置了不同類型的括號，應避免使用它們，或者刪除它們或用`|`替換。

### Mermaid心智圖可以顯示不同形狀的節點:

* 方形：`id[text]`
* 圓角方形：`id(I am a rounded square)`
* 圓形：`id((I am a circle))`
* 雷聲：`id))I am a bang((`
* 雲形：`id)I am a cloud(`
* 六邊形：`id{{I am a hexagon}}`
* 默認形狀：I am the default shape

### 創建心智圖的重要規則:

* 根節點語法：
  
  * 只允許一個根節點；不允許多個根節點。
  * 根節點應該有一個有意義的標題，而不僅僅是“TD”。
  * 根節點語法為`root((my title))`。例如，`root((Main Topic))`。
* “Markdown Strings”功能通過提供更多樣化的字符串類型來增強心智圖，支持如粗體和斜體的文本格式選項，並自動在標籤內換行。

```
mindmap
    id1["`**Root** with\na second line\nUnicode works too: 🤓`"]
      id2["`The dog in **the** hog... a *very long text* that wraps to a new line`"]
      id3[Regular labels still works]
```

* 格式化：粗體文本，使用雙星號`**`在文本前後。斜體，使用單星號`*`在文本前後。傳統字符串需要添加標籤才能在節點中換行。然而，markdown字符串在文本過長時自動換行，並允許您僅使用換行字符而不是標籤來開始新行。

### 示例:

用戶請求：“為我展示一個關於心智圖的心智圖”

```
mindmap
root((mindmap))
    Origins
        Long history
        Popularisation
            British popular psychology author Tony Buzan
    Research
        On effectiveness and features
        On Automatic creation
            Uses
                Creative techniques
                Strategic planning
                Argument mapping
    Tools
        Pen and paper
        Mermaid
```

此外，還提供了Mermaid主題選擇，包括`default`、`neutral`、`dark`、`forest`和`base`。可以使用`init`指令自定義個別圖表的主題。例如：

%%{init: {'theme':'forest'}}%%

