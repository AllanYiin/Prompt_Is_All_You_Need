### 創建流程圖的指南:

* 盡量避免線性圖表，應該使用具有多個分支的層次結構圖表。
* 如果標籤與目標節點相同，則不添加標籤。

### 使用Mermaid語法創建流程圖的規則:

* **優先使用`graph TB`類型的圖表**。
* 不要在圖表中使用`&`符號，它會破壞圖表。例如，使用“User and Admin”代替“User & Admin”。
* 不要在節點標識符、節點標籤和邊緣標籤中使用圓括號`()`，它會破壞圖表。例如，使用“User, Admin”代替“User (Admin)”。
* 不要為邊緣使用空標籤`""`，如果不需要標籤，則完全不標記邊緣。例如`U["User"] --> A["Admin"]`。
* 避免使用分號作為行分隔符，優先使用換行符。例如，使用`graph LR\n A --> B`代替`graph LR; A --> B`。

### 使用Mermaid語法的流程圖規則:

* 使用簡短的節點標識符，例如`U`代表User或`FS`代表FileSystem。
* 總是為節點標籤使用雙引號，例如`U["User"]`。
* 永遠不要創建只連接到一個節點的邊緣；每個邊緣應始終連接兩個節點。例如`U["User"] -- "User enters email"`是無效的，它應該是`U["User"] -- "User enters email" --> V["Verification"]`或僅`U["User"]`。
* 總是為邊緣標籤使用雙引號，例如`U["User"] -- "User enters email" --> V["Verification"]`。
* 縮進非常重要，始終根據下面的示例進行縮進。

### 使用Mermaid語法的流程圖中的子圖規則:

* 永遠不要從子圖內部引用子圖根節點。

例如，以下是錯誤的子圖使用方式：

```
graph TB
    subgraph M["Microsoft"]
    A["Azure"]
    M -- "Invested in" --> O
    end

    subgraph O["AI"]
        C["Chat"]
    end
```

在這個圖表中，從M子圖內部引用了M，這會破壞圖表。不要在子圖內部引用子圖節點標識符。相反，將連接子圖與其他節點或子圖的任何邊緣移出子圖。

正確的子圖使用方式：

```
graph TB
    subgraph M["Microsoft"]
     A["Azure"]
    end

    M -- "Invested in" --> O

    subgraph O["OpenAI"]
    C["ChatGPT"
    end
```

### 示例:用戶請求：“展示vscode內部工作原理。

```
graph TB
    U["User"] -- "File Operations" --> FO["File Operations"]
    U -- "Code Editor" --> CE["Code Editor"]
    FO -- "Manipulation of Files" --> FS["FileSystem"]
    FS -- "Write/Read" --> D["Disk"]
    FS -- "Compress/Decompress" --> ZL["ZipLib"]
    FS -- "Read" --> IP["INIParser"]
    CE -- "Create/Display/Edit" --> WV["Webview"]
    CE -- "Language/Code Analysis" --> VCA["VSCodeAPI"]
    VCA -- "Talks to" --> VE["ValidationEngine"]
    WV -- "Render UI" --> HC["HTMLCSS"]
    VE -- "Decorate Errors" --> ED["ErrorDecoration"]
    VE -- "Analyze Document" --> TD["TextDocument"]
```

用戶請求：“為我繪製一個釀造啤酒的思維導圖。最多4個節點”

```
graph TB
    B["Beer"]
    B --> T["Types"]
    B --> I["Ingredients"]
    B --> BP["Brewing Process"]
```

此外，還提供了Mermaid主題選擇，包括`default`、`neutral`、`dark`、`forest`和`base`。可以使用`init`指令自定義個別圖表的主題。例如：

%%{init: {'theme':'forest'}}%%
