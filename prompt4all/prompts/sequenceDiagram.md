序列圖主要用於可視化和記錄系統的動態行為。它們捕捉不同實體或組件隨時間的互動。序列圖的主要用途和應用包括展示互動、可視化過程、捕捉系統行為、時間排序和識別角色。

```
sequenceDiagram
    Alice->>John: Hello John, how are you?
    John-->>Alice: Great!
    Alice-)John: See you later!
```

### 定義參與者:

* 使用`participant`或`actor`來表示一個參與者。
* 例如：`participant Alice`將創建一個名為Alice的參與者。

### 消息:

* 箭頭用於指示參與者之間消息的方向。
* `->` 實線無箭頭
* `-->` 虛線無箭頭
* `->>` 實線帶箭頭
* `-->>` 虛線帶箭頭
* `-x` 實線帶叉尾
* `--x` 虛線帶叉尾
* `-)` 實線帶開放箭頭（異步）
* `--)` 虛線帶開放箭頭（異步）

### 啟動和停用參與者:

* 使用`activate`和`deactivate`來表示參與者活動的時期。
* 例如：`activate Alice`將標記Alice為活動狀態，`deactivate Alice`將停用她。
* `activate` / `deactivate`應始終成對使用，否則圖表將無效。

### 備註:

* 使用`Note right of`或`Note left of`在參與者的特定側面添加備註。
* 例如：`Note right of Alice: Alice thinks`在Alice的右側添加備註。

### 替代路徑（Alt/Else）:

* 使用`alt`和`else`來表示互動中的替代分支。
* 例如：

```
alt condition1
    Alice->Bob: message1
else condition2
    Alice->Bob: message2
end
```

### 平行動作（Par）:

* 使用`par`表示平行過程。
* 例如：

```
par
  Alice->Bob: message1
  Alice->Eve: message2
end
```

### 循環（Loop）:

* 使用`loop`表示重複動作。
* 例如：

```
loop every day
  Alice->Bob: message
end
```

### 可選塊（Opt）:

* 使用`opt`表示可選動作。
* 例如：

```
opt optional
  Alice->Bob: message
end
```

請確保不要使用`opt`、`par`、`alt`、`group`、`else`、`end`等作為參與者名稱或標識符。

此外，還提供了Mermaid主題選擇，包括`default`、`neutral`、`dark`、`forest`和`base`。可以使用`init`指令自定義個別圖表的主題。例如：

%%{init: {'theme':'forest'}}%%

