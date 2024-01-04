### 創建甘特圖的指南:

* 甘特圖以關鍵字`gantt`開始，接著是軸和任務的定義。
* 注意：
  * 不要在部分名稱中使用`:`，而應使用其他字符，如`-`。
  * 不要在部分名稱、任務或任務描述中使用`;`、`\`、`|`、`/`，而應使用其他字符，如`-`。
  * 如果用戶提供的文本中包含`:`，則將其更改為`-`。

### 簡單甘特圖示例:

```
gantt
  title Project A
  dateFormat  YYYY-MM-DD
  section Section
  Task 1           :a1, 2023-01-01, 30d
  Task 2           :after a1  , 20d

```


### 關鍵元素:

* `title` — 圖表的標題。
* `dateFormat` — 圖表中使用的日期格式。
* `section` — 圖表中任務的部分或類別。
* 任務描述 — 包括任務名稱、標識符、開始日期和持續時間。

### 持續時間和依賴關係:

* 任務的持續時間以天（30d）、週（5w）、月（2m）或小時（40h）指定。
* 任務之間的依賴關係使用`after`後跟當前任務所依賴的任務標識符來表示。

### 附加參數:

* `excludes` — 從時間表中排除的天數（例如，週末或假期）。
* `todayMarker` — 圖表上當前日期的標記。
* `sectionLabels` — 部分的標籤。

### 樣式:

* 可以使用如`Task :crit, done, 2014-01-06,24d`的語法設置任務的顏色。
* `crit`（關鍵路徑）、`done`（已完成任務）、`active`（活動任務）是任務的樣式類。

### 複雜甘特圖示例:

```
gantt
    title Project X
    dateFormat  YYYY-MM-DD
    excludes    weekends 2023-01-10
    todayMarker off

    section Development
    Research          :active, a1, 2023-01-01, 10d
    Prototyping       :after a1, 20d
    Development       :2023-01-15  , 25d

    section Testing
    Alpha Testing     :b1, after a2, 15d
    Beta Testing      :after b1, 18d

    section Documentation
    Technical Docs    :after b1, 22d
    Manuals           :doc1, after b2, 25d

    section Deployment
    Preparation       :d1, after doc1, 5d
    Release           : 2d
    Post-Release      : 3d

```

此外，還提供了Mermaid主題選擇，包括`default`、`neutral`、`dark`、`forest`和`base`。可以使用`init`指令自定義個別圖表的主題。例如：

``%%{init: {'theme':'forest'}}%%``


