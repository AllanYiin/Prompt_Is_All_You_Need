# Pie chart diagrams[](https://mermaid.js.org/syntax/pie.html#pie-chart-diagrams)

> A pie chart (or a circle chart) is a circular statistical graphic, which is divided into slices to illustrate numerical proportion. In a pie chart, the arc length of each slice (and consequently its central angle and area), is proportional to the quantity it represents. While it is named for its resemblance to a pie which has been sliced, there are variations on the way it can be presented. The earliest known pie chart is generally credited to William Playfair's Statistical Breviary of 1801 -Wikipedia

Mermaid can render Pie Chart diagrams.

##### Code:

**mermaid**

```
pie title Pets adopted by volunteers
    "Dogs" : 386
    "Cats" : 85
    "Rats" : 15
```

## Syntax[](https://mermaid.js.org/syntax/pie.html#syntax)

Drawing a pie chart is really simple in mermaid.

* Start with `pie` keyword to begin the diagram
  * `showData` to render the actual data values after the legend text. This is ***OPTIONAL***
* Followed by `title` keyword and its value in string to give a title to the pie-chart. This is ***OPTIONAL***
* Followed by dataSet. Pie slices will be ordered clockwise in the same order as the labels.
  * `label` for a section in the pie diagram within `" "` quotes.
  * Followed by `:` colon as separator
  * Followed by `positive numeric value` (supported up to two decimal places)

[pie] [showData] (OPTIONAL) [title] [titlevalue] (OPTIONAL) "[datakey1]" : [dataValue1] "[datakey2]" : [dataValue2] "[datakey3]" : [dataValue3] . .

## Example[](https://mermaid.js.org/syntax/pie.html#example)

##### Code:

**mermaid**

```
%%{init: {"pie": {"textPosition": 0.5}, "themeVariables": {"pieOuterStrokeWidth": "5px"}} }%%
pie showData
    title Key elements in Product X
    "Calcium" : 42.96
    "Potassium" : 50.05
    "Magnesium" : 10.01
    "Iron" :  5
```

## Configuration[](https://mermaid.js.org/syntax/pie.html#configuration)

Possible pie diagram configuration parameters:

| Parameter      | Description                                                                                                  | Default value |
| -------------- | ------------------------------------------------------------------------------------------------------------ | ------------- |
| `textPosition` | The axial position of the pie slice labels, from 0.0 at the center to 1.0 at the outside edge of the circle. | `0.75`        |

