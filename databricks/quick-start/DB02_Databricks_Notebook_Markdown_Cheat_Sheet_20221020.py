# Databricks notebook source
# MAGIC %md
# MAGIC # Intro

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks notebook can include text documentation by changing a cell to a markdown cell using the `%md` magic command. Most of the markdown syntax works for Databricks, but some do not. This tutorial talks about the commonly used markdown syntax for Databricks notebook. We will cover:
# MAGIC 
# MAGIC * How to format text?
# MAGIC * How to create an item list and checklist?
# MAGIC * How to create mathematical equations?
# MAGIC * How to display an image?
# MAGIC * How to link to Databricks notebooks and folders?
# MAGIC 
# MAGIC Let's get started!

# COMMAND ----------

# MAGIC %md
# MAGIC # Section 1: Format Text

# COMMAND ----------

# MAGIC %md
# MAGIC We listed commonly used text styles and the corresponding markdown syntax in the format text section.
# MAGIC * **Headings** are created by adding a pound sign (#) at the beginning.
# MAGIC ```
# MAGIC # Heading 1
# MAGIC ## Heading 2
# MAGIC ### Heading 3
# MAGIC ```
# MAGIC * **Bold** is achieved by adding two stars (`**`) at both ends. `**This is bold**`
# MAGIC * **Italic** is achieved by adding one star (`*`) at both ends. `*This is italicized*`
# MAGIC * **Strikethrough** is achieved by adding two tildes (`~~`) at both ends. `~~This is a strikethrough~~`
# MAGIC * **Blockquote** is achieved by adding one greater than sign (`>`) at the beginning. `> This is a blockquote`
# MAGIC * **Inline code** is achieved by adding one backtick (`` ` ``) at both ends. `` `This is an inline code` `` 
# MAGIC * To escape the backtick in the inline code, use double backticks (`` `` ``). `` ``  `This escapes backtick` `` `` `` ``
# MAGIC * A **code block** is achieved by adding a triple backtick (```` ``` ````) at both ends.
# MAGIC ~~~
# MAGIC ```
# MAGIC This 
# MAGIC is
# MAGIC a
# MAGIC code
# MAGIC blcok
# MAGIC ```
# MAGIC ~~~
# MAGIC * To escape the triple backticks in the code blocks, wrap with four backticks (````` ```` `````) or three tildes (`~~~`).
# MAGIC * **Link** is created by wrapping the text with square brackets and the link with parenthesis. `[GrabNGoInfo](https://grabngoinfo.com/)`
# MAGIC * **Horizontal rule** is created by three dashes (`---`).
# MAGIC * **Highlight** is achieved by setting the background color in the html span tag. `This is <span style="background-color: #FFFF00">highlighted</span>`
# MAGIC * **Color** is achieved by setting the color in the html span tag. `This is <span style="color:red">red</span>`
# MAGIC * To add **space between lines**, we can use the html br tag. `Line above space <br/><br/> Line below space`
# MAGIC * Databricks notebook does not support **emoji** shortcode such as `:heart:`, but we can copy the emoji image and paste it directly to the markdown cell. For example, `GrabNGoInfo is awesome! ❤️`
# MAGIC * **Subscript** is achieved by adding underscore in curly braces. `\\(A{_i}{_j}\\)`
# MAGIC * **Superscript** is achieved by adding caret in curly braces. `\\(A{^i}{^j}\\)`
# MAGIC * **Table** is achieved using the combination of pipes and dashes.
# MAGIC ```
# MAGIC | Column 1 | Column 2 |
# MAGIC | ----------- | ----------- |
# MAGIC | Row 1 | Value 1 |
# MAGIC | Row 2 | Value 2 |
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC This is the examples for each of the text format discussed.

# COMMAND ----------

# Heading 1
## Heading 2
### Heading 3

**This is bold**

*This is italicized*

~~This is a strikethrough~~

> This is a blockquote

`This is an inline code`

`` `This escapes backtick` ``

```
This 
is
a
code
blcok
```

[GrabNGoInfo](https://grabngoinfo.com/)

---

This is <span style="background-color: #FFFF00">highlighted</span>

This is <span style="color:red">red</span>

Line above space <br/><br/> Line below space

GrabNGoInfo is awesome! ❤️

\\(A{_i}{_j}\\)

\\(A{^i}{^j}\\)

| Column 1 | Column 2 |
| ----------- | ----------- |
| Row 1 | Value 1 |
| Row 2 | Value 2 |

# COMMAND ----------

# MAGIC %md
# MAGIC The code renders to the output below.

# COMMAND ----------

# MAGIC %md
# MAGIC # Heading 1
# MAGIC ## Heading 2
# MAGIC ### Heading 3
# MAGIC 
# MAGIC **This is bold**
# MAGIC 
# MAGIC *This is italicized*
# MAGIC 
# MAGIC ~~This is a strikethrough~~
# MAGIC 
# MAGIC > This is a blockquote
# MAGIC 
# MAGIC `This is an inline code`
# MAGIC 
# MAGIC `` `This escapes backtick` ``
# MAGIC 
# MAGIC ```
# MAGIC This 
# MAGIC is
# MAGIC a
# MAGIC code
# MAGIC blcok
# MAGIC ```
# MAGIC 
# MAGIC [GrabNGoInfo](https://grabngoinfo.com/)
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC This is <span style="background-color: #FFFF00">highlighted</span>
# MAGIC 
# MAGIC This is <span style="color:red">red</span>
# MAGIC 
# MAGIC Line above space <br/><br/> Line below space
# MAGIC 
# MAGIC GrabNGoInfo is awesome! ❤️
# MAGIC 
# MAGIC \\(A{_i}{_j}\\)
# MAGIC 
# MAGIC \\(A{^i}{^j}\\)
# MAGIC 
# MAGIC | Column 1 | Column 2 |
# MAGIC | ----------- | ----------- |
# MAGIC | Row 1 | Value 1 |
# MAGIC | Row 2 | Value 2 |

# COMMAND ----------

# MAGIC %md
# MAGIC # Section 2: Item List

# COMMAND ----------

# MAGIC %md
# MAGIC In the item list section, we will talk about creating an ordered list, unordered list, and checklist.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC * An ordered list is created by adding numbers at the beginning.
# MAGIC 
# MAGIC ```
# MAGIC 1. ordered item 1
# MAGIC 2. ordered item 2
# MAGIC 3. ordered item 3
# MAGIC ```
# MAGIC <br/><br/>
# MAGIC 
# MAGIC * An unordered bullet point list is created by adding a dash (-) or a star (*) at the beginning.
# MAGIC 
# MAGIC ```
# MAGIC - bullet point 1
# MAGIC - bullet point  2
# MAGIC - bullet point  3
# MAGIC * bullet point 4
# MAGIC * bullet point 5
# MAGIC * bullet point 6
# MAGIC ```
# MAGIC 
# MAGIC <br/><br/>
# MAGIC * A checklist can be created by adding cross (`&cross;`), check (`&check;`), and underscore (`_`) at the beginning.
# MAGIC 
# MAGIC ```
# MAGIC - &cross; bullet point 1
# MAGIC - _ bullet point  2
# MAGIC - &check; bullet point  3
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC The code renders to the output below.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 1. ordered item 1
# MAGIC 2. ordered item 2
# MAGIC 3. ordered item 3
# MAGIC 
# MAGIC <br><br/>
# MAGIC - bullet point 1
# MAGIC - bullet point  2
# MAGIC - bullet point  3
# MAGIC * bullet point 4
# MAGIC * bullet point 5
# MAGIC * bullet point 6
# MAGIC 
# MAGIC <br><br/>
# MAGIC - &cross; bullet point 1
# MAGIC - _ bullet point  2
# MAGIC - &check; bullet point  3

# COMMAND ----------

# MAGIC %md
# MAGIC To create a list in the cell of a table, use html.
# MAGIC 
# MAGIC ```
# MAGIC | Column 1 | Column 2 | Column 3 |
# MAGIC | ----------- | ----------- | ------------|
# MAGIC | Row 1 | Value 1 | <li> item 1 </li>  <li> item 2 </li>
# MAGIC | Row 2 | Value 2 | <ol> <li> item 1 </li>  <li> item 2 </li> </ol>
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC The code gives us bullet point list and ordered list in a table.

# COMMAND ----------

# MAGIC %md
# MAGIC | Column 1 | Column 2 | Column 3 |
# MAGIC | ----------- | ----------- | ------------|
# MAGIC | Row 1 | Value 1 | <li> item 1 </li>  <li> item 2 </li>
# MAGIC | Row 2 | Value 2 | <ol> <li> item 1 </li>  <li> item 2 </li> </ol>

# COMMAND ----------

# MAGIC %md
# MAGIC To create a nested list, add two spaces in front of a dash (-) or a star (*)
# MAGIC ```
# MAGIC - bullet point 1
# MAGIC   - nested bullet point 1
# MAGIC   - nested bullet point 2    
# MAGIC * bullet point  2
# MAGIC   * nested bullet point 1
# MAGIC   * nested bullet point 2
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC The code generates the nested list below.

# COMMAND ----------

# MAGIC %md
# MAGIC - bullet point 1
# MAGIC   - nested bullet point 1
# MAGIC   - nested bullet point 2    
# MAGIC * bullet point  2
# MAGIC   * nested bullet point 1
# MAGIC   * nested bullet point 2

# COMMAND ----------

# MAGIC %md
# MAGIC # Section 3: Mathematical Equations

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks notebook supports [KaTex](https://katex.org/). Below are examples from the Databricks documentation.
# MAGIC ```
# MAGIC \\(c = \\pm\\sqrt{a^2 + b^2} \\)
# MAGIC 
# MAGIC \\(A{_i}{_j}=B{_i}{_j}\\)
# MAGIC 
# MAGIC $$c = \\pm\\sqrt{a^2 + b^2}$$
# MAGIC 
# MAGIC \\[A{_i}{_j}=B{_i}{_j}\\]
# MAGIC 
# MAGIC \\( f(\beta)= -Y_t^T X_t \beta + \sum log( 1+{e}^{X_t\bullet\beta}) + \frac{1}{2}\delta^t S_t^{-1}\delta\\)
# MAGIC 
# MAGIC where \\(\delta=(\beta - \mu_{t-1})\\)
# MAGIC 
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC This renders to the equations below
# MAGIC 
# MAGIC \\(c = \\pm\\sqrt{a^2 + b^2} \\)
# MAGIC 
# MAGIC \\(A{_i}{_j}=B{_i}{_j}\\)
# MAGIC 
# MAGIC $$c = \\pm\\sqrt{a^2 + b^2}$$
# MAGIC 
# MAGIC \\[A{_i}{_j}=B{_i}{_j}\\]
# MAGIC 
# MAGIC \\( f(\beta)= -Y_t^T X_t \beta + \sum log( 1+{e}^{X_t\bullet\beta}) + \frac{1}{2}\delta^t S_t^{-1}\delta\\)
# MAGIC 
# MAGIC where \\(\delta=(\beta - \mu_{t-1})\\)

# COMMAND ----------

# MAGIC %md
# MAGIC # Section 4: Image

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks markdown can take images from a URL or FileStore.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Access Image Via URL

# COMMAND ----------

# MAGIC %md
# MAGIC To read an image from a URL, we can use an exclamation mark, followed by the same syntax as creating a link. Recall that a link is created by wrapping the text with square brackets and the URL with parenthesis. For example, the code `![Notebook Tips](https://github.com/tmg-ling/mlflow-tmg-ling/blob/main/databricks/ml-lifecyle-workshop/notebook-tips.png)` renders to the image below.

# COMMAND ----------

# MAGIC %md
# MAGIC ![Notebook Tips](https://github.com/tmg-ling/mlflow-tmg-ling/blob/main/databricks/ml-lifecyle-workshop/notebook-tips.png)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also format the appearance of the image using HTML. For example, the code below specifies the image alignment, line height, padding, and image size.
# MAGIC ```
# MAGIC <div style="text-align: center; line-height: 10; padding-top: 30px;  padding-bottom: 30px;">
# MAGIC   <img src="https://grabngoinfo.com/wp-content/uploads/2021/09/cropped-Converted-PNG-768x219.png" alt='GrabNGoInfo Logo' style="width: 500px" >
# MAGIC </div>
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC It renders a smaller image in the center of the cell with some padding spaces.

# COMMAND ----------

# MAGIC %md
# MAGIC To have the image and text in the same cell, just add the image link. For instance, the following code add an image in front of the heading one text.
# MAGIC ```
# MAGIC # ![Image Test](https://github.com/tmg-ling/mlflow-tmg-ling/blob/main/databricks/ml-lifecyle-workshop/notebook-tips.pngj) 
# MAGIC GrabNGoInfo Machine Learning Tutorials
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC To access an image through FileStore, we need first to upload the image. Using `%fs ls`, we can see that the image was saved under `dbfs:/FileStore/tables/`.

# COMMAND ----------

# MAGIC %fs ls FileStore/tables

# COMMAND ----------

# MAGIC %md
# MAGIC # Section 5: Link To Databricks Notebooks And Folders

# COMMAND ----------

# MAGIC %md
# MAGIC The code below can be used to link to other notebooks or folders using relative paths.
# MAGIC ```
# MAGIC <a href="$./myNotebook">Link to notebook in same folder as current notebook</a>
# MAGIC <a href="$../myFolder">Link to folder in parent folder of current notebook</a>
# MAGIC <a href="$./myFolder2/myNotebook2">Link to nested notebook</a>
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC For example, the code `<a href="$./DB01_Databricks Mount To AWS S3 And Import Data">Databricks Mount To AWS S3 And Import Data</a>` render to the link to the notebook called 'DB01_Databricks Mount To AWS S3 And Import Data'.
