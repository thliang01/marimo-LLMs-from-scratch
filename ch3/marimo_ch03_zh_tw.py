# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch==2.9.1",
# ]
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <table style="width:100%">
    <tr>
    <td style="vertical-align:middle; text-align:left;">
    <font size="2">
    <a href="http://mng.bz/orYv">從零開始構建大型語言模型</a>一書的補充程式碼，作者為 <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>
    <br>程式碼儲存庫：<a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>
    </font>
    </td>
    <td style="vertical-align:middle; text-align:left;">
    <a href="http://mng.bz/orYv"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover-small.webp" width="100px"></a>
    </td>
    </tr>
    </table>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 第三章：實作注意力機制
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    本筆記本中使用的套件：
    """)
    return


@app.cell
def _():
    from importlib.metadata import version

    print("torch version:", version("torch"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 本章涵蓋注意力機制，這是大型語言模型的引擎：
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/01.webp?123" width="500px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/02.webp" width="600px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.1 建模長序列的問題
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 本節沒有程式碼
    - 由於來源語言和目標語言之間的語法結構差異，逐字翻譯文本是不可行的：
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/03.webp" width="400px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 在引入 transformer 模型之前，編碼器-解碼器 RNN 常用於機器翻譯任務
    - 在這種設定中，編碼器處理來自來源語言的標記序列，使用隱藏狀態（神經網路內部的一種中間層）來生成整個輸入序列的壓縮表示：
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/04.webp" width="500px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.2 使用注意力機制捕捉資料依賴性
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 本節沒有程式碼
    - 透過注意力機制，網路的文本生成解碼器部分能夠選擇性地存取所有輸入標記，這意味著某些輸入標記在生成特定輸出標記時比其他標記更重要：
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/05.webp" width="500px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Transformer 中的自注意力是一種技術，旨在透過使序列中的每個位置能夠與同一序列中的每個其他位置互動並確定其相關性，來增強輸入表示
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/06.webp" width="300px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.3 使用自注意力關注輸入的不同部分
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3.1 沒有可訓練權重的簡單自注意力機制
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 本節解釋了一個非常簡化的自注意力變體，它不包含任何可訓練的權重
    - 這純粹是為了說明目的，並不是 transformer 中使用的注意力機制
    - 下一節（3.3.2）將擴展這個簡單的注意力機制以實現真正的自注意力機制
    - 假設我們有一個輸入序列 $x^{(1)}$ 到 $x^{(T)}$
      - 輸入是一段文本（例如，像「Your journey starts with one step」這樣的句子），已經轉換為第二章中描述的標記嵌入
      - 例如，$x^{(1)}$ 是一個 d 維向量，代表單詞「Your」，依此類推
    - **目標：** 為 $x^{(1)}$ 到 $x^{(T)}$ 中的每個輸入序列元素 $x^{(i)}$ 計算上下文向量 $z^{(i)}$（其中 $z$ 和 $x$ 具有相同的維度）
        - 上下文向量 $z^{(i)}$ 是輸入 $x^{(1)}$ 到 $x^{(T)}$ 的加權和
        - 上下文向量對於特定輸入是「上下文」特定的
          - 我們不使用 $x^{(i)}$ 作為任意輸入標記的佔位符，而是考慮第二個輸入 $x^{(2)}$
          - 繼續一個具體的例子，而不是佔位符 $z^{(i)}$，我們考慮第二個輸出上下文向量 $z^{(2)}$
          - 第二個上下文向量 $z^{(2)}$ 是所有輸入 $x^{(1)}$ 到 $x^{(T)}$ 相對於第二個輸入元素 $x^{(2)}$ 的加權和
          - 注意力權重是確定在計算 $z^{(2)}$ 時每個輸入元素貢獻多少加權和的權重
          - 簡而言之，將 $z^{(2)}$ 視為 $x^{(2)}$ 的修改版本，它還包含有關與手頭任務相關的所有其他輸入元素的資訊
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/07.webp" width="400px">

    - (請注意，此圖中的數字被截斷為小數點後一位以減少視覺混亂；同樣，其他圖也可能包含截斷的值)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 按照慣例，未正規化的注意力權重稱為**「注意力分數」**，而正規化的注意力分數（總和為 1）稱為**「注意力權重」**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 下面的程式碼逐步演示上圖

    <br>

    - **步驟 1：** 計算未正規化的注意力分數 $\omega$
    - 假設我們使用第二個輸入標記作為查詢，即 $q^{(2)} = x^{(2)}$，我們透過點積計算未正規化的注意力分數：
        - $\omega_{21} = x^{(1)} q^{(2)\top}$
        - $\omega_{22} = x^{(2)} q^{(2)\top}$
        - $\omega_{23} = x^{(3)} q^{(2)\top}$
        - ...
        - $\omega_{2T} = x^{(T)} q^{(2)\top}$
    - 以上，$\omega$ 是希臘字母「omega」，用於象徵未正規化的注意力分數
        - $\omega_{21}$ 中的下標「21」表示輸入序列元素 2 被用作針對輸入序列元素 1 的查詢
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 假設我們有以下已經嵌入為 3 維向量的輸入句子，如第二章所述（為了說明目的，我們在這裡使用非常小的嵌入維度，以便它可以在頁面上顯示而不換行）：
    """)
    return


@app.cell
def _():
    import torch

    inputs = torch.tensor(
      [[0.43, 0.15, 0.89], # Your     (x^1)
       [0.55, 0.87, 0.66], # journey  (x^2)
       [0.57, 0.85, 0.64], # starts   (x^3)
       [0.22, 0.58, 0.33], # with     (x^4)
       [0.77, 0.25, 0.10], # one      (x^5)
       [0.05, 0.80, 0.55]] # step     (x^6)
    )
    return inputs, torch


@app.cell
def _(inputs):
    inputs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - (在本書中，我們遵循常見的機器學習和深度學習慣例，其中訓練範例表示為行，特徵值表示為列；在上面顯示的張量的情況下，每一行代表一個單詞，每一列代表一個嵌入維度)

    - 本節的主要目標是示範如何使用第二個輸入序列 $x^{(2)}$ 作為查詢來計算上下文向量 $z^{(2)}$

    - 該圖描述了此過程的初始步驟，即透過點積運算計算 $x^{(2)}$ 與所有其他輸入元素之間的注意力分數 ω
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/08.webp" width="400px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 我們使用輸入序列元素 2，$x^{(2)}$，作為範例來計算上下文向量 $z^{(2)}$；稍後在本節中，我們將推廣這一點以計算所有上下文向量。
    - 第一步是透過計算查詢 $x^{(2)}$ 和所有其他輸入標記之間的點積來計算未正規化的注意力分數：
    """)
    return


@app.cell
def _(inputs, torch):
    query = inputs[1]  # 2nd input token is the query
    attn_scores_2 = torch.empty(inputs.shape[0])
    for _i, _x_i in enumerate(inputs):
        attn_scores_2[_i] = torch.dot(_x_i, query)
    print(attn_scores_2)  # dot product (transpose not necessary here since they are 1-dim vectors)
    return attn_scores_2, query


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 附註：點積本質上是將兩個向量逐元素相乘並對結果乘積求和的簡寫：
    """)
    return


@app.cell
def _(inputs, query, torch):
    res = 0.0
    for idx, element in enumerate(inputs[0]):
        res = res + inputs[0][idx] * query[idx]
    print(res)
    print(torch.dot(inputs[0], query))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **步驟 2：** 正規化未正規化的注意力分數（「omega」，$\omega$），使它們總和為 1
    - 這是一個簡單的方法來正規化未正規化的注意力分數，使其總和為 1（一種慣例，對解釋有用，對訓練穩定性也很重要）：
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/09.webp" width="500px">
    """)
    return


@app.cell
def _(attn_scores_2):
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 然而，在實踐中，使用 softmax 函數進行正規化很常見且受到推薦，它更能處理極值，並在訓練期間具有更理想的梯度屬性。
    - 這是 softmax 函數的一個簡單實作，用於縮放，它還正規化向量元素，使其總和為 1：
    """)
    return


@app.cell
def _(attn_scores_2, torch):
    def softmax_naive(x):
        return torch.exp(x) / torch.exp(x).sum(dim=0)

    attn_weights_2_naive = softmax_naive(attn_scores_2)

    print("Attention weights:", attn_weights_2_naive)
    print("Sum:", attn_weights_2_naive.sum())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 上面的簡單實作對於大或小的輸入值可能會遇到數值不穩定問題，因為溢出和下溢問題
    - 因此，在實踐中，建議使用 PyTorch 的 softmax 實作，它已經針對效能進行了高度優化：
    """)
    return


@app.cell
def _(attn_scores_2, torch):
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())
    return (attn_weights_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **步驟 3**：透過將嵌入的輸入標記 $x^{(i)}$ 與注意力權重相乘並對結果向量求和來計算上下文向量 $z^{(2)}$：
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/10.webp" width="500px">
    """)
    return


@app.cell
def _(attn_weights_2, inputs, torch):
    query_1 = inputs[1]
    context_vec_2 = torch.zeros(query_1.shape)
    for _i, _x_i in enumerate(inputs):
        context_vec_2 = context_vec_2 + attn_weights_2[_i] * _x_i
    print(context_vec_2)
    return (context_vec_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3.2 計算所有輸入標記的注意力權重
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 推廣到所有輸入序列標記：

    - 在上面，我們計算了輸入 2 的注意力權重和上下文向量（如下圖中突出顯示的行所示）
    - 接下來，我們將推廣此計算以計算所有注意力權重和上下文向量
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/11.webp" width="400px">

    - (請注意，此圖中的數字被截斷為小數點後兩位以減少視覺混亂；每行中的值應加起來為 1.0 或 100%；同樣，其他圖中的數字也被截斷)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 在自注意力中，過程從計算注意力分數開始，這些分數隨後被正規化以得出總和為 1 的注意力權重
    - 然後透過對輸入進行加權求和來利用這些注意力權重生成上下文向量
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/12.webp" width="400px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 將先前的**步驟 1** 應用於所有成對元素以計算未正規化的注意力分數矩陣：
    """)
    return


@app.cell
def _(inputs, torch):
    attn_scores = torch.empty(6, 6)
    for _i, _x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[_i, j] = torch.dot(_x_i, x_j)
    print(attn_scores)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 我們可以透過矩陣乘法更有效地實現上述相同的結果：
    """)
    return


@app.cell
def _(inputs):
    attn_scores_1 = inputs @ inputs.T
    print(attn_scores_1)
    return (attn_scores_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 與先前的**步驟 2** 類似，我們正規化每一行，使每行中的值總和為 1：
    """)
    return


@app.cell
def _(attn_scores_1, torch):
    attn_weights = torch.softmax(attn_scores_1, dim=-1)
    print(attn_weights)
    return (attn_weights,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 快速驗證每行中的值確實總和為 1：
    """)
    return


@app.cell
def _(attn_weights):
    row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
    print("Row 2 sum:", row_2_sum)

    print("All row sums:", attn_weights.sum(dim=-1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 應用先前的**步驟 3** 來計算所有上下文向量：
    """)
    return


@app.cell
def _(attn_weights, inputs):
    all_context_vecs = attn_weights @ inputs
    print(all_context_vecs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 作為健全性檢查，先前計算的上下文向量 $z^{(2)} = [0.4419, 0.6515, 0.5683]$ 可以在上面的第 2 行中找到：
    """)
    return


@app.cell
def _(context_vec_2):
    print("Previous 2nd context vector:", context_vec_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.4 實作具有可訓練權重的自注意力
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 一個概念框架，說明本節開發的自注意力機制如何整合到本書和本章的整體敘述和結構中
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/13.webp" width="400px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.4.1 逐步計算注意力權重
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 在本節中，我們正在實作原始 transformer 架構、GPT 模型和大多數其他流行的大型語言模型中使用的自注意力機制
    - 這種自注意力機制也稱為「縮放點積注意力」
    - 整體思想與之前類似：
      - 我們想要計算上下文向量作為特定於某個輸入元素的輸入向量的加權和
      - 為了上述目的，我們需要注意力權重
    - 如您所見，與之前介紹的基本注意力機制相比，只有輕微的差異：
      - 最顯著的差異是引入了在模型訓練期間更新的權重矩陣
      - 這些可訓練的權重矩陣至關重要，這樣模型（特別是模型內部的注意力模組）可以學習產生「良好」的上下文向量
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/14.webp" width="600px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 逐步實作自注意力機制，我們將首先引入三個訓練權重矩陣 $W_q$、$W_k$ 和 $W_v$
    - 這三個矩陣用於透過矩陣乘法將嵌入的輸入標記 $x^{(i)}$ 投影到查詢、鍵和值向量：

      - *Q* 查詢向量：$q^{(i)} = x^{(i)}\,W_q$
      - *K* 鍵向量：$k^{(i)} = x^{(i)}\,W_k$
      - *V* 值向量：$v^{(i)} = x^{(i)}\,W_v$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 輸入 $x$ 和查詢向量 $q$ 的嵌入維度可以相同或不同，具體取決於模型的設計和具體實作
    - 在 GPT 模型中，輸入和輸出維度通常相同，但為了說明目的，為了更好地跟蹤計算，我們在這裡選擇不同的輸入和輸出維度：
    """)
    return


@app.cell
def _(inputs):
    x_2 = inputs[1] # second input element
    d_in = inputs.shape[1] # the input embedding size, d=3
    d_out = 2 # the output embedding size, d=2
    return d_in, d_out, x_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 下面，我們初始化三個權重矩陣；請注意，為了說明目的，我們設定 `requires_grad=False` 以減少輸出中的混亂，但如果我們要將權重矩陣用於模型訓練，我們會設定 `requires_grad=True` 以在模型訓練期間更新這些矩陣
    """)
    return


@app.cell
def _(d_in, d_out, torch):
    torch.manual_seed(42)

    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    return W_key, W_query, W_value


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 接下來我們計算查詢、鍵和值向量 (**Q**, **K**, **V**)：
    """)
    return


@app.cell
def _(W_key, W_query, W_value, x_2):
    query_2 = x_2 @ W_query # _2 because it's with respect to the 2nd input element
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value

    print(query_2)
    return (query_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 如下所示，我們成功地將 6 個輸入標記從 3D 投影到 2D 嵌入空間：
    """)
    return


@app.cell
def _(W_key, W_value, inputs):
    keys = inputs @ W_key
    values = inputs @ W_value

    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)
    return keys, values


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 在下一步，**步驟 2**，我們透過計算查詢和每個 *k* 鍵向量之間的點積來計算未正規化的注意力分數：
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/15.webp" width="600px">
    """)
    return


@app.cell
def _(keys, query_2):
    keys_2 = keys[1] # Python starts index at 0
    attn_score_22 = query_2.dot(keys_2)
    print(attn_score_22)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 由於我們有 6 個輸入，對於給定的查詢向量，我們有 6 個注意力分數：
    """)
    return


@app.cell
def _(keys, query_2):
    attn_scores_2_1 = query_2 @ keys.T  # All attention scores for given query
    print(attn_scores_2_1)
    return (attn_scores_2_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/16.webp" width="600px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 接下來，在**步驟 3**，我們使用之前使用的 softmax 函數計算注意力權重（總和為 1 的正規化注意力分數）
    - 與之前的不同之處在於，我們現在透過將注意力分數除以嵌入維度的平方根 $\sqrt{d_k}$（即 `d_k**0.5`）來縮放注意力分數：
    """)
    return


@app.cell
def _(attn_scores_2_1, keys, torch):
    d_k = keys.shape[1]
    attn_weights_2_1 = torch.softmax(attn_scores_2_1 / d_k ** 0.5, dim=-1)
    print(attn_weights_2_1)
    return (attn_weights_2_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/17.webp" width="600px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 在**步驟 4**，我們現在計算輸入查詢向量 2 的上下文向量：
    """)
    return


@app.cell
def _(attn_weights_2_1, values):
    context_vec_2_1 = attn_weights_2_1 @ values
    print(context_vec_2_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.4.2 實作一個緊湊的 SelfAttention 類別
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 將所有內容放在一起，我們可以如下實作自注意力機制：
    """)
    return


@app.cell
def _(d_in, d_out, inputs, torch):
    import torch.nn as nn

    class SelfAttention_v1(nn.Module):

        def __init__(self, d_in, d_out):
            super().__init__()
            self.W_query = nn.Parameter(torch.rand(d_in, d_out))
            self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
            self.W_value = nn.Parameter(torch.rand(d_in, d_out))

        def forward(self, x):
            keys = x @ self.W_key
            queries = x @ self.W_query
            values = x @ self.W_value

            attn_scores = queries @ keys.T # omega
            attn_weights = torch.softmax(
                attn_scores / keys.shape[-1]**0.5, dim=-1
            )

            context_vec = attn_weights @ values
            return context_vec

    torch.manual_seed(42)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))
    return (nn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/18.webp" width="400px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 我們可以使用 PyTorch 的 Linear 層簡化上述實作，如果我們停用偏差單元(bias units)，它相當於矩陣乘法
    - 使用 `nn.Linear` 而不是我們手動的 `nn.Parameter(torch.rand(...))` 方法的另一個很大的優勢是 `nn.Linear` 具有首選的權重初始化方案，可以實現更穩定的模型訓練
    """)
    return


@app.cell
def _(d_in, d_out, inputs, nn, torch):
    class SelfAttention_v2(nn.Module):

        def __init__(self, d_in, d_out, qkv_bias=False):
            super().__init__()
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        def forward(self, x):
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)

            attn_scores = queries @ keys.T
            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

            context_vec = attn_weights @ values
            return context_vec

    torch.manual_seed(42)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))
    return (sa_v2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 請注意，`SelfAttention_v1` 和 `SelfAttention_v2` 給出不同的輸出，因為它們對權重矩陣使用不同的初始權重
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.5 使用因果注意力隱藏未來的詞
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 在因果注意力中，對角線上方的注意力權重被遮蔽，確保對於任何給定的輸入，大型語言模型在使用注意力權重計算上下文向量時無法利用未來的標記
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/19.webp" width="400px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.5.1 應用因果注意力遮罩
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 在本節中，我們將先前的自注意力機制轉換為因果自注意力機制
    - 因果自注意力確保模型對序列中某個位置的預測僅依賴於先前位置的已知輸出，而不依賴於未來位置
    - 用更簡單的話說，這確保了每個下一個詞的預測應該只依賴於前面的詞
    - 為了實現這一點，對於每個給定的標記，我們遮蔽掉未來的標記（輸入文本中當前標記之後的標記）：
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/20.webp" width="600px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 為了說明和實作因果自注意力，讓我們使用上一節的注意力分數和權重：
    """)
    return


@app.cell
def _(inputs, sa_v2, torch):
    # Reuse the query and key weight matrices of the
    # SelfAttention_v2 object from the previous section for convenience
    queries = sa_v2.W_query(inputs)
    keys_1 = sa_v2.W_key(inputs)
    attn_scores_3 = queries @ keys_1.T
    attn_weights_1 = torch.softmax(attn_scores_3 / keys_1.shape[-1] ** 0.5, dim=-1)
    print(attn_weights_1)
    return attn_scores_3, attn_weights_1, keys_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 遮蔽掉未來注意力權重的最簡單方法是透過 PyTorch 的 tril 函數創建一個遮罩，其中主對角線下方（包括對角線本身）的元素設定為 1，主對角線上方的元素設定為 0：
    """)
    return


@app.cell
def _(attn_scores_3, torch):
    context_length = attn_scores_3.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print(mask_simple)
    return context_length, mask_simple


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 然後，我們可以將注意力權重與此遮罩相乘，以將對角線上方的注意力分數歸零：
    """)
    return


@app.cell
def _(attn_weights_1, mask_simple):
    masked_simple = attn_weights_1 * mask_simple
    print(masked_simple)
    return (masked_simple,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 但是，如果像上面那樣在 softmax 之後應用遮罩，它會破壞 softmax 創建的機率分佈
    - Softmax 確保所有輸出值總和為 1
    - 在 softmax 之後進行遮罩需要重新正規化輸出以再次總和為 1，這會使過程變得複雜，並可能導致意外的影響
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 為了確保行總和為 1，我們可以按如下方式正規化注意力權重：
    """)
    return


@app.cell
def _(masked_simple):
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums
    print(masked_simple_norm)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 雖然我們現在在技術上已經完成了因果注意力機制的實作，但讓我們簡要地看一下實現上述相同目標的更有效方法
    - 因此，我們可以在未正規化的注意力分數進入 softmax 函數之前，用負無窮大遮蔽對角線上方的未正規化注意力分數，而不是將對角線上方的注意力權重歸零並重新正規化結果：
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/21.webp" width="450px">
    """)
    return


@app.cell
def _(attn_scores_3, context_length, torch):
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores_3.masked_fill(mask.bool(), -torch.inf)
    print(masked)
    return (masked,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 如下所示，現在每行中的注意力權重再次正確地總和為 1：
    """)
    return


@app.cell
def _(keys_1, masked, torch):
    attn_weights_3 = torch.softmax(masked / keys_1.shape[-1] ** 0.5, dim=-1)
    print(attn_weights_3)
    return (attn_weights_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.5.2 使用 dropout 遮蔽額外的注意力權重
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 此外，我們還應用 dropout 來減少訓練期間的過度擬合
    - Dropout 可以應用於多個地方：
      - 例如，在計算注意力權重之後；
      - 或在將注意力權重與值向量相乘之後
    - 在這裡，我們將在計算注意力權重之後應用 dropout 遮罩，因為這更常見

    - 此外，在這個具體的範例中，我們使用 50% 的 dropout 率，這意味著隨機遮蔽掉一半的注意力權重。（當我們稍後訓練 GPT 模型時，我們將使用較低的 dropout 率，例如 0.1 或 0.2
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/22.webp" width="400px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 如果我們應用 0.5（50%）的 dropout 率，未丟棄的值將相應地按 1/0.5 = 2 的因數進行縮放
    - 縮放由公式 1 / (1 - `dropout_rate`) 計算
    """)
    return


@app.cell
def _(torch):
    torch.manual_seed(42)
    dropout = torch.nn.Dropout(0.5) # dropout rate of 50%
    example = torch.ones(6, 6) # create a matrix of ones

    print(dropout(example))
    return (dropout,)


@app.cell
def _(attn_weights_3, dropout, torch):
    torch.manual_seed(42)
    print(dropout(attn_weights_3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 請注意，根據您的作業系統，產生的 dropout 輸出可能看起來不同；您可以在 [PyTorch 問題追蹤器上閱讀更多關於這種不一致性的資訊](https://github.com/pytorch/pytorch/issues/121595)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.5.3 實作一個緊湊的因果自注意力類別
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 現在，我們準備實作一個自注意力的工作實作，包括因果和 dropout 遮罩
    - 還有一件事是實作程式碼來處理由多個輸入組成的批次，以便我們的 `CausalAttention` 類別支援我們在第 2 章中實作的資料載入器產生的批次輸出
    - 為了簡單起見，為了模擬這樣的批次輸入，我們複製輸入文本範例：
    """)
    return


@app.cell
def _(inputs, torch):
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape) # 2 inputs with 6 tokens each, and each token has embedding dimension 3
    return (batch,)


@app.cell
def _(batch, d_in, d_out, nn, torch):
    class CausalAttention(nn.Module):

        def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
            super().__init__()
            self.d_out = d_out
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.dropout = nn.Dropout(dropout)
            self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

        def forward(self, x):
            b, num_tokens, d_in = x.shape
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)
            attn_scores = queries @ keys.transpose(1, 2)
            attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)
            context_vec = attn_weights @ values
            return context_vec
    torch.manual_seed(123)
    context_length_1 = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length_1, 0.0)
    _context_vecs = ca(batch)
    print(_context_vecs)
    print('context_vecs.shape:', _context_vecs.shape)
    return (CausalAttention,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 請注意，dropout 僅在訓練期間應用，而不在推理期間應用
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/23.webp" width="500px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.6 將單頭注意力擴展到多頭注意力
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.6.1 堆疊多個單頭注意力層
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 以下是之前實作的自注意力的摘要（為了簡單起見，未顯示因果和 dropout 遮罩）

    - 這也稱為單頭注意力：

    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/24.webp" width="400px">

    - 我們只需堆疊多個單頭注意力模組即可獲得多頭注意力模組：

    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/25.webp" width="400px">

    - 多頭注意力背後的主要思想是使用不同的學習線性投影多次（並行）運行注意力機制。這允許模型在不同位置共同關注來自不同表示子空間的資訊。
    """)
    return


@app.cell
def _(CausalAttention, batch, nn, torch):
    class MultiHeadAttentionWrapper(nn.Module):

        def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
            super().__init__()
            self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])

        def forward(self, x):
            return torch.cat([head(x) for head in self.heads], dim=-1)
    torch.manual_seed(123)
    context_length_2 = batch.shape[1]
    d_in_1, d_out_1 = (3, 2)
    _mha = MultiHeadAttentionWrapper(d_in_1, d_out_1, context_length_2, 0.0, num_heads=2)
    _context_vecs = _mha(batch)
    print(_context_vecs)
    print('context_vecs.shape:', _context_vecs.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 在上述實作中，嵌入維度為 4，因為我們使用 `d_out=2` 作為鍵、查詢和值向量以及上下文向量的嵌入維度。由於我們有 2 個注意力頭，我們的輸出嵌入維度為 2*2=4
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.6.2 使用權重分割實作多頭注意力
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 雖然上述是多頭注意力的直觀且功能齊全的實作（包裝了之前的單頭注意力 `CausalAttention` 實作），但我們可以編寫一個名為 `MultiHeadAttention` 的獨立類別來實現相同的目標

    - 對於這個獨立的 `MultiHeadAttention` 類別，我們不連接單個注意力頭
    - 相反，我們創建單個 W_query、W_key 和 W_value 權重矩陣，然後將這些矩陣分割成每個注意力頭的單獨矩陣：
    """)
    return


@app.cell
def _(batch, nn, torch):
    class MultiHeadAttention(nn.Module):

        def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
            super().__init__()
            assert d_out % num_heads == 0, 'd_out must be divisible by num_heads'
            self.d_out = d_out
            self.num_heads = num_heads
            self.head_dim = d_out // num_heads
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.out_proj = nn.Linear(d_out, d_out)
            self.dropout = nn.Dropout(dropout)
            self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

        def forward(self, x):
            b, num_tokens, d_in = x.shape
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)
            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
            keys = keys.transpose(1, 2)
            queries = queries.transpose(1, 2)
            values = values.transpose(1, 2)
            attn_scores = queries @ keys.transpose(2, 3)
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)
            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)
            context_vec = (attn_weights @ values).transpose(1, 2)
            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
            context_vec = self.out_proj(context_vec)
            return context_vec
    torch.manual_seed(123)
    batch_size, context_length_3, d_in_2 = batch.shape
    d_out_2 = 2
    _mha = MultiHeadAttention(d_in_2, d_out_2, context_length_3, 0.0, num_heads=2)
    _context_vecs = _mha(batch)
    print(_context_vecs)
    print('context_vecs.shape:', _context_vecs.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 請注意，上述本質上是 `MultiHeadAttentionWrapper` 的重寫版本，更有效率
    - 由於隨機權重初始化不同，產生的輸出看起來有點不同，但兩者都是可以在我們將在接下來的章節中實作的 GPT 類別中使用的功能齊全的實作
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    **關於輸出維度的說明**

    - 在上面的 `MultiHeadAttention` 中，我使用 `d_out=2` 來使用與之前的 `MultiHeadAttentionWrapper` 類別相同的設定
    - `MultiHeadAttentionWrapper` 由於連接，返回輸出頭維度 `d_out * num_heads`（即 `2*2 = 4`）
    - 但是，`MultiHeadAttention` 類別（使其更加使用者友好）允許我們直接透過 `d_out` 控制輸出頭維度；這意味著，如果我們設定 `d_out = 2`，輸出頭維度將為 2，無論頭的數量如何
    - 事後看來，正如讀者[指出的](https://github.com/rasbt/LLMs-from-scratch/pull/859)，使用 `d_out = 4` 的 `MultiHeadAttention` 可能更直觀，這樣它產生與使用 `d_out = 2` 的 `MultiHeadAttentionWrapper` 相同的輸出維度。

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 請注意，此外，我們在上面的 `MultiHeadAttention` 類別中添加了一個線性投影層（`self.out_proj`）。這只是一個不改變維度的線性變換。在大型語言模型實作中使用這樣的投影層是標準慣例，但並非嚴格必要（最近的研究表明，可以在不影響建模效能的情況下將其移除；請參閱本章末尾的進一步閱讀部分）
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/26.webp" width="400px">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 請注意，如果您對上述的緊湊和高效實作感興趣，您也可以考慮 PyTorch 中的 [`torch.nn.MultiheadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) 類別
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 由於上述實作乍一看可能有點複雜，讓我們看看執行 `attn_scores = queries @ keys.transpose(2, 3)` 時會發生什麼：
    """)
    return


@app.cell
def _(torch):
    # (b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4)
    a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                        [0.8993, 0.0390, 0.9268, 0.7388],
                        [0.7179, 0.7058, 0.9156, 0.4340]],

                       [[0.0772, 0.3565, 0.1479, 0.5331],
                        [0.4066, 0.2318, 0.4545, 0.9737],
                        [0.4606, 0.5159, 0.4220, 0.5786]]]])

    print(a @ a.transpose(2, 3))
    return (a,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - 在這種情況下，PyTorch 中的矩陣乘法實作將處理 4 維輸入張量，以便在最後 2 個維度（num_tokens，head_dim）之間執行矩陣乘法，然後為各個頭重複

    - 例如，以下成為為每個頭分別計算矩陣乘法的更緊湊方式：
    """)
    return


@app.cell
def _(a):
    first_head = a[0, 0, :, :]
    first_res = first_head @ first_head.T
    print("First head:\n", first_res)

    second_head = a[0, 1, :, :]
    second_res = second_head @ second_head.T
    print("\nSecond head:\n", second_res)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 總結和要點
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 基本原理：注意力機制利用「點積」計算權重，將輸入轉換為包含全局資訊的上下文向量。
    - 核心機制 (Q/K/V)：引入可訓練權重矩陣生成 Query、Key、Value，實現「縮放點積注意力」。
    - 遮罩技術 (Masking)：
      - 因果遮罩：防止偷看未來答案。
      - Dropout 遮罩：防止模型死背（過度擬合）。
    - 多頭注意力 (Multi-head)：透過「批次矩陣乘法」平行處理多個注意力頭，提升模型捕捉不同特徵的能力。
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
