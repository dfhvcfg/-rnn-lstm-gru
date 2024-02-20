# 人名分类器-rnn-lstm-gru
**关于人名分类问题:**

以一个人名为输入, 使用模型帮助我们判断它最有可能是来自哪一个国家的人名, 这在某些国际化公司的业务中具有重要意义, 在用户注册过程中, 会根据用户填写的名字直接给他分配可能的国家或地区选项, 以及该国家或地区的国旗, 限制手机号码位数等等.


**数据格式说明:**
每一行第一个单词为人名，第二个单词为国家名。中间用制表符tab分割
# 训练日志

## RNN模型训练日志输出

| 轮次 | 损失    | 时间 | 准确率 |
| ---- | ------- | ---- | ------ |
| 3    | 1.002102 | 54   | 0.700  |
| 3    | 0.993880 | 56   | 0.703  |
| 3    | 0.986200 | 58   | 0.705  |
| 3    | 0.981136 | 61   | 0.706  |
| 3    | 0.976931 | 63   | 0.707  |
| 3    | 0.972190 | 65   | 0.708  |
| 3    | 0.967081 | 68   | 0.710  |
| 3    | 0.964384 | 70   | 0.711  |
| 4    | 0.958782 | 72   | 0.713  |
| 4    | 0.955343 | 75   | 0.713  |
| 4    | 0.950741 | 77   | 0.715  |
| 4    | 0.945756 | 80   | 0.716  |
| 4    | 0.942663 | 82   | 0.717  |
| 4    | 0.939319 | 84   | 0.718  |
| 4    | 0.936169 | 87   | 0.719  |
| 4    | 0.933440 | 89   | 0.720  |
| 4    | 0.930918 | 91   | 0.720  |
| 4    | 0.927330 | 94   | 0.721  |

## LSTM模型训练日志输出

| 轮次 | 损失    | 时间 | 准确率 |
| ---- | ------- | ---- | ------ |
| 3    | 0.805885 | 118  | 0.759  |
| 3    | 0.794148 | 123  | 0.762  |
| 3    | 0.783356 | 128  | 0.765  |
| 3    | 0.774931 | 133  | 0.767  |
| 3    | 0.765427 | 137  | 0.769  |
| 3    | 0.757254 | 142  | 0.771  |
| 3    | 0.750375 | 147  | 0.773  |
| 3    | 0.743092 | 152  | 0.775  |
| 4    | 0.732983 | 157  | 0.778  |
| 4    | 0.723816 | 162  | 0.780  |
| 4    | 0.716507 | 167  | 0.782  |
| 4    | 0.708377 | 172  | 0.785  |
| 4    | 0.700820 | 177  | 0.787  |
| 4    | 0.694714 | 182  | 0.788  |
| 4    | 0.688386 | 187  | 0.790  |
| 4    | 0.683056 | 191  | 0.791  |
| 4    | 0.677051 | 196  | 0.793  |
| 4    | 0.671668 | 201  | 0.794  |

## GRU模型训练日志输出

| 轮次 | 损失    | 时间 | 准确率 |
| ---- | ------- | ---- | ------ |
| 3    | 0.743891 | 106  | 0.772  |
| 3    | 0.733144 | 111  | 0.775  |
| 3    | 0.723484 | 116  | 0.777  |
| 3    | 0.714760 | 120  | 0.780  |
| 3    | 0.706929 | 125  | 0.782  |
| 3    | 0.698657 | 130  | 0.784  |
| 3    | 0.690443 | 134  | 0.787  |
| 3    | 0.683878 | 139  | 0.789  |
| 4    | 0.674766 | 144  | 0.791  |
| 4    | 0.665543 | 148  | 0.794  |
| 4    | 0.657179 | 153  | 0.796  |
| 4    | 0.650314 | 157  | 0.798  |
| 4    | 0.643698 | 162  | 0.800  |
| 4    | 0.637341 | 167  | 0.802  |
| 4    | 0.632063 | 171  | 0.803  |
| 4    | 0.626060 | 176  | 0.805  |
| 4    | 0.621460 | 180  | 0.806  |
| 4    | 0.616704 | 185  | 0.808  |


## 5 模型训练结果分析

### 1 损失对比曲线分析

![1个轮次损失对比曲线](link_to_image1) ![4个轮次损失对比曲线](link_to_image2)

模型训练的损失降低快慢代表模型收敛程度。由图可知，传统RNN的模型第一个轮次开始收敛情况最好，然后是GRU，最后是LSTM，这是因为RNN模型简单参数少，见效快。随着训练数据的增加，GRU效果最好、LSTM效果次之、RNN效果排最后。

所以在以后的模型选用时，要通过对任务的分析以及实验对比，选择最适合的模型。

### 2 训练耗时分析

![训练耗时对比图](link_to_image3)

模型训练的耗时长短代表模型的计算复杂度，由图可知，也正如我们之前的理论分析，传统RNN复杂度最低，耗时几乎只是后两者的一半，然后是GRU，最后是复杂度最高的LSTM。

### 3 训练准确率分析

![训练准确率对比图](link_to_image4)

由图可知，GRU效果最好、LSTM效果次之、RNN效果排最后。

### 4 结论

模型选用一般应通过实验对比，并非越复杂或越先进的模型表现越好，而是需要结合自己的特定任务，从对数据的分析和实验结果中获得最佳答案。

