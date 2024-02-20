# 人名分类器-rnn-lstm-gru
**关于人名分类问题:**

以一个人名为输入, 使用模型帮助我们判断它最有可能是来自哪一个国家的人名, 这在某些国际化公司的业务中具有重要意义, 在用户注册过程中, 会根据用户填写的名字直接给他分配可能的国家或地区选项, 以及该国家或地区的国旗, 限制手机号码位数等等.


**数据格式说明:**
每一行第一个单词为人名，第二个单词为国家名。中间用制表符tab分割
