# 速通机器学习：从浑水摸鱼到游刃有余
## 前置科技
1. 线性代数
    * 不准确的说法：同时计算大量数据（标量->向量），跨维度映射
1. 计算速度
    * 受自然界启发的人工神经网络在起初时的计算量过大，又碰上了金融危机。直到英伟达的GPU解决计算问题，LeNet的全连接神经网络进行手写数字识别（2024年物理诺奖，足以说明其影响力）后才得以发展
        > 人工神经网络证明了人类对复杂系统的理解，我们不需要算出每一个原子的位置也能模拟宇宙

1. **多层感知机**：
    * 对天然神经网络的初步模仿：每一个神经元接受上一层的全部输出，每一个输入都有对应不同的权重，偏移后计算激活函数，决定是否输出，输出多少
        * 偏移：+b
        * 激活函数：
            1. ReLu：max(0,x)
            1. segmoid:$\frac{1}{1+e^{-x}}$
            1. 
        * 归一化：限制数据大小范围，从-∞，+∞映射到0，1间
            > 这使得模型不过于依赖数据精度，减少计算量从而提高训练速度，适应不同平台上的推理

    * 使用矩阵计算简化：每一层输出视作列向量，每一层神经元行向量代表该行代表的神经元的输入权重
1. **卷积神经网络CNN**：
    1. 什么是卷积：
        * 公式：连续：$(f*g)(t)=\int_{-\inf}^{+\inf} f(\tau)h(t-\tau)d\tau$
        * 解释：h(t)称为卷积核，翻转后按$\tau$进行平移，卷积结果是叠加部分随位移的函数
            * 有时称h(t)为过滤器
            * 时域上的卷积 等价于 频域上的点积
    * 擅长场景：拥有时间空间关联的数据，比如图片，声音（频率，时间）等
    * 计算：
        > 严格意义上这并不是矩阵运算，但是展现形式类似
        1. 每一层具有一个矩阵卷积核，存储训练好的矩阵
        1. 输入与卷积核同等大小的矩阵，输出为输入矩阵和卷积矩阵中同位置数据乘积的总和。
        1. 位移一个数据，重复计算，输出也位移   
        * 卷积核能提取出输入特征，相似程度反映在数值上。
        * 池化：最大值池化/平均值池化，对矩阵下采样降低计算量的同时不丢失特征
        * 最后几层使用全连接
    * 问题：梯度消失/爆炸，超深层特征提取失败
        * 解决办法：残差网络ResNet——相比于传统CNN，增加卷积层间的跳跃连接。这使得卷积结果累加在输入上，特征提取存在保底，进而可以建立上千层的卷积而才出现性能倒退

1. 循环神经网络RNN：
    * 每一个神经元接受输入和前一个神经元的输出，将其拼接在一起进行计算：$x_i=[h_{t-1},x_t]$
    * 不同于传统CNN的区别：同一层的神经元可以单向交流,可以不使用多层感知机
    * **长短期记忆神经网络**：  
        * 两个交流向量：$h_t和C_t$
        * 三个矩阵映射: 
            1. 遗忘门：控制本神经元接受上一个神经元的程度：$f(t)=\sigma(W_f\cdot x_i+b_f)$
            1. 记忆门：控制本神经元记忆接受输入的程度：$i(t)=\sigma(W_i\cdot x_i+b_i)$
            1. 输出门：控制本神经元输出的程度：$o(t) = \sigma(W_o\cdot x_i+b_o)$ 
        * 运输：  
            将当前信息映射到修改空间：  
            $\tilde C(t) = \tanh(W_C\cdot x_i+b_C)$  
            更新信息，其中$h_t$可视作输出：  
            $C(t) = f_t*C_{t-1} + i_t*\tilde C(t)$  
            $h_t=o_{t}*\tanh(C_t)$
1. **Transformer**
    > 不要笑挑战:随机一篇人工智能论文,通过5次以内的引用跳转到Attention is All You Need(Transformer原论文)   

    流程如下,由3blue1brown的科普视频总结而来,只是简述推理过程,暂未提及反向传播:  
    1. 输入的文字分解为token,并对token进行编码(词向量嵌入),词向量的维度为12288(4096*3)
    1. 注意力机制,词向量间会互相沟通并调整自己的向量
        1. 词向量会经过两个矩阵,query查询矩阵,key键矩阵
            * 比如某查询矩阵的含义是查询自己前面有没有形容词,那么查询向量与符合要求的形容词的键值向量内积很大
        1. 对应计算点积并归一化,该值会作为该词的值向量的权重成为原词向量的修正
            * 在词向量空间中,一个复合概念可以由几个基础概念向量和修饰词向量相加得到,比如:  
            V[背带裤]+V[中分头]约等于V[蔡徐坤]  
            这成为了大模型存储知识和思维模式的主要形式
            * q,k矩阵12288 * 128的向量,v矩阵12288 * 12288，超大数据量
        * 像这样的q矩阵和k矩阵总共有96个,他们之间并行计算.所以称为多头注意力
    1. 多层感知机,每个词向量会
    1. 重复上述两个步骤若干次,最后的词向量经过反嵌入矩阵后形成对多个词的条件概率
        * 多层多头注意力会将所有细节集中于最后一个词向量

## Papers Summarize：
1. **Open-loop VLM Robot Planning: An Investigation of Fine-tuning and Prompt Engineering Strategies**
    * 西红柿鸡蛋式论文，但对新手来说刚好
        * GPT的出现极大的鼓舞了AI研究，尤其助长了LLM（large language model），顺带VLM（video language model）也沾了些光，将其代入机器人领域后发现**只需要自然语言就能简单控制机械结构或做出简单的任务规划**
            > CNC编程仔的工资已经够低了……,但我还是想把价格打下来（
        * 油然而生的对云端AI的不信任，更希望本地运行
    * 讨论方向：VideioLLaMA在整合过去信息，上下文，常识知识的能力，通过将训练集的视频转为多个问答
        * 结论：调不了——微调效果不明显，提示造成反作用
        * 在本论文中砍掉了音频组件（在完全体的模型中与视频组件同等地位），因为基准测试没有涉及到音频相关，这样也能减少训练内容
            * 视频逐帧切片后编码，结合查询向量与实践步长信息后线性映射为适合语言模型的特征向量
                > 这里面的每一个名词我都需要解释
        * 数据集：bingbingpongpong
        * 微调：
            > 调什么？怎么调？为啥调？我是谁？我在哪？我要干啥？
        * 评估方法：
            1. 基础方法：向VLM提供当前帧，过往关键帧，文本提示——高级任务目标，解释图片，下一步干嘛，然后做选择题
            1. 思维链：在提示的末尾添加了"Think step by step"
                > 9z？像极了我敲打GPT3的时候
            1. 自我反思：这道题做两次，第二次会包含第一次的结果并问"这是对的吗"，再作回答
            1. 自洽性：LLM会对同一个问题可能会做出不同的答案，但是挑回答的最多的那项能大幅提升准确性
        * 结果
            * 动词/名词的成功率与出现频率无关
            * 微调：
                1. 微调数据集：可能会影响其他数据集训练出来的效果，体现在评估与训练内容相似/不同的环境上
                1. 微调视觉编码器：加速适应，但不一定对泛化有用
                1. 参数数量：7b与13b的区别不大
            * 提示方法：
                1. 思维链：基本不影响，某个项目中出现了过拟合指令提示从而影响了指令接受
                1. 自我反省：对模型影响很小，但是会大幅影响一些项目——模型内省就错推错，没有起到预期作用，或者说压根没推理
                1. 自洽：唯一选择自由作答，使用推理链不会给出完整答案，使用Ego训练集的只会生成候选答案
            * 数据集分析：计算成对的欧几里得距离后再计算数据集中的Wasserstein-1 距离（**?**）
                * 视觉：通过预训练的CLIP模型。对训练集中随机视频帧提取视觉嵌入
                * 文字： fasttext-wiki-news-subwords300模型提取词嵌入
2. **Learning Transferable Visual Models From Natural Language Supervision(CLIP)**
    * 动机：预训练在NLP中掀起了一场革命，文本-文本输入输出中表现出极佳的零样本表现
        > 零样本：在数据集外的样本，类似于泛化，但是模型从未见识过  
        
        传统cv仍需监督学习有限分类的资料，对零样本的表现不佳，通过文本的迁移学习可以赋予cv较为优秀的零样本性能  
        先前已有人提出了ConVIRT的模型——用于医疗图像诊断，作者简化了该模型，使用自己爬取的数据集合，提出了CLIP
    * 结构：类似于tf的单头注意力机制，但首先我们需要好好了解一下什么是transfomer：详见上文
        1. 文本编码器：Transformer简化版——减少注意力头，主要在于描述文本十分简略（a xx of xx）
        1. 图像编码器：  
        ResNet修改版：-D改进——抗锯齿，-2改进——全局平均池化->类TF的注意力池化机制  
        Vision Transfomer：尝鲜
        * 计算余弦相似度：即点乘
    * 表现：
        * 测试集为imagenet-a，是imangenet的精选错题集，从天然图片中挑选ai最容易出错的部分
        ```
        ╰─ & C:/DevelopTool/Anacoda/envs/CLIP/python.exe c:/DevelopTool/Anacoda/envs/CLIP/WorkSpace/check-result-csv.py
        WorkSpace/clip ViT-B/16 imagene-a results.csv: Accuracy = 47.89%
        WorkSpace/clip ViT-B/32 imagene-a results.csv: Accuracy = 30.27%
        WorkSpace/clip ViT-L/14 imagene-a results.csv: Accuracy = 67.47%
        WorkSpace/clip ViT-L/14@336px imagene-a results.csv: Accuracy = 74.29%
        WorkSpace/clip RN50x16 imagene-a results.csv: Accuracy = 52.05%
        WorkSpace/clip RN50x64 imagene-a results.csv: Accuracy = 65.87%
        WorkSpace/clip RN101 imagene-a results.csv: Accuracy = 27.83%
        ```
        * 表现完全取决于视觉编码器，这部分的改进已经做得很详细了，但接下来我要讲述一下zero-shot了
    * Zero-shot 能力
        * 定义：输入来自数据集以外，模型根据已有知识理解
        * 一般的视觉识别“知其然但不知所以然”，但是clip引入的语言编码不一样
        * 通过足够大的预训练，文本Transformer模型中**两个概念的组合等价于概念嵌入的向量相加**
            > ![](../image/image-embedding.png)
        * 同样的效果影响到了视觉编码，模型能将视觉和概念一一对应
        * 我们能否找到一些嵌入，他能对应上视觉嵌入中一些**不被人注意到的东西**，比如ai生成，对抗攻击，图像修改痕迹
        > 或许挖掘LLM就够了？我还没探索Glaze等反ai绘图的效果，这些东西仅限于实验室，即使声势浩大也没人动

    * **Zero-shot Capability**  
        - **Definition**: The model can recognize and understand inputs that are not present in the training dataset by leveraging its prior knowledge.  
        - Traditional vision models typically **"know what it is but not why it is"**, whereas CLIP introduces text encoding that enables a more flexible understanding of concepts.  
        - With sufficiently large-scale pretraining, the text Transformer model can **represent a combination of two concepts as the sum of their embeddings**, improving generalization.   
        - This effect also extends to visual encoding, allowing the model to associate visual features with concepts.  
        - **Can we discover specific embeddings that correspond to subtle visual features that might not be easily noticeable to humans?**  
            - For example, AI-generated content, adversarial attacks, or traces of image modifications.  

3. ClipCap: CLIP Prefix for Image Captioning
    * 一个简单的映射网络就能把图片嵌入”翻译“到文本嵌入，使得LLM具有多模态能力
    > 能不能将其他东西映射到文本嵌入？不行，预训练成本过高

4. 《一系列视频-文本联合训练的仿CLIP模型》
    * 训练成本过高，几乎从头训练
    * 想在成熟模型中引入一个时间向量几乎不可能，必须重头构建
    * 能不能小修小补？如果有为什么他们不早弄