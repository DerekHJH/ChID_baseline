# Goal

We hope to verify that, by incorporating the candidate idioms into the input sentence, we could improve the model's performance (accuracy).

# Approach

We define a sentence to be a segment of text with only one idiom to be filled in. This sentence has a corresponding set of candidate idioms. 

In the original method, the model can only see the sentence itself, but cannot see the corresponding set of candidate idioms that constrains the result. we believe this blindness hinders the model from learning what could be filled into the right position of the sentence. And therefore the model might try to output the "correct" (making sense in the current context) answer that is out of the constraint, negatively impacting the accuracy. 

We splice all the candidate idoms at the beginning of each sentence, separated by a special ``[SEP]`` token, in the hope that the masked token will pay more attention to the given candidates when deciding the answer.

# Experiment

First, for each sentence, we splice its candidates at the beginning, separated by a seperation token ``[SEP]``. For example, suppose the sentence is ``随后的一幕，[MASK][MASK][MASK][MASK]，易建联双手将球狠狠地砸进篮筐......`` and the candidates are ``['瓜熟蒂落', '诗情画意', '皆大欢喜', '倚老卖老', '铮铮铁骨', '迎刃而解', '水到渠成']``. After the splice, the sentence becomes ``克勤克俭[SEP]好吃懒做[SEP]鼻青眼肿[SEP]孜孜不倦[SEP]无微不至[SEP]任劳任怨[SEP]引火烧身[SEP]随后的一幕，[MASK][MASK][MASK][MASK]，易建联双手将球狠狠地砸进篮筐......``.

We preprocess the dataset as specified above and keep the model structure and all the hyperparameters unchanged (the same as the baseline). And our experiments are conducted on a machine equipped with one GeForce GTX 3090 GPU, two AMD EPYC 7H12 CPU @ 2.6GHz with 64 core processors, and 512G RAM.

# Results and Analysis

The following is a table showing the dev and test accuracy using our approach. ``CS`` stands for `Candidate Splicing`, in which we splice the candidate idioms at the beginning of each sentence. And ``BS`` stands for ``Baseline``.

We first reproduce the results of the baseline (the first two columns of the table), and find out that the accuracy in our settings are relatively lower than the original baseline settings, but the numbers are very close. This validates that our experiment is set up properly.

Then, we produce the results of our approach (the last two columns of the table), and find out that the model performance is already slightly improved in the zero-shot scenario, by simply splicing the candidate idioms at the beginning of each sentence. In addition, after finetuning the model using different scale of training datasets, we achieve an accuracy improvement by a large margin (the improvement is over 5% using train_data_1w, over 10% using train_data_5w, over 15% using train_data_10w, close to 20% using the full training set). 

| #train data | dev (BS) | test (BS) | dev (CS) | test (CS) |
|-------------|:--------:|:---------:|:--------:|:---------:|
| 0           |   51.55  |   51.87   |   52.78  |   52.93   |
| 1w          |   64.28  |   64.29   |   69.82  |   70.07   |
| 5w          |   71.54  |   71.31   |   83.45  |   83.47   | 
| 10w         |   74.15  |   74.02   |   90.05  |   90.12   |
| full (50w)  |   80.59  |   80.68   |   98.20  |   98.24   |

## Attention Score

We have shown that our approach works and improves the accuracy by a large margin. Next, we show that the model is indeed paying much "attention" to the spliced candidate idioms at the beginning of each sentence.

We use the model trained using train_data_1w, and pick out one example to demonstrate this phenomenon. But this phenomenon is not limited to this specific example.

For simplicity, in this example, we only observe the attention of the first attention head of the last layer of the RoBerta model. And we only focus on the attention of the four ``[MASK]`` tokens. The sentence is ``朱洪达打从懂得恨起就恨爹，一碗白水般的纯洁心里实实地恨爹。伺候他左右的是驴脸长髯#idiom#的彪形莽汉，终日禁锢在高墙深院之中，与世隔绝一般。戴着瓶子底眼镜的先生，#idiom#教他背百家姓、千字文、学算盘，#idiom#，赵钱孙李，归片大扒皮，烦透啦！有时候趁先生不备，他舔破书屋的窗户纸，窥视出出进进大院的人，骑着毛管发亮的[MASK][MASK][MASK][MASK]，耀武扬威，他梦想骑骑马，也挎挎匣子枪，可爹却让他读书……爷爷咽气那天，他被拉出来，整日身披重孝，昼夜守在骇人的棺材旁，听那嚎嚎啕啕，又陪磕头，六天六夜，真够少爷受的。后来他在迷迷糊糊中被装进筐掠上骡子背......``. The companying candidate idioms are ``["器宇轩昂", "前呼后拥", "天作之合", "忍无可忍", "高头大马", "杀气腾腾", "歪打正着"]``. The spliced sentence is ``器宇轩昂[SEP]前呼后拥[SEP]天作之合[SEP]忍无可忍[SEP]高头大马[SEP]杀气腾腾[SEP]歪打正着[SEP]朱洪达打从懂得恨起就恨爹，一碗白水般的纯洁心里实实地恨爹。伺候他左右的是驴脸长髯#idiom#的彪形莽汉，终日禁锢在高墙深院之中，与世隔绝一般。戴着瓶子底眼镜的先生，#idiom#教他背百家姓、千字文、学算盘，#idiom#，赵钱孙李，归片大扒皮，烦透啦！有时候趁先生不备，他舔破书屋的窗户纸，窥视出出进进大院的人，骑着毛管发亮的[MASK][MASK][MASK][MASK]，耀武扬威，他梦想骑骑马，也挎挎匣子枪，可爹却让他读书……爷爷咽气那天，他被拉出来，整日身披重孝，昼夜守在骇人的棺材旁，听那嚎嚎啕啕，又陪磕头，六天六夜，真够少爷受的。后来他在迷迷糊糊中被装进筐掠上骡子背......``. And the right answer (label) is 4 (the forth candidate idiom ``高头大马``). 

Since the attention map is too big to put it in here, for each ``[MASK]``, we select the top-5 attention score as below:

- ``[MASK]1``: 
values=tensor([0.3790, 0.2592, 0.1681, 0.0373, 0.0324], device='cuda:0'),
indices=tensor([ 21高, 196``[MASK]2``,  22头,  23大, 194的], device='cuda:0'))
- ``[MASK]2``:
values=tensor([0.3187, 0.1286, 0.0882, 0.0530, 0.0437], device='cuda:0'),
indices=tensor([ 22头,  23大,  24马, 194的, 195``[MASK]1``], device='cuda:0'))

- ``[MASK]3``:
values=tensor([0.5357, 0.1681, 0.0471, 0.0288, 0.0202], device='cuda:0'),
indices=tensor([ 23大,  24马, 198``[MASK]4``, 194的, 196``[MASK]2``], device='cuda:0'))
- ``[MASK]4``:
values=tensor([0.1909, 0.0675, 0.0570, 0.0434, 0.0426], device='cuda:0'),
indices=tensor([ 24马, 199, 198``[MASK]4``, 196``[MASK]2``,  10``[SEP]``], device='cuda:0'))

candidate_mask index = 195, 196, 197, 198

We can see that, in the last layer where the final prediction is to be made, the attention of all four masks are focused on the correct candidate idiom, verifying that our approach works the way we expected. Besides, the masks also pay attention to words like ``的``, probably to make sure that the masked word are an adjective.

## Training and Testing Dataset Overlap

We see that, using the full training set, the accuracy in the testing set is abnormally high. We think that there might be some overlap between the full training set and the testing set.

To verify our assumption, we try to calculate the overlap percentage between the full training set and the testing set. Each data entry has one sentence and one set of candidate idioms. For two data entries, We calculate the normalized edit distance, normalized Longest Common Sequence of the two sentences. "Normalize" means we divide the edit distance and the LCS by the length of the longer sentence, making sure that these two metrics fall in the range of (0, 1). Then, we choose the larger metric and set the threshold as 0.8. If the larger metric is larger than the threshold, and the candidate idioms of two data entries are set-equivalent, the two data entries are considered equivalent.

Under this setting, we find that the overlap percentage is about (Still calculating, too slow). 

Due to the computation complexity, and another four final projects to be finished before the end of this semester, we decide to leave this part of work to the future. For now, we roughly look through the first 30 data entries in test_data.json and try to find the "match" in the train_data.json. We successfully found 3 data leakage examples, described as the table below.

|                      |   index   |   index   |   index   |
|----------------------|:---------:|:---------:|:---------:|
| test_data entry idx  |    15     |    24     |    27     |
| train_data entry idx |   230016  |  317505   |   289346  |

# Limitations

Our approach might lose precision when the sentence is too long and has to be cut off. When a sentence is too long and has to be cut off, the baseline model can keep more content of the sentence, while our approach can keep less content due to the extra space occupied by the spliced candidate idioms. We add information by splicing the candidate idioms at the beginning of each sentence, but at the same time we lose information by cutting off more content (context) of a long sentence.

In addition, in the original dataset, there might be multiple ``hole``s to fill in idioms in the content field of each data entry. However, same as the baseline, we consider each ``hole`` separately. Because of this, we might lose chances to let each ``hole`` cross reference each other and thus get more information about the context.

# Conclusions

By incorporating the candidate idioms into the input sentence, we could indeed improve the model's performance (accuracy) in the expected way: The model indeed pays more attention to the correct idiom when making predictions.

In addition, we found that there is indeed some overlap between the full training set and the testing set.

# Thoughts

何依波：我负责的主要是数据处理，数据分析这一部分内容。通过本次大作业，我更加深刻的了解到数据处理是机器学习流程里面非常重要也非常困难的一部分。我主要的感想有两点。首先，通过肉眼观察数据，了解数据的基本构造、分布，对于后续模型的设计、实验的设计，以及实验结果的分析，都大有裨益。其次，数据分析其实是一个很困难的问题，特别是大数据分析。比如，在本次试验中，我负责找出训练集和测试集之间的重叠部分。训练数据集有50w条数据，测试数据集有1w条数据，这就意味着至少有50w * 1w = 50亿次比对。而每次比对，都需要计算两条数据里面句子的最长公共子序列、编辑距离等。假设每条数据的句子长度都为100（实际上平均长度可能大于100），且只考虑最长公共子序列的复杂度O(n^2)，那么大概需要50亿 * 100^2 = 50w亿次计算，根本不可能在有限时间内得到结果。如此高的复杂度，即使利用并行的方式处理数据或许也不能够解决上述问题，再加上计算资源受限，最终我们选择暂时停止对训练集和测试集重叠部分的分析，把注意力放在其他分析和优化上。

胡俊豪：我负责的主要是方法的提出和实验。在看完baseline之后，第一个蹦出来的想法就是：“为什么选项没有被模型利用起来呢？”。于是很自然的，我就想到把选项也喂给模型，让模型看到句子的同时，也看到选项，肯定能够提高模型的表现吧？在这样的想法驱使下，我先简单的把所有选项都插在每个句子的最前面，用分隔符隔开，然后的配置都不变，直接跑实验。结果不出意外，效果直接飙升。但是我们也不能做这种纯intuitive的事情，到底为什么模型的效果会变好，是我们下一个需要分析的问题。所以我开始分析attention score，最终发现，模型在作出判断的时候，确实把更多的注意力放在了选项上。通过本次大作业，再次应证了：科学研究确实就是从灵光一现，然后到把灵光实现出来，然后根据实际效果进行进一步分析讨论原因的过程。

华子曰：整个大作业合作过程中，我负责协调、控制，把关实验进度，整合思路，制作PPT等。在最开始，我们三个人是打算一个人做一部分，然后整合起来作为大作业提交。但是做到后来，我们发现，经常三个人之间互相搞不清楚对方在做什么，交流成本巨大。于是我们决定，还是必须得有一个人“领头”，从头到尾的整个逻辑思路，实验流程理清楚。这就好像论文发表一样，不能是第一二三四作者平均发力，必需要第一作者统领整个工作；否则工作容易乱套，没有主心骨，也没有主要推动人，容易坏掉。有一个人统领全局的好处，拿做PPT举例，就避免了三个人一个人做一个板块的内容，但是三部分内容有机整合到一起比较困难。


# Division of Labor

以下按学号顺序排序，不分先后
- 何依波 (2201111635): 探索整理数据集，调查训练集和测试集之间的重叠，整理对比实验数据，撰写结论
- 胡俊豪 (2201111636): 阅读并理解实验框架，修改代码，完成实验，记录实验结果，撰写结论
- 华子曰 (2201111637): 整理实验思路、方法、实验流程、局限性等，最后完成课堂报告PPT
