# Goal

We hope to verify that, by incorporating the candidate idioms into the input sentence, we could improve the model's performance (accuracy).

# Approach

We define a sentence to be a segment of natural language with only one idiom to be filled in. This sentence has a corresponding set of candidate idioms. 

In the original method, the model can only see the sentence itself, but cannot see the corresponding set of candidate idioms that constrains the result. we believe this blindness hinders the model from learning what could be filled into the right position of the sentence. And therefore the model might output the "correct" (making sense in the current context) answer that is out of the constraint, negatively impacting the accuracy. 

We splice all the candidate idoms at the beginning of each sentence, seperated by a special ``[SEP]`` token, in the hope that the masked token will pay more attention to the given candidates when deciding the answer.

# Experiment

First, for each sentence, we splice its candidates at the beginning, seperated by a seperation token ``[SEP]``. For example, suppose the sentence is ``随后的一幕，[MASK][MASK][MASK][MASK]，易建联双手将球狠狠地砸进篮筐......`` and the candidates are ``['瓜熟蒂落', '诗情画意', '皆大欢喜', '倚老卖老', '铮铮铁骨', '迎刃而解', '水到渠成']``. After the splice, the sentence becomes ``克勤克俭[SEP]好吃懒做[SEP]鼻青眼肿[SEP]孜孜不倦[SEP]无微不至[SEP]任劳任怨[SEP]引火烧身[SEP]随后的一幕，[MASK][MASK][MASK][MASK]，易建联双手将球狠狠地砸进篮筐......``.

We preprocess the dataset as specified above and keep the model structure and all the hyperparameters unchanged (the same as the baseline). In addition, due to the lack of computation resources, we only use train_data_1w for training. And our experiments are conducted on a machine equipped with one GeForce GTX 3090 GPU, two AMD EPYC 7H12 CPU @ 2.6GHz with 64 core processors, and 512G RAM.

# Results and Analysis

The following is a table showing the dev and test accuracy using our approach. ``CS`` stands for `Candidate Splicing`, in which we splice the candidate idioms at the beginning of each sentence. And ``BS`` stands for ``Baseline``.

| #train data | dev (BS) | test (BS) | dev (CS) | test (CS) |
|-------------|:--------:|:---------:|:--------:|:---------:|
| 0           |   51.55  |   51.87   |   52.78  |   52.93   |
| 1w          |   64.28  |   64.29   |   69.82  |   70.07   |
| 5w          |          |           |   83.45  |   83.47   | 
| 10w         |          |           |   90.05  |   90.12   |
| full (50w)  |          |           |          |           |

As we can see from the table, the model performance is already slightly improved in the zero-shot scenario, by simply splicing the candidate idioms at the beginning of each sentence. After training using the train_data_1w dataset, we achieve an accuracy improvement more then 5%.

# Limitations

Our approach might lose precision when the sentence is too long and has to be cut off. When a sentence is too long and has to be cut off, the baseline model can keep more content of the sentence, while our approach can keep less content due to the extra space occupied by the spliced candidate idioms. We add information by splicing the candidate idioms at the beginning of each sentence, but at the same time we lose information by cutting off more content of a long sentence.

In addition, in the original dataset, there might be multiple ``hole``s to fill in idioms in the content field of each data entry. However, same as the baseline, we consider each ``hole`` separately. Because of this, we might lose chances to let each ``hole`` cross reference each other and thus get more information about the context.

# Conclusions

# Division of Labor

