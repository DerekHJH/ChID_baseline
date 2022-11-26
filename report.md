# Goal

We hope to verify that, by incorporating the candidate idioms into the input sentence, we could improve the model's performance (accuracy).

# Approach

We define a sentence to be a segment of natural language with only one idiom to be filled in. This sentence has a corresponding set of candidate idioms. 

In the original method, the model can only see the sentence itself, but cannot see the corresponding set of candidate idioms that constrains the result. we believe this blindness hinders the model from learning what could be filled into the right position of the sentence. And therefore the model might output the "correct" (making sense in the current context) answer that is out of the constraint, negatively impacting the accuracy. 

We splice all the candidate idoms at the beginning of each sentence, seperated by a special ``[SEP]`` token. Then, we input the newly created sentence into the model, and use the output of the ``[SEP]`` token as the probability score for choosing each candidate idioms.

# Experiment

First, for each sentence, we splice its candidates at the beginning, seperated by a seperation token ``[SEP]``. For example, suppose the sentence is ``随后的一幕，[MASK][MASK][MASK][MASK]，易建联双手将球狠狠地砸进篮筐......`` and the candidates are ``['瓜熟蒂落', '诗情画意', '皆大欢喜', '倚老卖老', '铮铮铁骨', '迎刃而解', '水到渠成']``. After the splice, the sentence becomes ``克勤克俭[SEP]好吃懒做[SEP]鼻青眼肿[SEP]孜孜不倦[SEP]无微不至[SEP]任劳任怨[SEP]引火烧身[SEP]随后的一幕，[MASK][MASK][MASK][MASK]，易建联双手将球狠狠地砸进篮筐......``.

In our first experiment, we only preprocess the dataset as specified above and do nothing else. We directly use the baseline model and scripts to train and test the model.

In our second experiment, we change the scripts and use the output of the ``[SEP]`` token for each candidate idiom as its probability score.

We keep the model structure and all the hyperparameters unchanged (the same as the baseline). In addition, due to the lack of computation resources, we only choose the train_data_1w dataset for training. And our experiments are conducted on a machine equipped with one GeForce GTX 3090 GPU, two AMD EPYC 7H12 CPU @ 2.6GHz with 64 core processors, and 512G RAM.

# Results and Analysis
|   method    |  dev  |  test |
|-------------|:-----:|:-----:|
|  baseline   | 64.28 | 64.29 |
|     CS      | 69.82 | 70.07 |
|           |  |  |
|          |  |  |
|   |  |  |


# Limitations

Our approach might lose precision when the sentence is too long and has to be cut off. In the original 

# Conclusions

# Division of Labor

