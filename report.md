# Goal

We hope to verify that, by incorporating the candidate idioms into the input sentence, we could improve the model's performance (accuracy).

# Approach

We define a sentence to be a segment of natural language with only one idiom to be filled in. This sentence has a corresponding set of candidate idioms. 

In the original method, the model can only see the sentence itself, but cannot see the corresponding set of candidate idioms that constrains the result. we believe this blindness hinders the model from learning what could be filled into the right position of the sentence. And therefore the model might output the ``correct" (making sense in the current context) answer that is out of the constraint, negatively impacting the accuracy. 

We append all the candidate idoms at the beginning of each sentence, seperated by a special [SEP] token. Then, we input the newly created sentence into the model, and use the output of the [SEP] token as the probability score for choosing each candidate idioms.

# Experiment

# Results and Analysis

# Limitations

# Conclusions

# Division of Labor

