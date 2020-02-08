# story-endding-prediction
A simple story endding prediction model with bert

concerning the Story Cloze Test

## description of the model

The model is simply consists of a Bert layer, a linear layer and a softmax layer to output the classification result.

The Bert layer is a pretrained Bert model, large and cased version.

Mean pooling is done on the output of the Bert layer.

## training

Dataset: Story Cloze Test Dataset

The training data has been omitted, and the validation data has been used as training data.
