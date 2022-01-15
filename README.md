# Multilingual-Sentiment-Analysis

This was a project on Multilingual Sentiment Analysis of movie reviews. The movie reviews are from the IMDB Movie Review Dataset. The original dataset was translated to Hindi and Bangla using GoogleTranslate API. The results of the model were as follows:

Test score: 34.033170832395555

Test accuracy: 87.46000000000001

Th results were within 8% of the state of the art.

We used 300 dimensional word vectors from Fasttext model, by facebook research.
We also tried a sparse categorical crossentropy loss function and RMSprop optimizer.
The main model was a Bidirectional LSTM.
The following scores were obtained after running on 4-5 epochs on a Mac.
Better results can be obtained after training for more epochs, with more training data or by training more sophisticated models.
