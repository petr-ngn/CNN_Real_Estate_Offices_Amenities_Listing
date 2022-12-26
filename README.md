# Multi-label classification of real estate offices' amenities using Convolutional Neural Network (CNN)
Within the course _**Trends in business analytics I (4IZ481)**_, we had to introduce a business solution of unstructured data usage with partial implementation in Python. Our team came up with a solution which combines both image processing and natural language processing.

Particularly, based on our research, we introduced a model for object detection which helps to detect and recognize the real estates' amenities and then outputs a list of amenities which given real estate has, based on the provided real-estate pictures. Afterwards, such predicted amenities' lists are used as keywords for text generator of property listing within real-estate advertisement.

Our partial solution regarded the image processing of real-estate amenities using Keras and Tensorflow. We were provided with web-scrapped dataset of offices' pictures and CSV file with the labels. After image processing such as loading the pictures as 3D arrays with further reshaping, normalization and tensor conversion, we trained a custom Convolutional Neural Network, which had been tuned using Bayesian Optimization, for a multi-label classification, particularly for calculation of a probability for each amenity. Such probabilities were assigned to each offices' picture.

Within the evaluation, we average the probabilities to get aggregated probabilities on the office level, which were then used for a classification of the amenities' occurrences. By taking the predicted amenities' occurrences (predicted labels) or the predicted probabilities and the actual amenities occurrences (true labels), we calculate overall metrics F1 score, accuracy, precision, recall or AUC.

Although the model performed well on the test set with respect to the amenities such as elevator or kitchen, regarding the other amenities the model couldn't detect them at all. We recommended to improve image processing, re-definition of the labels which can be observed from pictures or sample only qualitative offices' pictures with high resolution.
