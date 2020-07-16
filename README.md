# Image-Classification-in-PySpark-Deep-Learning
In this we are implementing simple image classification using Spark Deep Learning.We will try to classify images of two persons (Person1 and Person2) using Transfer Learning technique.

Spark Deep Learning supports the following models:
- InceptionV3
- Xception
- ResNet50
- VGG16
- VGG19

Here, I have implemented using InceptionV3 model.You can try with other supported models as well.

# Required python packages to run spark-deep-learning on python:
  - nose
  - pillow
  - keras
  - h5py
  - py4j

# Run pyspark with spark-deep-learning library
./bin/pyspark --packages databricks:spark-deep-learning:0.1.0-spark2.1-s_2.11 --driver-memory 5g
