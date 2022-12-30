docker run -d --name serving_base tensorflow/serving
docker cp /absolute/path/to/saved_model/model_name serving_base:/models/model_name
docker commit --change “ENV MODEL_NAME model_name” serving_base devfest-mobilenet-demo
docker run -t -p 8501:8501 devfest-mobilenet-demo