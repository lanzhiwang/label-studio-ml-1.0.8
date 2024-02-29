# Text classification with Scikit-Learn
使用 Scikit-Learn 进行文本分类

* https://labelstud.io/tutorials/sklearn-text-classifier

This tutorial explains the basics of using a Machine Learning (ML) backend with Label Studio using a simple text classification model powered by the [scikit-learn](https://scikit-learn.org/stable/) library.
本教程介绍了使用由 scikit-learn 库支持的简单文本分类模型与 Label Studio 结合使用机器学习 (ML) 后端的基础知识。

Follow this tutorial with a text classification project, where the labeling interface uses the `<Choices>` control tag with the `<Text>` object tag. The following is an example label config that you can use:
按照本教程进行一个文本分类项目，其中标签界面使用 `<Choices>` 控制标记和 `<Text>` 对象标记。以下是您可以使用的标签配置示例：

```xml
<View>
  <Text name="news" value="$text"/>
  <Choices name="topic" toName="news">
    <Choice value="Politics"/>
    <Choice value="Technology"/>
    <Choice value="Sport"/>
    <Choice value="Weather"/>
  </Choices>
</View>

政治
技术
运动
天气

```

### Create a model script
创建模型脚本

If you create an ML backend using [Label Studio’s ML SDK](https://labelstud.io/guide/ml_create), make sure your ML backend script does the following:
如果您使用 Label Studio 的 ML SDK 创建 ML 后端，请确保您的 ML 后端脚本执行以下操作：

- Inherit the created model class from `label_studio_ml.LabelStudioMLBase`
  继承 `label_studio_ml.LabelStudioMLBase` 创建的模型类

- Override the 2 methods: 重写2个方法：

  - `predict()`, which takes [input tasks](https://labelstud.io/guide/tasks#Basic-Label-Studio-JSON-format) and outputs [predictions](https://labelstud.io/guide/predictions) in the Label Studio JSON format.
    `predict()` ，它接受输入任务并以 Label Studio JSON 格式输出预测。

  - `fit()`, which receives [annotations](https://labelstud.io/guide/export#Label-Studio-JSON-format-of-annotated-tasks) iterable and returns a dictionary with created links and resources. This dictionary is used later to load models with the `self.train_output` field.
    `fit()` ，它接收可迭代的注释并返回包含创建的链接和资源的字典。该字典稍后用于加载带有 `self.train_output` 字段的模型。

Create a file `model.py` with the following content:
创建包含以下内容的文件 `model.py` ：

```python
import pickle
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from label_studio_ml.model import LabelStudioMLBase


class SimpleTextClassifier(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(SimpleTextClassifier, self).__init__(**kwargs)

        # then collect all keys from config which will be used to extract data from task and to form prediction
        # Parsed label config contains only one output of <Choices> type
        assert len(self.parsed_label_config) == 1
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        assert self.info['type'] == 'Choices'

        # the model has only one textual input
        assert len(self.info['to_name']) == 1
        assert len(self.info['inputs']) == 1
        assert self.info['inputs'][0]['type'] == 'Text'
        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']

        if not self.train_output:
            # If there is no trainings, define cold-started the simple TF-IDF text classifier
            self.reset_model()
            # This is an array of <Choice> labels
            self.labels = self.info['labels']
            # make some dummy initialization
            self.model.fit(X=self.labels, y=list(range(len(self.labels))))
            print('Initialized with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
                from_name=self.from_name, to_name=self.to_name, labels=str(self.labels)
            ))
        else:
            # otherwise load the model from the latest training results
            self.model_file = self.train_output['model_file']
            with open(self.model_file, mode='rb') as f:
                self.model = pickle.load(f)
            # and use the labels from training outputs
            self.labels = self.train_output['labels']
            print('Loaded from train output with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
                from_name=self.from_name, to_name=self.to_name, labels=str(self.labels)
            ))

    def reset_model(self):
        self.model = make_pipeline(TfidfVectorizer(ngram_range=(1, 3)), LogisticRegression(C=10, verbose=True))

    def predict(self, tasks, **kwargs):
        # collect input texts
        input_texts = []
        for task in tasks:
            input_texts.append(task['data'][self.value])

        # get model predictions
        probabilities = self.model.predict_proba(input_texts)
        predicted_label_indices = np.argmax(probabilities, axis=1)
        predicted_scores = probabilities[np.arange(len(predicted_label_indices)), predicted_label_indices]
        predictions = []
        for idx, score in zip(predicted_label_indices, predicted_scores):
            predicted_label = self.labels[idx]
            # prediction result for the single task
            result = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'choices',
                'value': {'choices': [predicted_label]}
            }]

            # expand predictions with their scores for all tasks
            predictions.append({'result': result, 'score': score})

        return predictions

    def fit(self, completions, workdir=None, **kwargs):
        input_texts = []
        output_labels, output_labels_idx = [], []
        label2idx = {l: i for i, l in enumerate(self.labels)}

        for completion in completions:
            # get input text from task data
            print(completion)
            if completion['annotations'][0].get('skipped') or completion['annotations'][0].get('was_cancelled'):
                continue

            input_text = completion['data'][self.value]
            input_texts.append(input_text)

            # get an annotation
            output_label = completion['annotations'][0]['result'][0]['value']['choices'][0]
            output_labels.append(output_label)
            output_label_idx = label2idx[output_label]
            output_labels_idx.append(output_label_idx)

        new_labels = set(output_labels)
        if len(new_labels) != len(self.labels):
            self.labels = list(sorted(new_labels))
            print('Label set has been changed:' + str(self.labels))
            label2idx = {l: i for i, l in enumerate(self.labels)}
            output_labels_idx = [label2idx[label] for label in output_labels]

        # train the model
        self.reset_model()
        self.model.fit(input_texts, output_labels_idx)

        # save output resources
        model_file = os.path.join(workdir, 'model.pkl')
        with open(model_file, mode='wb') as fout:
            pickle.dump(self.model, fout)

        train_output = {
            'labels': self.labels,
            'model_file': model_file
        }
        return train_output
```

### Create ML backend configs & scripts
创建 ML 后端配置和脚本

Label Studio can automatically create all necessary configs and scripts needed to run ML backend from your newly created model.
Label Studio 可以自动创建从新创建的模型运行 ML 后端所需的所有必要配置和脚本。

Call your ML backend `my_backend` and from the command line, initialize the ML backend directory `./my_backend`:
调用您的 ML 后端 `my_backend` 并从命令行初始化 ML 后端目录 `./my_backend` ：

```bash
label-studio-ml init my_backend
```

The last command takes your script `./model.py` and creates an `./my_backend` directory at the same level, copying the configs and scripts needed to launch the ML backend in either development or production modes.
最后一个命令采用脚本 `./model.py` 并在同一级别创建 `./my_backend` 目录，复制在开发或生产模式下启动 ML 后端所需的配置和脚本。

> NOTE 笔记
> You can specify different location for your model script, for example: `label-studio-ml init my_backend --script /path/to/my/script.py`.
> 您可以为模型脚本指定不同的位置，例如： `label-studio-ml init my_backend --script /path/to/my/script.py` 。

### Launch ML backend server
启动 ML 后端服务器

#### Development mode
发展模式

In development mode, training and inference are done in a single process, therefore the server doesn’t respond to incoming prediction requests while the model trains.
在开发模式下，训练和推理是在单个进程中完成的，因此在模型训练时服务器不会响应传入的预测请求。

To launch ML backend server in a Flask development mode, run the following from the command line:
要在 Flask 开发模式下启动 ML 后端服务器，请从命令行运行以下命令：

```bash
label-studio-ml start my_backend
```

The server started on `http://localhost:9090` and outputs logs in console.
服务器于 `http://localhost:9090` 启动并在控制台中输出日志。

#### Production mode
生产模式

Production mode is powered by a Redis server and RQ jobs that take care of background training processes. This means that you can start training your model and continue making requests for predictions from the current model state.
生产模式由 Redis 服务器和负责后台训练过程的 RQ 作业提供支持。这意味着您可以开始训练模型并继续根据当前模型状态发出预测请求。

After the model finishes the training process, the new model version updates automatically.
模型完成训练过程后，新模型版本会自动更新。

For production mode, please make sure you have Docker and docker-compose installed on your system. Then run the following from the command line:
对于生产模式，请确保您的系统上安装了 Docker 和 docker-compose。然后从命令行运行以下命令：

```bash
cd my_backend/
docker-compose up
```

You can explore runtime logs in `my_backend/logs/uwsgi.log` and RQ training logs in `my_backend/logs/rq.log`
您可以在 `my_backend/logs/uwsgi.log` 中探索运行时日志，在 `my_backend/logs/rq.log` 中探索 RQ 训练日志

### Using ML backend with Label Studio
将 ML 后端与 Label Studio 结合使用

Initialize and start a new Label Studio project connecting to the running ML backend:
初始化并启动一个连接到正在运行的 ML 后端的新 Label Studio 项目：

```bash
label-studio start my_project --init --ml-backends http://localhost:9090
```

#### Getting predictions
获得预测

You should see model predictions in a labeling interface. See [Set up machine learning with Label Studio](https://labelstud.io/guide/ml).
您应该在标签界面中看到模型预测。请参阅使用 Label Studio 设置机器学习。

#### Model training 模型训练

Trigger model training manually by pressing the `Start training` button the Machine Learning page of the project settings, or using an API call:
通过按项目设置的机器学习页面的 `Start training` 按钮或使用 API 调用来手动触发模型训练：

```bash
curl -X POST http://localhost:8080/api/models/train
```

