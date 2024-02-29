import pickle
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from label_studio_ml.model import LabelStudioMLBase


class SimpleTextClassifier(LabelStudioMLBase):

    # model = cls.model_class(label_config=label_config, train_output=train_output, **kwargs),
    # model = cls.model_class(label_config=label_config, train_output=job_result, **kwargs)
    # model = cls.model_class(label_config=label_config, **kwargs)
    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(SimpleTextClassifier, self).__init__(**kwargs)

        print("SimpleTextClassifier self.label_config:", self.label_config)
        print("SimpleTextClassifier self.parsed_label_config:", self.parsed_label_config)
        print("SimpleTextClassifier self.train_output:", self.train_output)
        print("SimpleTextClassifier self.hostname:", self.hostname)
        print("SimpleTextClassifier self.access_token:", self.access_token)
        # SimpleTextClassifier self.label_config:
        # <View>
        # <Text name="news" value="$text"/>
        # <Choices name="topic" toName="news">
        #     <Choice value="Politics"/>
        #     <Choice value="Technology"/>
        #     <Choice value="Sport"/>
        #     <Choice value="Weather"/>
        # </Choices>
        # </View>
        # SimpleTextClassifier self.parsed_label_config:
        # {
        #     'topic': {
        #         'type': 'Choices',
        #         'to_name': ['news'],
        #         'inputs': [{'type': 'Text', 'value': 'text'}],
        #         'labels': ['Politics', 'Technology', 'Sport', 'Weather'],
        #         'labels_attrs': {
        #             'Politics': {'value': 'Politics'},
        #             'Technology': {'value': 'Technology'},
        #             'Sport': {'value': 'Sport'},
        #             'Weather': {'value': 'Weather'}
        #         }
        #     }
        # }
        # SimpleTextClassifier self.train_output: {}
        # SimpleTextClassifier self.hostname: http://localhost:8080
        # SimpleTextClassifier self.access_token: 0327fc45919b33b67e13daf121ded44d896023b1

        # then collect all keys from config which will be used to extract data from task and to form prediction
        # Parsed label config contains only one output of <Choices> type
        assert len(self.parsed_label_config) == 1
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        print("SimpleTextClassifier self.from_name:", self.from_name)
        print("SimpleTextClassifier self.info:", self.info)
        # SimpleTextClassifier self.from_name: topic
        # SimpleTextClassifier self.info:
        # {
        #     'type': 'Choices',
        #     'to_name': ['news'],
        #     'inputs': [{'type': 'Text', 'value': 'text'}],
        #     'labels': ['Politics', 'Technology', 'Sport', 'Weather'],
        #     'labels_attrs': {
        #         'Politics': {'value': 'Politics'},
        #         'Technology': {'value': 'Technology'},
        #         'Sport': {'value': 'Sport'},
        #         'Weather': {'value': 'Weather'}
        #     }
        # }
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
        print("SimpleTextClassifier predict tasks:", tasks)
        print("SimpleTextClassifier predict kwargs:", kwargs)
        # SimpleTextClassifier predict tasks:
        # [
        #     {
        #         'id': 2,
        #         'data': {'text': '02 Technology'},
        #         'meta': {},
        #         'created_at': '2024-02-29T06:35:15.742490Z',
        #         'updated_at': '2024-02-29T06:35:15.742503Z',
        #         'is_labeled': False,
        #         'overlap': 1,
        #         'inner_id': 2,
        #         'total_annotations': 0,
        #         'cancelled_annotations': 0,
        #         'total_predictions': 0,
        #         'comment_count': 0,
        #         'unresolved_comment_count': 0,
        #         'last_comment_updated_at': None,
        #         'project': 1,
        #         'updated_by': None,
        #         'file_upload': 1,
        #         'comment_authors': [],
        #         'annotations': [],
        #         'predictions': []
        #     }
        # ]
        # SimpleTextClassifier predict kwargs: {'login': None, 'password': None, 'context': None}



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

        print("SimpleTextClassifier predict predictions:", predictions)
        # SimpleTextClassifier predict predictions:
        # [
        #     {
        #         'result': [
        #             {
        #                 'from_name': 'topic',
        #                 'to_name': 'news',
        #                 'type': 'choices',
        #                 'value': {'choices': ['Weather']}
        #             }
        #         ],
        #         'score': 0.8091945092003712
        #     }
        # ]


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
