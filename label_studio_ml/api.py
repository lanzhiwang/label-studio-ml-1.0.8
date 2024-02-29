import logging
import os

from flask import Flask, request, jsonify
from rq.exceptions import NoSuchJobError

from .model import LabelStudioMLManager, LABEL_STUDIO_ML_BACKEND_V2_DEFAULT
from .exceptions import exception_handler

logger = logging.getLogger(__name__)

_server = Flask(__name__)
_manager = LabelStudioMLManager()


def init_app(model_class, **kwargs):
    print("api.py model_class:", model_class)
    # api.py model_class: <class 'model.SimpleTextClassifier'>

    print("api.py kwargs:", kwargs)
    # api.py kwargs: {
    #     'model_dir': './my_backend',
    #     'redis_queue': 'default',
    #     'redis_host': 'localhost',
    #     'redis_port': 6379
    # }

    global _manager
    _manager.initialize(model_class, **kwargs)
    return _server


@_server.route('/predict', methods=['POST'])
@exception_handler
def _predict():
    print("***************************api.py request /predict")
    data = request.json
    print("***************************api.py data:", data)
    # data:
    # {
    #     'tasks': [
    #         {
    #             'id': 1,
    #             'data': {'text': '01 Politics'},
    #             'meta': {},
    #             'created_at': '2024-02-29T06:35:15.742406Z',
    #             'updated_at': '2024-02-29T06:35:15.742437Z',
    #             'is_labeled': False,
    #             'overlap': 1,
    #             'inner_id': 1,
    #             'total_annotations': 0,
    #             'cancelled_annotations': 0,
    #             'total_predictions': 0,
    #             'comment_count': 0,
    #             'unresolved_comment_count': 0,
    #             'last_comment_updated_at': None,
    #             'project': 1,
    #             'updated_by': None,
    #             'file_upload': 1,
    #             'comment_authors': [],
    #             'annotations': [],
    #             'predictions': []
    #         }
    #     ],
    #     'model_version': '',
    #     'project': '1.1709187777',
    #     'label_config': '<View>\n  <Text name="news" value="$text"/>\n  <Choices name="topic" toName="news">\n    <Choice value="Politics"/>\n    <Choice value="Technology"/>\n    <Choice value="Sport"/>\n    <Choice value="Weather"/>\n  </Choices>\n</View>',
    #     'params': {
    #         'login': None,
    #         'password': None,
    #         'context': None
    #     }
    # }

    tasks = data.get('tasks')
    project = data.get('project')
    label_config = data.get('label_config')
    force_reload = data.get('force_reload', False)
    try_fetch = data.get('try_fetch', True)
    params = data.get('params') or {}
    predictions, model = _manager.predict(tasks, project, label_config, force_reload, try_fetch, **params)
    response = {
        'results': predictions,
        'model_version': model.model_version
    }
    return jsonify(response)


@_server.route('/setup', methods=['POST'])
@exception_handler
def _setup():
    print("***************************api.py request /setup")
    data = request.json
    print("***************************api.py data:", data)
    # data:
    # {
    #     'project': '1.1709187777',
    #     'schema': '<View>\n  <Text name="news" value="$text"/>\n  <Choices name="topic" toName="news">\n    <Choice value="Politics"/>\n    <Choice value="Technology"/>\n    <Choice value="Sport"/>\n    <Choice value="Weather"/>\n  </Choices>\n</View>'
    #     'hostname': 'http://localhost:8080',
    #     'access_token': '0327fc45919b33b67e13daf121ded44d896023b1',
    #     'model_version': None
    # }
    # {
    #     'project': '1.1709187777',
    #     'schema': '<View>\n  <Text name="news" value="$text"/>\n  <Choices name="topic" toName="news">\n    <Choice value="Politics"/>\n    <Choice value="Technology"/>\n    <Choice value="Sport"/>\n    <Choice value="Weather"/>\n  </Choices>\n</View>',
    #     'hostname': 'http://localhost:8080',
    #     'access_token': '0327fc45919b33b67e13daf121ded44d896023b1',
    #     'model_version': ''
    # }

    logger.debug(data)
    project = data.get('project')
    schema = data.get('schema')
    force_reload = data.get('force_reload', False)
    hostname = data.get('hostname', '')  # host name for uploaded files and building urls
    access_token = data.get('access_token', '')  # user access token to retrieve data
    model = _manager.fetch(project, schema, force_reload, hostname=hostname, access_token=access_token)
    logger.debug('Fetch model version: {}'.format(model.model_version))
    return jsonify({'model_version': model.model_version})


@_server.route('/train', methods=['POST'])
@exception_handler
def _train():
    print("***************************api.py request /train")
    logger.warning("=> Warning: API /train is deprecated since Label Studio 1.4.1. "
                   "ML backend used API /train for training previously, "
                   "but since 1.4.1 Label Studio backend and ML backend use /webhook for the training run.")
    data = request.json
    annotations = data.get('annotations', 'No annotations provided')
    project = data.get('project')
    label_config = data.get('label_config')
    params = data.get('params', {})
    if isinstance(project, dict):
        project = ""
    if len(annotations) == 0:
        return jsonify('No annotations found.'), 400
    job = _manager.train(annotations, project, label_config, **params)
    response = {'job': job.id} if job else {}
    return jsonify(response), 201


@_server.route('/webhook', methods=['POST'])
def webhook():
    print("***************************api.py request /webhook")
    data = request.json
    event = data.pop('action')
    run = _manager.webhook(event, data)
    return jsonify(run), 201


@_server.route('/is_training', methods=['GET'])
@exception_handler
def _is_training():
    print("***************************api.py request /is_training")
    project = request.args.get('project')
    output = _manager.is_training(project)
    return jsonify(output)


@_server.route('/health', methods=['GET'])
@_server.route('/', methods=['GET'])
@exception_handler
def health():
    print("***************************api.py request /health")
    print({
        'status': 'UP',
        'model_dir': _manager.model_dir,
        'v2': os.getenv('LABEL_STUDIO_ML_BACKEND_V2', default=LABEL_STUDIO_ML_BACKEND_V2_DEFAULT)
    })
    # {'status': 'UP', 'model_dir': './my_backend', 'v2': False}
    return jsonify({
        'status': 'UP',
        'model_dir': _manager.model_dir,
        'v2': os.getenv('LABEL_STUDIO_ML_BACKEND_V2', default=LABEL_STUDIO_ML_BACKEND_V2_DEFAULT)
    })


@_server.route('/metrics', methods=['GET'])
@exception_handler
def metrics():
    print("***************************api.py request /metrics")
    return jsonify({})


@_server.errorhandler(NoSuchJobError)
def no_such_job_error_handler(error):
    logger.warning('Got error: ' + str(error))
    return str(error), 410


@_server.errorhandler(FileNotFoundError)
def file_not_found_error_handler(error):
    logger.warning('Got error: ' + str(error))
    return str(error), 404


@_server.errorhandler(AssertionError)
def assertion_error(error):
    logger.error(str(error), exc_info=True)
    return str(error), 500


@_server.errorhandler(IndexError)
def index_error(error):
    logger.error(str(error), exc_info=True)
    return str(error), 500


@_server.before_request
def log_request_info():
    logger.debug('Request headers: %s', request.headers)
    logger.debug('Request body: %s', request.get_data())


@_server.after_request
def log_response_info(response):
    logger.debug('Response status: %s', response.status)
    logger.debug('Response headers: %s', response.headers)
    logger.debug('Response body: %s', response.get_data())
    return response
