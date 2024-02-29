

```bash
docker run -ti --rm -p 0.0.0.0:9090:9090 -v /Users/huzhi/work/code/go_code/ai/HumanSignal/label-studio-ml-1.0.8:/label-studio-ml-1.0.8 -w /label-studio-ml-1.0.8 python:3.8-slim bash

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt


docker pull heartexlabs/label-studio:1.9.1

docker run -p 0.0.0.0:8080:8080 -d -v `pwd`/mydata:/label-studio/data heartexlabs/label-studio:1.9.1 label-studio --log-level DEBUG

admin@cpaas.io
1q2w3e4r@

```
