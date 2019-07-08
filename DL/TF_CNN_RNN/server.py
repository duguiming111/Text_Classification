"""
@author:duguiming
@description:对外提供接口服务
"""
import json
from predict import CNNModel
from flask import Flask, request, jsonify

app = Flask(__name__)
cnn_model = CNNModel()


@app.route('/predict', methods=['POST'])
def predict():
    res = dict()
    if request.method == 'POST':
        data = request.get_data()
        json_data = json.loads(data.decode('utf-8'))
        line = json_data['text']
        pred_label = cnn_model.predict(line)
        res['code'] = 200
        res['msg'] = '成功'
        res['label'] = pred_label
        return jsonify(res)
    else:
        res['code'] = 405
        res['msg'] = "请求方式错误，请换为POST请求"
        return jsonify(res)


if __name__ == "__main__":
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True, threaded=True, host='0.0.0.0', port=9095)
