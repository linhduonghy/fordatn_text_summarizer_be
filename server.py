import traceback
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_ngrok import run_with_ngrok
from summarizer import Summarizer

app = Flask(__name__)
CORS(app)
run_with_ngrok(app)

@app.route('/', methods=['GET'])
def index():
  return "Hello From LinhDV!"

@app.route('/summary', methods=['POST'])
def summary():
    req = request.get_json()
    text = ''
    n_summ = None
    if 'text' in req:
        text = str(req['text'])
        text = text.strip()
    if 'n_summ' in req:
        n_summ = int(req['n_summ'])
    if not text:
        return jsonify({'code': 200, 'data': '', 'msg': 'Thành công!'})
    try:
        summarizer = Summarizer(text, n_summ, 'textrank')
        summ_text = summarizer.summarize()
        summ_text = [s.strip() for s in summ_text if s]
        summ_text = [s.capitalize() for s in summ_text]
        summary = ' '.join(summ_text)
        return jsonify({'code': 200, 'data': summary, 'msg': 'Thành công!'})
    except Exception as e:
        print("---------ERROR----------")
        # log error
        with open("/content/log.txt", "a") as log:
            e_type, e_val, e_tb = sys.exc_info()
            traceback.print_exception(e_type, e_val, e_tb, file=log)
        return jsonify({'code': 500, 'data': None, 'msg': 'Có lỗi xảy ra!'})
    
if __name__ == '__main__':
    app.run()
