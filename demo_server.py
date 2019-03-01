import argparse
import chardet
import thriftpy
import falcon
import tensorflow as tf
import io
import re
import os
from datasets import audio
from mainstay import Mainstay
from hparams import hparams
from infolog import log
from tacotron.synthesizer import Synthesizer
from wsgiref import simple_server
from pypinyin import pinyin, lazy_pinyin, Style


html_body = '''<html><title>Tacotron-2 Demo</title><meta charset='utf-8'>
<style>
body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
input {font-size: 14px; padding: 8px 12px; outline: none; border: 1px solid #ddd}
input:focus {box-shadow: 0 1px 2px rgba(0,0,0,.15)}
p {padding: 12px}
button {background: #28d; padding: 9px 14px; margin-left: 8px; border: none; outline: none;
	color: #fff; font-size: 14px; border-radius: 4px; cursor: pointer;}
button:hover {box-shadow: 0 1px 2px rgba(0,0,0,.15); opacity: 0.9;}
button:active {background: #29f;}
button[disabled] {opacity: 0.4; cursor: default}
</style>
<body>
<form>
  <input id="text" type="text" size="40" placeholder="请输入文字">
  <button id="button" name="synthesize">合成</button>
</form>
<p id="message"></p>
<audio id="audio" controls autoplay hidden></audio>
<script>
function q(selector) {return document.querySelector(selector)}
q('#text').focus()
q('#button').addEventListener('click', function(e) {
  text = q('#text').value.trim()
  if (text) {
	q('#message').textContent = '合成中...'
	q('#button').disabled = true
	q('#audio').hidden = true
	synthesize(text)
  }
  e.preventDefault()
  return false
})
function synthesize(text) {
  fetch('/synthesize?text=' + encodeURIComponent(text), {cache: 'no-cache'})
	.then(function(res) {
	  if (!res.ok) throw Error(res.statusText)
	  return res.blob()
	}).then(function(blob) {
	  q('#message').textContent = ''
	  q('#button').disabled = false
	  q('#audio').src = URL.createObjectURL(blob)
	  q('#audio').hidden = false
	}).catch(function(err) {
	  q('#message').textContent = '出错: ' + err.message
	  q('#button').disabled = false
	})
}
</script></body></html>
'''

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='pretrained/', help='Path to model checkpoint')
parser.add_argument('--hparams', default='',help='Hyperparameter overrides as a comma-separated list of name=value pairs')
parser.add_argument('--port', default=6006,help='Port of Http service')
parser.add_argument('--host', default="localhost",help='Host of Http service')
parser.add_argument('--name', help='Name of logging directory if the two models were trained together.')
args = parser.parse_args()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
checkpoint = os.path.join('logs-Tacotron', 'taco_' + args.checkpoint)
try:
	checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
	log('loaded model at {}'.format(checkpoint_path))
except:
	raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

synth = Synthesizer()
modified_hp = hparams.parse(args.hparams)
synth.load(checkpoint_path, modified_hp)

class Res:
	def on_get(self,req,res):
		res.body = html_body
		res.content_type = "text/html"

class Syn:
	def on_get(self,req,res):
		if not req.params.get('text'):
			raise falcon.HTTPBadRequest()
		chs = req.params.get('text')
		print(chs.encode("utf-8").decode("utf-8"))
		pys = chs_pinyin(chs)
		out = io.BytesIO()
		evals = synth.eval(pys)
		audio.save_wav(evals, out, hparams)
		res.data = out.getvalue()
		res.content_type = "audio/wav"

def chs_pinyin(text):
	pys = pinyin(text, style=Style.TONE3)
	results = []
	sentence = []
	for i in range(len(pys)):
		if pys[i][0][0] == "，" or pys[i][0][0] == "、" or pys[i][0][0] == '·':
			pys[i][0] = ','
		elif pys[i][0][0] == '。' or pys[i][0][0] == "…":
			pys[i][0] = '.'
		elif pys[i][0][0] == '―' or pys[i][0][0] == "――" or pys[i][0][0] == '—' or pys[i][0][0] == '——':
			pys[i][0] = ','
		elif pys[i][0][0] == "；":
			pys[i][0] = ';'
		elif pys[i][0][0] == "：":
			pys[i][0] = ':'
		elif pys[i][0][0] == "？":
			pys[i][0] = '?'
		elif pys[i][0][0] == "！":
			pys[i][0] = '!'
		elif pys[i][0][0] == "《" or pys[i][0][0] == '》' or pys[i][0][0] == '（' or pys[i][0][0] == '）':
			continue
		elif pys[i][0][0] == '“' or pys[i][0][0] == '”' or pys[i][0][0] == '‘' or pys[i][0][0] == '’' or pys[i][0][0] == '＂':
			continue
		elif pys[i][0][0] == '(' or pys[i][0][0] == ')' or pys[i][0][0] == '"' or pys[i][0][0] == '\'':
			continue
		elif pys[i][0][0] == ' ' or pys[i][0][0] == '/' or pys[i][0][0] == '<' or pys[i][0][0] == '>' or pys[i][0][0] == '「' or pys[i][0][0] == '」':
			continue

		sentence.append(pys[i][0])
		if pys[i][0] in ",.;?!:":
			results.append(' '.join(sentence))
			sentence = []

	if len(sentence) > 0:
		results.append(' '.join(sentence))

	for i, res in enumerate(results):
		print(res)

	return results


api = falcon.API()
api.add_route("/",Res())
api.add_route("/synthesize",Syn())
log("host:{},port:{}".format(args.host,int(args.port)))
simple_server.make_server(args.host,int(args.port),api).serve_forever()
