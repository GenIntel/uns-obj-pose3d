import flask
import requests
import pickle
import io
import torch

bytes_io = io.BytesIO()
#data = {'img': batch.rgb, 'cam_tform4x4_obj': batch.cam_tform4x4_obj}
data = {"foo": torch.zeros(3,), "blub": torch.ones(4,)}
pickle.dump(data, bytes_io, pickle.HIGHEST_PROTOCOL)
bytes_io.seek(0)
resp = requests.post("http://localhost:5000/predict", files={"file": bytes_io})
print(resp.json())
# resp: flask.Response =