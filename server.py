# 1

import threading
import time
from queue import Queue, Empty
from flask import Flask, request, make_response, jsonify

import jax
from jax.experimental import maps
import numpy as np
import optax
import transformers

from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer

params = {
  "layers": 28,
  "d_model": 4096,
  "n_heads": 16,
  "n_vocab": 50400,
  "norm": "layernorm",
  "pe": "rotary",
  "pe_rotary_dims": 64,

  "seq": 2048,
  "cores_per_replica": 8,
  "per_replica_batch": 1,
}

per_replica_batch = params["per_replica_batch"]
cores_per_replica = params["cores_per_replica"]
seq = params["seq"]

# 2

print("load 0 1")

params["sampler"] = nucleaus_sample

# here we "remove" the optimizer parameters from the model (as we don't need them for inference)
params["optimizer"] = optax.scale(0)

print("load 0 2")

mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
devices = np.array(jax.devices()).reshape(mesh_shape)

print("load 0 3")

maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')))

print("load 0 4")

tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

# 3

print("load 1")

total_batch = per_replica_batch * jax.device_count() // cores_per_replica

print("load 2")

network = CausalTransformer(params)

print("load 3")

network.state = read_ckpt(network.state, "step_383500/", devices.shape[1])

print("load 4")

network.state = network.move_xmap(network.state, np.zeros(cores_per_replica))

print("load 5")

# inference

def infer(context, top_p=1, temp=0.8, gen_len=512):
    tokens = tokenizer.encode(context)

    provided_ctx = len(tokens)
    pad_amount = seq - provided_ctx

    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
    batched_tokens = np.array([padded_tokens] * total_batch)
    length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

    start = time.time()
    output = network.generate(batched_tokens, length, gen_len, {"top_p": np.ones(total_batch) * top_p, "temp": np.ones(total_batch) * temp})

    samples = []
    decoded_tokens = output[1][0]

    for o in decoded_tokens[:, :, 0]:
      samples.append(f"\033[1m{context}\033[0m{tokenizer.decode(o)}")

    print(f"completion done in {time.time() - start:06}s")
    return samples

print(infer("question: who is the best pony? answer: ")[0])

# server

app = Flask(__name__)

requests_queue = Queue()

"""
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"context":"eleutherai", "top_p": 0.9, "temp": 0.75}' \
  http://localhost:5000/complete
"""

"""
curl --header "Content-Type: application/json" --request POST --data '{"context":"Prompt: What is best way to inspire people to create the metaverse? Answer: ", "top_p": 0.9, "temp": 0.9, "gen_len": 64}' http://localhost:5000/complete
"""

def _build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response


def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route('/complete', methods=['POST', 'OPTIONS'])
def complete():
    if request.method == "OPTIONS":  # CORS preflight
        return _build_cors_prelight_response()
    elif request.method == "POST":  # The actual request following the preflight
        content = request.json

        if requests_queue.qsize() > 100:
            return {"error": "queue full, try again later"}

        response_queue = Queue()

        requests_queue.put(({
                                "context": content["context"],
                                "top_p": float(content["top_p"]),
                                "temp": float(content["temp"]),
                                "gen_len": int(content["gen_len"])
                            }, response_queue))

        return _corsify_actual_response(jsonify({"completion": response_queue.get()}))
    else:
        raise RuntimeError("Weird - don't know how to handle method {}".format(request.method))

threading.Thread(target=app.run, kwargs={"port": 5000, "host": "0.0.0.0"}).start()
while True:
                try:
                    o, q = requests_queue.get(block=False)
										
                    context = o["context"]
                    top_p = o["top_p"]
                    temp = o["temp"]
                    gen_len = o["gen_len"]
                    result = infer(context=context, top_p=top_p, temp=temp, gen_len=gen_len)[0]
                    q.put(result)
                except Empty:
                    time.sleep(0.01)