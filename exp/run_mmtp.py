"""Управляющий раннер: 4 фолда прицельного промпта на картах 3,4,6,7.
Держит их как дочерние процессы ОДНОГО процесса — переживает то, что убивало
отвязанные nohup-процессы (на сервере /proc не смонтирован, отвязанные сносятся)."""
import subprocess, os, sys, time
ROOT="/workspace/counter/"
GT=r'model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'
jobs=[("3","0"),("4","1"),("6","3"),("7","4")]
procs=[]
for card,fold in jobs:
    env=dict(os.environ, CUDA_VISIBLE_DEVICES=card, HF_HOME=ROOT+".cache_hf" if False else "/workspace/.cache/huggingface/",
             PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
             LORA_IMAGES="1", LORA_MAX_PIXELS="261120", LORA_MAXLEN="1600",
             LORA_BS="2", LORA_ACC="16", PROMPT_VARIANT="badimg", LORA_TAG="_mmtp",
             LORA_BASE_MODEL="google/gemma-4-E4B-it", LORA_TARGETS=GT, LORA_SEED="0")
    out=open(ROOT+f"exp/lora_fold{fold}_mmtp.out","w")
    p=subprocess.Popen([ROOT+".venv/bin/python","-u","lora_train.py",fold,"2"],
                       cwd=ROOT+"exp", env=env, stdout=out, stderr=subprocess.STDOUT)
    procs.append((card,fold,p))
    print(f"запущен фолд {fold} на карте {card}, pid {p.pid}", flush=True)
    time.sleep(20)
print("все запущены, жду завершения...", flush=True)
for card,fold,p in procs:
    p.wait()
    print(f"фолд {fold} (карта {card}) завершён, код {p.returncode}", flush=True)
print("ГОТОВО", flush=True)
