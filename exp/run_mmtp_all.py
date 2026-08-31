import subprocess, os
ROOT="/workspace/counter/"
GT=r'model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)'
env=dict(os.environ, CUDA_VISIBLE_DEVICES="6", HF_HOME="/workspace/.cache/huggingface/",
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", LORA_IMAGES="1", LORA_MAX_PIXELS="261120",
    LORA_MAXLEN="1600", LORA_BS="2", LORA_ACC="16", PROMPT_VARIANT="badimg", LORA_TAG="_mmtpall",
    LORA_BASE_MODEL="google/gemma-4-E4B-it", LORA_TARGETS=GT, LORA_SEED="0")
out=open(ROOT+"exp/lora_foldall_mmtpall.out","w")
p=subprocess.Popen([ROOT+".venv/bin/python","-u","lora_train.py","all","2"],cwd=ROOT+"exp",env=env,stdout=out,stderr=subprocess.STDOUT)
print("pid",p.pid); p.wait(); print("код",p.returncode)
