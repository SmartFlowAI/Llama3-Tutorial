# Llama3 8B

### 10GB 青春版 (QLoRA, 最长序列 512)
```bash
xtuner train llama3_8b_512_qlora_alpaca_e3.py --deepspeed deepspeed_zero1
```

### 24GB 满血版（QLoRA, 最长序列 8192）
```bash
xtuner train llama3_8b_8k_qlora_alpaca_e3.py --deepspeed deepspeed_zero1
```

### 2*A100 Max版（全量微调，最长序列 8192）
```bash
CUDA_VISIBLE_DEVICES=0,1
NPROC_PER_NODE=2 xtuner train llama3_8b_8k_full_alpaca_e3_sp2.py --deepspeed deepspeed_zero3
```

### 8*A100 Pro Max版（全量微调，最长序列 8192）
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8 xtuner train llama3_8b_8k_full_alpaca_e3.py --deepspeed deepspeed_zero3
```


### 8*A100 Ultra 版（全量微调，最长序列 128k）
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8 xtuner train llama3_8b_128k_full_alpaca_e3.py --deepspeed deepspeed_zero3
```


### 全量微调测速

|   Model   | Sequence Length | GPU Number |  ZeRO  | Sequence Parallel | Tokens per Second | TFLOPs |
| :-------: | :-------------: | :--------: | :----: | :---------------: | :---------------: | :----: |
| Llama3 8B |       8k        |     2      | ZeRO-3 |         2         |      1037.0       |  76.8  |
| Llama3 8B |       8k        |     4      | ZeRO-3 |         1         |      2331.3       | 172.6  |
| Llama3 8B |       8k        |     8      | ZeRO-3 |         1         |      2771.2       | 205.1  |

|   Model   | Sequence Length | GPU Number |  ZeRO  | Sequence Parallel | Tokens per Second | TFLOPs |
| :-------: | :-------------: | :--------: | :----: | :---------------: | :---------------: | :----: |
| Llama3 8B |       8k        |     8      | ZeRO-3 |         1         |      2771.2       | 205.1  |
| Llama3 8B |       16k       |     8      | ZeRO-3 |         2         |      2320.7       | 191.7  |
| Llama3 8B |       32k       |     8      | ZeRO-3 |         4         |      1870.2       | 186.6  |
| Llama3 8B |       64k       |     8      | ZeRO-3 |         8         |      1356.4       | 182.0  |
| Llama3 8B |      128k       |     8      | ZeRO-3 |         8         |       875.7       | 177.7  |
