## Model A vs Model B (Adam)

| Model | Total params | Trainable params | MACs / image | Size (KB, float32) | Size (KB, .keras file) | Epoch time (s) | Test acc | Macro precision | Macro recall |
|---|---|---|---|---|---|---|---|---|---|
| A | 250,886 | 250,182 | 50,733,440 | 980.0 | 3020.6 | 36.4 | 0.8915 | 0.8941 | 0.8919 |
| B | 38,817 | 38,113 | 6,271,360 | 151.6 | 537.2 | 28.0 | 0.8489 | 0.8517 | 0.8503 |

## Optimizer comparison

| Run | Best epoch | Best val loss | Best val acc | Test acc |
|---|---|---|---|---|
| A_adam | 18 | 0.3485 | 0.8813 | 0.8915 |
| B_adam | 18 | 0.4407 | 0.8480 | 0.8489 |
| B_sgd_momentum | 20 | 0.3873 | 0.8632 | 0.8622 |
| B_sgd | 19 | 0.4984 | 0.8260 | 0.8130 |

## Model B vs fine-tuned lightweight SOTA

| Model | Total params | Size (MB) | Epoch time (s) | Test acc | Macro precision | Macro recall |
|---|---|---|---|---|---|---|
| Model B | 38,817 | 0.148 | 28.0 | 0.8489 | 0.8517 | 0.8503 |
| efficientnetb0 | 4,057,257 | 15.48 | 151.4 | 0.8728 | 0.8754 | 0.8743 |
| mobilenetv2 | 2,265,670 | 8.64 | 100.3 | 0.8653 | 0.8704 | 0.8671 |
