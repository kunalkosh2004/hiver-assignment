# Consolidated benchmark (Phase C) — same 200 golden rows every method

| method | intent acc | macro-F1 | esc-rate | decision acc | esc-F1 | false_auto | grounded% | empty | chars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| majority | 0.160 | 0.020 | – | – | – | – | – | – | – |
| logistic | 0.230 | 0.164 | – | – | – | – | – | – | – |
| svm | 0.245 | 0.144 | – | – | – | – | – | – | – |
| svm (msg+context) | 0.280 | 0.135 | – | – | – | – | – | – | – |
| agent (weak-201k intent) | 0.245 | 0.050 | 0.305 | 0.555 | 0.461 | 66 | 95.7 | 0 | 260 |

Policy probe on the decision benchmark (same 200 rows):

| policy | accuracy | escalate-F1 | false_auto | false_escalate | esc-rate |
|---|---:|---:|---:|---:|---:|
| always_auto | 0.480 | 0.000 | 104 | 0 | 0.000 |
| always_escalate | 0.520 | 0.684 | 0 | 96 | 1.000 |
| risk_intent_policy | 0.550 | 0.274 | 87 | 3 | 0.100 |
| risk_true_intent_policy | 0.875 | 0.876 | 16 | 9 | 0.485 |
| agent_floor_policy | 0.555 | 0.461 | 66 | 23 | 0.305 |
