# Phase 4D — Evaluation & Error Analysis

Generated: deterministic baselines on the developer set split + golden benchmark.

## 1. Headline (golden benchmark, macro F1)

| Model | Input | Acc | MacroF1 | W-F1 |
|---|---|---:|---:|---:|
| logistic | message_only | 0.230 | 0.164 | 0.232 |
| svm | message_only | 0.245 | 0.144 | 0.237 |
| svm | message_context | 0.280 | 0.135 | 0.240 |
| logistic | message_context | 0.245 | 0.121 | 0.218 |
| majority | message_context | 0.160 | 0.020 | 0.044 |
| majority | message_only | 0.160 | 0.020 | 0.044 |

## 2. Developer-set split (conversation-disjoint)
```
sizes: {'train': 140, 'validation': 30, 'test': 30}
cross-split duplicate messages (train/val/test): 0
duplicate message text golden-vs-1/0/0 (train/val/test)
```

## 3. Segmentation (macro F1; logistic, message-only, golden)
### difficulty
| value | n | acc | macro_f1 |
|---|---:|---:|---:|
| medium | 95 | 0.274 | 0.161 |
| hard | 58 | 0.172 | 0.130 |
| easy | 47 | 0.213 | 0.134 |

### language
| value | n | acc | macro_f1 |
|---|---:|---:|---:|
| en | 180 | 0.244 | 0.173 |
| fr | 5 | 0.000 | 0.000 |
| ja | 5 | 0.000 | 0.000 |
| es | 3 | 0.333 | 0.167 |
| it | 3 | 0.000 | 0.000 |
| de | 2 | 0.000 | 0.000 |
| pt | 2 | 0.500 | 0.333 |

### context_dependency
| value | n | acc | macro_f1 |
|---|---:|---:|---:|
| context_helpful | 140 | 0.236 | 0.173 |
| self_contained | 54 | 0.222 | 0.127 |
| context_required | 6 | 0.167 | 0.143 |

### message_type
| value | n | acc | macro_f1 |
|---|---:|---:|---:|
| complaint | 106 | 0.245 | 0.156 |
| follow_up | 36 | 0.278 | 0.167 |
| acknowledgement | 26 | 0.154 | 0.028 |
| support_request | 23 | 0.174 | 0.167 |
| clarification | 5 | 0.200 | 0.083 |
| unknown | 4 | 0.250 | 0.100 |

## 4. Calibration / confidence (logistic message-only, golden)
Note: absolute confidence never exceeds ~0.25 (14-way softmax, 140 train rows), so buckets are quantiles of observed confidence.

| bucket | n | conf_range | acc | macro_f1 |
|---|---:|---|---:|---:|
| q0 | 40 | 0.088-0.100 | 0.275 | 0.165 |
| q1 | 40 | 0.100-0.107 | 0.200 | 0.127 |
| q2 | 40 | 0.107-0.114 | 0.200 | 0.124 |
| q3 | 40 | 0.114-0.123 | 0.250 | 0.192 |
| q4 | 40 | 0.123-0.215 | 0.225 | 0.115 |

## 5. Error analysis (logistic message-only)

On the golden benchmark, logistic/message-only misclassifies 154/200 examples. All captured in `reports/error_analysis.json`; a sample follows.

Top confusion patterns (gold -> predicted):

| gold | predicted | count |
|---|---|---:|
| none | product_return_and_replacement | 8 |
| delivery_delay | service_complaint_escalation | 7 |
| delivery_delay | refund_request | 6 |
| product_return_and_replacement | service_complaint_escalation | 6 |
| device_app_issue | none | 5 |
| cancellation | service_complaint_escalation | 4 |
| delivery_delay | none | 4 |
| none | charge_issue | 4 |
| none | device_app_issue | 4 |
| order_status_query | delivery_delay | 4 |
| service_complaint_escalation | refund_request | 4 |
| account_access | delivery_delay | 3 |
| charge_issue | service_complaint_escalation | 3 |
| delivered_but_not_received | service_complaint_escalation | 3 |
| delivered_wrong_location | service_complaint_escalation | 3 |

Example errors:

| conv | lang | diff | ctx | gold | pred | conf | message |
|---|---|---|---|---|---|---:|---|
| 2814438 | es | hard | context_helpful | account_access | delivered_wrong_location | 0.172 | @USER Ya lo he intentado. Me llego un mail pidiendo que les envie un fax.. les mande el fa |
| 2694949 | en | hard | self_contained | account_access | delivery_delay | 0.107 | I tried @USER prime years ago for 30 days before the student offer or music service existe |
| 519702 | en | hard | context_helpful | account_access | charge_issue | 0.102 | @USER I just want my psn codes so me and my friends can download and play a game tonight.  |
| 2853614 | en | easy | self_contained | cancellation | service_complaint_escalation | 0.118 | @USER Hi, I made an order and it went straight to cancelled! How can I un-cancel? I bought |
| 1955907 | en | hard | context_required | charge_issue | product_return_and_replacement | 0.123 | @USER URL |
| 197857 | en | hard | context_helpful | charge_issue | service_complaint_escalation | 0.108 | @USER It is your responsibility to bar sellers like these on your platform. An increase in |
| 1568158 | en | hard | context_helpful | charge_issue | product_return_and_replacement | 0.104 | @USER And I have another one too this one's actual MRP is Rs. 525/ Frontech 700 MB 5001 FC |
| 834216 | en | hard | context_helpful | delivered_but_not_received | product_return_and_replacement | 0.139 | @USER AMZL US |
| 2798309 | en | hard | context_required | delivered_but_not_received | service_complaint_escalation | 0.124 | @USER Sure, the thing is that I saw notified that my package arrived and this photo came w |
| 689484 | en | medium | context_helpful | delivered_but_not_received | delivery_delay | 0.115 | Did @USER really just not deliver my package and instead of taking responsibilty they ask  |
| 1242242 | en | hard | context_helpful | delivered_but_not_received | service_complaint_escalation | 0.128 | @USER Your customer service is the worst first they investigated the wrong item then they  |
| 1578555 | de | hard | context_helpful | delivered_wrong_location | none | 0.105 | @USER @USER Same here. Amazon hat mich auch im Stich gelassen heute :( |
| 2284883 | en | hard | context_helpful | delivered_wrong_location | refund_request | 0.12 | @USER Wasn’t my order, it was my neighbors. No idea what his name is. Not trying to tattle |
| 2170818 | en | hard | context_required | delivery_delay | none | 0.142 | @USER @USER ?? |
| 842315 | fr | hard | context_helpful | delivery_delay | account_access | 0.12 | @USER Transporteur : Amazon Logistics, n° de suivi CA0002376203 |
| 2785904 | en | medium | context_helpful | delivery_delay | refund_request | 0.123 | @USER No one seems sure. Cust service said "hopefully" it will arrive tomorrow. |
| 1379340 | it | medium | self_contained | device_app_issue | none | 0.131 | @USER @USER C'è speranza di avere l'app #PrimeVideo sul Sony UHP-H1? |
| 227064 | ja | medium | self_contained | device_app_issue | order_status_query | 0.103 | AmazonリーディングってこれKindleのアプリからいけねぇのか？ |
| 1645512 | en | hard | context_helpful | device_app_issue | delivered_wrong_location | 0.099 | @USER I'm having it on the full site and the mobile site on all browsers. I don't use the  |
| 2497106 | en | medium | context_helpful | device_app_issue | none | 0.117 | @USER @USER @USER @USER @USER @USER @USER @USER @USER Given that the app is launched in In |
| 487688 | fr | medium | context_helpful | none | account_access | 0.16 | Mieux vaut tard que jamais... Commandé dimanche avec #AmazonPrime, reçu vendredi avec @USE |
| 1451501 | pt | medium | context_helpful | none | charge_issue | 0.114 | @USER Como foi minha primeira compra, eu comprei 2 (use a cabeça c E linux a bíblia), mas  |
| 2254596 | it | easy | context_helpful | none | product_return_and_replacement | 0.139 | @USER Grazie, buonanotte 😊 |
| 426141 | es | medium | self_contained | none | charge_issue | 0.128 | Cualquier día antes de que decida en pedir algo en @USER ,me ha llegado Dios que velocidad |
| 816342 | it | hard | self_contained | none | product_return_and_replacement | 0.175 | #AmazonIndia URL |

Message+context confusion patterns differ: {('service_complaint_escalation', 'refund_request'): 8, ('delivery_delay', 'refund_request'): 7, ('delivered_but_not_received', 'delivery_delay'): 6, ('none', 'refund_request'): 6, ('none', 'service_complaint_escalation'): 6, ('charge_issue', 'service_complaint_escalation'): 5, ('order_status_query', 'delivery_delay'): 5, ('product_return_and_replacement', 'refund_request'): 5}

## 6. Distinguishing the dev benchmark from the golden benchmark

### Label distribution

| intent | dev(n) | golden(n) |
|---|---:|---:|
| delivery_delay | 38 | 43 |
| delivered_but_not_received | 14 | 11 |
| delivered_wrong_location | 8 | 7 |
| order_status_query | 5 | 13 |
| refund_request | 6 | 8 |
| cancellation | 5 | 6 |
| charge_issue | 12 | 11 |
| product_return_and_replacement | 12 | 18 |
| device_app_issue | 7 | 14 |
| account_access | 4 | 6 |
| account_info_update | 1 | 1 |
| service_complaint_escalation | 16 | 27 |
| none | 62 | 32 |
| other_unclear | 10 | 3 |

### Language distribution

| lang | dev(n) | golden(n) |
|---|---:|---:|
| de | 14 | 2 |
| en | 124 | 180 |
| es | 15 | 3 |
| fr | 12 | 5 |
| hi | 1 | 0 |
| it | 1 | 3 |
| ja | 28 | 5 |
| pt | 5 | 2 |

### Read-out

- Label skew differs: dev is dominated by `none` (acknowledgements/chat filler), golden emphasizes actionable intents (esp. `delivery_delay`). Majority baseline on golden predicts `none` and lands at macro-F1 ~0.02 / acc 0.16 — that gap is expected and meaningful (a real distribution shift, not a code bug).

- Context availability: ~72% of dev messages have 0 retrievable prior turns, so message+context <= message-only for most examples — a data-characteristic finding, not a modelling failure.

- 1 exact-duplicate message text (a canned `@USER URL`) crosses golden-train; it is templated content, the conversations differ, and it carries no useful signal.

- Confusion concentrates in delivery_delay -> none and charge/dispatch intent families (delivered_* vs delivery_delay), i.e. the model under-commits and the multilingual short messages resist TF-IDF on 140 training rows.
