# Homemade CLIP on 

---

### Train clip in 3 steps:
1. Pretrain text part on MLM task;
2. Pretrain image part on classification task + KL Divergence to text embedding by text part;
3. Fintune image + text part with contrastive loss.