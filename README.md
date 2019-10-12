# micronet_challenge_2019
Submission for the [2019 Micronet Challenge](https://micronet-challenge.github.io).

### Solution Outline
Took a pretty simple approach. Used a [NasNet-A](https://arxiv.org/pdf/1707.07012.pdf) architecture and trained using gradient clipping, cutout, and learning rate cosine annealing. No quantization was used.

### Results
Accuracy on Cifar100 Test Set was 81.2800%. Total number of parameters was ~5.2M and parameter score after freebie quantization and 0.07153. Total mults was ~5.5B and total adds was ~5.5B and opearations score after freebie quantization for mults was 0.7809. Total score was **0.85245598**

### Verify Results

Step 1: Create and activate virtual env for this project.

```
virtualenv -p `which python3` micronet_env
source anomaly_env/bin/activate
```

Step 2: Download all dependencies for project.

```
pip install -r requirements.txt
python -m ipykernel install --user --name=micronet_env
```

Step 3: Run jupyter notebook, and open evaluate-nasnet.ipynb. Switch to the micronet_env kernel.

### Train Model

Following Step 1 and Step 2 in Verify Results.

Step 3: Run jupyter notebook, and open train-nasnet.ipynb. Switch to the micronet_env kernel.

### Thank Yous
Thank you to the organizers of this competition; I felt like I learned a lot counting mults and adds. Also, a huge thank you to `weiaicunzai`. I adopted much of his [code](https://github.com/weiaicunzai/pytorch-cifar100) for this submission.
