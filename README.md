# Deep Multi-semantic Fuzzy K-Means with Adaptive Weight Adjustment (DFKMP)
## datasets:
dataset ORL in ORL_32.mat. You can train and test on the ORL dataset
|Name| Size| Dimensionality|Class|
| --- | --- | --- | --- | 
|ORL| 400 |1024| 40 |
|YALE| 165| 1024| 15 |
|JAFFE |213| 1024| 10|
|UMIST| 575| 1024 |20|
|USPS |2000 |256| 10|
|CIFAR10 |60,000 |300| 10|
## Installation
required packages:
```python  
torch==1.10.1
torchvision==0.11.2
python==3.9.12
numpy==1.19.2
scipy==1.13.1 
```
## Training
Execute the command in the home directory：
```python
source activatie [Environment Name]
python MultiviewAEFCM_torch55551-ORL.py
```
## Brief Introduction
```python  
* DFKM.py: the main source code of DMFKM.
data_loader.py: load data from matlab files (*.mat).
utils.py: functions used in experiemnts.
metric.py: codes for evaluation of clustering results.
```
