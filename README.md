# Deep Multi-semantic Fuzzy K-Means with Adaptive Weight Adjustment (DFKMP)

(a)–(c) show the segmentation results of DFKM, while (d)–(f) present the results from DFKMP
![20241108224513](https://github.com/user-attachments/assets/a75c3831-b940-4706-8b93-d78f87fb575b)


## Datasets:
dataset ORL in ORL_32.mat.  
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
## The Core Code
Using P-net to get pseudo-category features：[https://github.com/hlf-art/DFKMP/blob/b487156765d08335f6c1ad673ddec76b040c59a2/MultiviewAEFCM_torch55551-ORL.py#L254 ](https://github.com/hlf-art/DFKMP/blob/b487156765d08335f6c1ad673ddec76b040c59a2/MultiviewAEFCM_torch55551-ORL.py#L254  "[title](https://github.com/hlf-art/DFKMP/blob/b487156765d08335f6c1ad673ddec76b040c59a2/MultiviewAEFCM_torch55551-ORL.py#L254 )")

Calculate the similarity between samples：https://github.com/hlf-art/DFKMP/blob/b487156765d08335f6c1ad673ddec76b040c59a2/MultiviewAEFCM_torch55551-ORL.py#L451C1-L452C1 

Update the fuzzy membership matrix： https://github.com/hlf-art/DFKMP/blob/b487156765d08335f6c1ad673ddec76b040c59a2/MultiviewAEFCM_torch55551-ORL.py#L455C13-L455C118 


## Training
Execute the command in the home directory：
```python
source activatie [Environment Name]
python MultiviewAEFCM_torch55551-ORL.py
```

## Testing
Execute the command in the home directory：
```python
python  metric.py
```
## Brief Introduction
```python  
* DFKM.py: the main source code of DMFKM.
* data_loader.py: load data from matlab files (*.mat).
* utils.py: functions used in experiemnts.
* metric.py: codes for evaluation of clustering results.
* functions.py: Calculate the similarity between samples.
* util.py: Calculating pseudo-label loss.
```
Samples to run the code is given as follows:
```python
if __name__ == '__main__':
    import data_loader as loader

    data, labels = loader.load_ORL()

    data = data.T
    a11 = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1e5, 1e8, 1e10, 1e15]
    data_echart = []
    for index, lam3 in enumerate(a11):

        acc_list, nmi_list, purity_list = [], [], []

        for i in range(10):
            lam = 0.001
            lam2 = 50
            fuzziness = 1.14
            lr = 1e-08
            print('lam={} lam2={} fuzziness={}'.format(lam,lam2,fuzziness))
            dfkm = DeepMultiviewFuzzyKMeans(data, labels, [data.shape[0], 256, 128], lam=lam, lam2=lam2,lam3=lam3, fuzziness = fuzziness, batch_size=128, lr=lr,num_views=3)
            acc,nmi,purity = dfkm.run()
            acc_list.append(acc),nmi_list.append(nmi),purity_list.append(purity)
        data_echart.append(np.mean(acc_list))
        print(data_echart)
```
When running different datasets, please adjust the parameters to be consistent with the paper
## Author of Code
If you have issues, please email: hlf4975035@163.com
