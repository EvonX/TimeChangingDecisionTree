# Time-Changing Decision Tree

## Time changing decision tree
Time changing decision tree is a learning algorithm that can infer policies from training data with policy changes. Examples of such policies and data includes access control policies and access logs. For example, along the time, a website admin may make some website resource, like url `/exam/midterm.pdf`, public or private, the corresponding access logs could be

```
GET /exam/midterm.pdf 200
GET /exam/midterm.pdf 403
```

This makes previous entropy or gini-based decision tree learning algorithms less effective on inferring the change as a single valid policy. For more details, please read our paper [Towards Continuous Access Control Validation and Forensics](https://evonx.github.io/files/pdiff.pdf).

## Install

```
python setup.py install --user
```


## Usage
Please look into example.ipynb or example.py for how to use tcdt.

## Reference
If you would like to use this tool, please cite our paper.

```
@inproceedings{xiang2019towards,
  title={Towards Continuous Access Control Validation and Forensics},
  author={Xiang, Chengcheng and Wu, Yudong and Shen, Bingyu and Shen, Mingyao and Huang, Haochen and Xu, Tianyin and Zhou, Yuanyuan and Moore, Cindy and Jin, Xinxin and Sheng, Tianwei},
  booktitle={Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security},
  pages={113--129},
  year={2019},
  organization={ACM}
}
```
