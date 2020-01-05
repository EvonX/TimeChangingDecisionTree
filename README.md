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
