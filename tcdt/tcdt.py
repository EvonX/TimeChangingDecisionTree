import pandas as pd
import numpy as np
import collections
from collections import deque
from pandas import Timestamp

def valid_idx(idx, findex, tmp):
    if len(findex) > 100000:
        return tmp[idx]
    else:
        return idx in findex
    
def count_change(larray):
    cnt = 0
    for i in range(len(larray)-1):
        if larray[i]!=larray[i+1]:
            cnt += 1
    return cnt

def count_change2(larray, vindex, findex, rlindex):
    '''
    Count the changes in larray
    
    Parameters
    ----------
    larray: label array
    vindex: index of valid data
    findex: index of data with the feature
    rlindex: index of data with the rare label
    '''
    #print(larray, vindex, findex, rlindex)
    fchange = 0 # change of the data with the feature
    nfchange = 0 # change of the data without the feature

    tmp = np.array([])
    if len(findex) > 100000:
        tmp = np.zeros(len(vindex), dtype=bool)
        if len(findex) >0:
            tmp[findex]=True

    for idx in rlindex:
        if valid_idx(idx, findex, tmp):
            idxidx = np.where(findex==idx)[0]
            if idxidx[0] > 0:
                idx2 = findex[idxidx[0]-1]
                if larray[idx] != larray[idx2]:
                    fchange += 1
            if idxidx[0] < len(findex)-1:
                idx2 = findex[idxidx[0]+1]
                if larray[idx] != larray[idx2]:
                    fchange += 1
        else:
            idx2 = idx-1
            while idx2 >= 0:
                if not valid_idx(idx2, findex, tmp) and vindex[idx2]:
                    if larray[idx] != larray[idx2]:
                        nfchange += 1
                    break
                idx2 -= 1

            idx2 = idx+1
            while idx2 < len(vindex):
                if not valid_idx(idx2, findex, tmp) and vindex[idx2]:
                    if larray[idx] != larray[idx2]:
                        nfchange += 1
                    break
                idx2 += 1
    return fchange, nfchange

def gen_fvindex(lognp, rlindex, trainingFeatures):
    fvindex={}
    for f in [f for fl in trainingFeatures for f in fl]:
        vindex=[]
        for v in np.unique(lognp[rlindex][:,f]):
            vindex.append((v, np.where(lognp[:,f]==v)[0]))
        fvindex[f] = sorted(vindex, key=lambda x:len(x[1]), reverse=True)
    return fvindex

def hier_features(features):
    maxHier = max([len(f) for f in features])
    for i in range(maxHier):
        yield [f[i] for f in features if i < len(f)]

def index_fname(features, fname):
    for i, f in enumerate(features):
        if fname in f:
            return i, f.index(fname)
    return None, None

class Node:
    def __init__(self, fname, fvalueL, fvalueR, left, right, \
                 ipath, depth, data, datamask, availFeatures,\
                 changeCount, fnamevalue, timeCol, labelCol):
        self.fname = fname
        self.fvalueL = fvalueL
        self.fvalueR = fvalueR
        self.left = left
        self.right = right
        self.ipath = ipath
        self.depth = depth
        self.data = data
        self.datamask = datamask
        self.availFeatures = availFeatures
        self.changeCount = changeCount
        self.fnamevalue = fnamevalue
        self.timeCol = timeCol
        self.labelCol = labelCol
    
    def update_timeseries(self, data, datamask):
        tmpdata = data
        if len(datamask) != 0:
            tmp = np.array([idx for idx in range(len(tmpdata)) if datamask[idx]])
            tmpdata = tmpdata[tmp]
        
        timeSeries = [tmpdata[0]]
        lastAdded = True
        for i in range(len(tmpdata)-1):
            row1 = tmpdata[i]
            row2 = tmpdata[i+1]
            if row1[self.labelCol] != row2[self.labelCol]:
                if not lastAdded:
                    timeSeries.append(row1)
                timeSeries.append(row2)
                lastAdded = True
            else:
                lastAdded = False
        if not lastAdded:
            timeSeries.append(tmpdata[-1])
        self.timeSeries = np.array(timeSeries)
                
    def __str__(self):
        if self.left == None:
            assert(self.right == None)
            tstr = ''
            for i in range(0, len(self.timeSeries)-1, 2):
                row1 = self.timeSeries[i]
                row2 = self.timeSeries[i+1]
                tstr += '{} -- {}: {}; '.format(row1[self.timeCol], 
                                            row2[self.timeCol],
                                            row1[self.labelCol])
            return "class: {}".format(tstr)
        else:
            nstr = ''
            for p in self.ipath:
                nstr += ''.join(p)
                nstr += ';'
            return nstr

class NodeSplitter:
    def split_node(self, node, rlindex, fvindex, labelCol, featureCol, timeCol): 
        fname, fvalue, leftcount, rightcount, availFeatures, findex =\
                self.find_bestsplit_hier(node, rlindex, fvindex, labelCol)

        #leaf node
        if fname == '':
            #mask data and set datamask
            if len(node.datamask) != 0:
                tmp = np.array([idx for idx in range(len(node.data)) if node.datamask[idx]])
                node.datamask = np.array([])
                node.data = node.data[tmp]
            node.update_timeseries(node.data, node.datamask)
            return False
        else:
            data = node.data
            node.fname = fname
            node.fvalueL = [fvalue]

            if len(node.datamask) == 0:
                dataleft = data[np.where(data[:,fname]==fvalue)[0]]
                dataright = data[np.where(data[:,fname]!=fvalue)[0]]
                maskleft = np.array([])
                maskright = np.array([])
            else:
                dataleft = data[findex]
                dataright = data
                maskleft = np.array([])
                maskright = node.datamask
                node.datamask = np.array([])
                maskright[findex] = False

            fright = list(availFeatures)
            fleft = list(availFeatures)

            # left node won't use the selected feature again
            i1, i2 = index_fname(fleft, fname)
            fleft[i1] = fleft[i1][i2+1:]

            lpath = node.ipath+[(featureCol[fname], '==', node.fvalueL[0])]
            rpath = node.ipath+[(featureCol[fname], '!=', node.fvalueL[0])]

            newdepth = node.depth+1
            node.left = Node('', [], [], None, None, lpath, newdepth,\
                             dataleft, maskleft, fleft, leftcount, node.fnamevalue, timeCol, labelCol)
            node.right = Node('', [], [], None, None, rpath, newdepth, \
                              dataright, maskright, fright, rightcount, node.fnamevalue, timeCol, labelCol)

            # intermediate node does not have data attached.
            node.data = None
            return True
    
    def find_bestsplit(self, node, availFlatFeatures, rlindex, fvindex, labelCol):    
        minChangeCount = node.changeCount
        minleftcount = 0
        minrightcount = 0
        minChangeFname= ''
        minChangeFvalue=''
        minfindex = np.array([])
        mini = 0

        totalChange = node.changeCount
        if totalChange == 0:
            return minChangeFname, minChangeFvalue, minleftcount, minrightcount, minfindex

        datamask = node.datamask
        if len(datamask) != 0:
            rlindex = np.array([idx for idx in rlindex if datamask[idx]])

        data = node.data
        #print(fvindex)
        for fname in availFlatFeatures:
            
            i = 0
            for fvalue, findex in fvindex[fname]:
                #print(fname, fvalue)
                if (fname, fvalue) in node.fnamevalue:
                    continue
                vcount = len(findex)
                if totalChange-2*vcount >= minChangeCount:
                    break

                changeCount = 0
                #print(datamask)
                if len(datamask) == 0:
                    datal = data[np.where(data[:,fname]==fvalue)[0]]
                    datar = data[np.where(data[:,fname]!=fvalue)[0]]

                    if len(datal) == 0 or len(datar) == 0:
                        continue
                    leftcount = count_change(datal[:,labelCol])
                    changeCount += leftcount
                    rightcount = count_change(datar[:,labelCol])
                    changeCount += rightcount
                else:
                    findex = np.array([idx for idx in findex if datamask[idx]])
                    if len(findex)==0:
                        continue
                    leftcount, rightcount = count_change2(data[:, labelCol], datamask, findex, rlindex)
                    changeCount += leftcount
                    changeCount += rightcount
                    #print(leftcount, rightcount)
                #print(changeCount, totalChange, minChangeCount)
                if changeCount < totalChange:
                    if changeCount < minChangeCount:
                        minleftcount = leftcount
                        minrightcount = rightcount
                        minChangeCount = changeCount
                        minChangeFname = fname
                        minChangeFvalue = fvalue
                        minfindex = findex
                        mini = i
                    i+=1
        node.fnamevalue.add((minChangeFname, minChangeFvalue))
        return minChangeFname, minChangeFvalue, minleftcount, minrightcount, minfindex

    def find_bestsplit_hier(self, node, rlindex, fvindex, labelCol):
        availFeatures = node.availFeatures
        newAvailFeatures = list(availFeatures)

        for availFlatFeatures in hier_features(newAvailFeatures):
            fname, fvalue, leftcount, rightcount, findex \
                    = self.find_bestsplit(node, availFlatFeatures, rlindex, fvindex, labelCol)
            if fname != '':
                i1, i2 = index_fname(newAvailFeatures, fname)
                newAvailFeatures[i1] = newAvailFeatures[i1][i2:]
                return fname, fvalue, leftcount, rightcount, newAvailFeatures, findex

        return '', '',  0, 0, newAvailFeatures, np.array([])

        
class TCDT:
    def __init__(self, ):
        self.root = None
        self.nodecnt = 0
        
    def fit(self, trainData, trainFeatures, labelCol, timeCol):
        """ Train a new tree
        
        Parameters
        ----------
        trainData: a panda Dataframe including a timestamp column, many feature columns and a label column
        trainFeatures: features for training
        labelCol: label column name 
        timeCol: timestamp column name
        """
        columns = list(trainData.columns)
        self.featureNames = columns
        trainData = trainData.to_numpy()

        trainFeatures = [(lambda x: [columns.index(y) for y in x])(features) for features in trainFeatures]
        labelCol = columns.index(labelCol)
        timeCol = columns.index(timeCol)
        
        self.root = None
        self.nodecnt = 0
        self.labelCol=labelCol
        self.timeCol=timeCol
        leastFrequentLabel=collections.Counter(trainData[:,labelCol]).most_common()[-1][0]
        rlindex = np.where(trainData[:,labelCol]==leastFrequentLabel)[0]
        fvindex = gen_fvindex(trainData, rlindex, trainFeatures)
        
        datamask = np.ones(len(trainData), dtype=bool)
        changecount = count_change2(trainData[:, labelCol], datamask, np.array([]), rlindex)[1]
        #print(changecount)
        assert(changecount!=0)
        self.root = Node('', [], [], None, None, [], 0, \
                    trainData, datamask, trainFeatures, changecount, set(), timeCol, labelCol)
        self.nodecnt=1
        
        nodequeue = deque()
        nodequeue.append(self.root)

        while len(nodequeue) != 0:
            node = nodequeue.popleft()
            if NodeSplitter().split_node(node, rlindex, fvindex, labelCol, columns, timeCol):
                nodequeue.append(node.left)
                nodequeue.append(node.right)
                self.nodecnt+=1
                #print(node)
        return self
    
    def get_allleaves(self):
        """
        return all leaves of the tree
        """
        nodequeue = deque()
        nodequeue.append(self.root)
        leaves = []

        while (len(nodequeue) != 0):
            node = nodequeue.popleft()
            if node.left == None:
                assert(node.right==None)
                leaves.append(node)
            else:       
                nodequeue.append(node.left)
                nodequeue.append(node.right)
        return leaves
    
    def get_leaf(self, case):
        """ Return the leaf that a new case goes
        """
        depth = 0
        root=self.root
        while True:
            if root.fname == '':
                return root, depth
            elif case[root.fname] in root.fvalueL:
                root = root.left
                depth += 1
            elif True: #case[root['fname']] in root['fvalueR']:
                root = root.right
                depth += 1
            else:
                return None, depth
    
    def predict(self, case):
        """ Predict the result of a new case
        """
        leaf, depth = self.get_leaf(case)
        if leaf is None or leaf.timeSeries is None:
            return 'unknown'
        else:
            ldata = leaf.timeSeries
            hdata = ldata[ldata[:, self.timeCol]<=case[self.timeCol]]
            if len(hdata) == 0:
                return 'unknown'
            else:
                return hdata[-1][self.labelCol]
    
    def update(self, case):
        """ Update current tree with a new case
        """
        leaf, depth = self.get_leaf(case)
        if leaf is None or leaf.timeSeries is None:
            print('Data to update is not fitted into the tree')
            return False
        else:
            ldata = leaf.timeSeries
            timeCol = leaf.timeCol
            ldata1 = ldata[ldata[:,timeCol]<=case[timeCol]]
            ldata2 = ldata[ldata[:,timeCol]> case[timeCol]]
            ldata = np.concatenate((ldata1, np.array([case]), ldata2), axis=0)
            leaf.update_timeseries(ldata, leaf.datamask)
            return True
            
    def export_text(self, spacing = 2, max_depth = 20):
        """Build a text report showing the rules of the decision tree.
        """
        report = ""

        def print_tree_recurse(node, report):
            indent = ("|" + (" " * spacing)) * (node.depth+1)
            indent = indent[:-spacing] + "-" * spacing + ' '

            if node.depth <= max_depth:
                info_fmt = "\n"
                info_fmt_left = info_fmt
                info_fmt_right = info_fmt

                if node.left != None:
                    assert(node.right != None)
                    report += indent
                    report += "{} == {}".format(self.featureNames[node.fname],
                                                            node.fvalueL[0])
                    report += info_fmt_left
                    report = print_tree_recurse(node.left, report)

                    report += indent
                    report += "{} != {}".format(self.featureNames[node.fname],
                                                            node.fvalueL[0])
                    report += info_fmt_right
                    report = print_tree_recurse(node.right, report)
                else:
                    report += indent
                    report += node.__str__()
                    report += '\n'
            return report
                
        print_tree_recurse(self.root, report)
        return report