'''流程：1获得向量数据.2构建索引。3进行检索'''

#预定义：3在构建完索引后进行检索的方法
def topk(index, k = 4):
    #k = 4  # topK的K值
    D, I = index.search(xq, k)  # xq为待检索向量，返回的I为每个待检索query最相似TopK的索引list，D为其对应的距离
    print(I[:5])
    print(D[-5:])

'''1创建向量数据阶段'''
import numpy as np
d = 64                                           # 向量维度
nb = 100000                                      # index向量库的数据量
nq = 10000                                       # 待检索query的数目
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.                # index向量库的向量
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.                # 待检索的query向量

'''2faiss检索向量数据阶段'''
import faiss


'''IndexFlatL2 - 最基础的Index检索'''
def IndexFlatL2():
    global xb, xq, d, nq
    print('------------------------IndexFlatL2 - 最基础的Index检索------------------------')
    index = faiss.IndexFlatL2(d)    #L2代表构建的index采用的相似度度量方法为L2范数，即欧氏距离
    print(index.is_trained)         # 输出为True，代表该类index不需要训练，只需要add向量进去即可
    index.add(xb)                   # 将向量库中的向量加入到index中
    print(index.ntotal)             # 输出index中包含的向量总数，为100000
    topk(index, 4)#上面定义的进行检索


'''
Flat ：暴力检索
优点：该方法是Faiss所有index中最准确的，召回率最高的方法，没有之一；
缺点：速度慢，占内存大。
使用情况：向量候选集很少，在50万以内，并且内存不紧张。
注：虽然都是暴力检索，faiss的暴力检索速度比一般程序猿自己写的暴力检索要快上不少，所以并不代表其无用武之地，建议有暴力检索需求的同学还是用下faiss。
'''
def Flat():
    global xb, xq, d, nq
    print('------------------------Flat ：暴力检索------------------------')
    dim, measure = 64, faiss.METRIC_L2
    param = 'Flat'
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)                                   # 输出为True
    index.add(xb)                                      # 向index中添加向量
    topk(index, 4)#上面定义的进行检索

'''
IVFx Flat ：倒排暴力检索
优点：IVF主要利用倒排的思想，在文档检索场景下的倒排技术是指，一个kw后面挂上很多个包含该词的doc，由于kw数量远远小于doc，因此会大大减少了检索的时间。在向量中如何使用倒排呢？可以拿出每个聚类中心下的向量ID，每个中心ID后面挂上一堆非中心向量，每次查询向量的时候找到最近的几个中心ID，分别搜索这几个中心下的非中心向量。通过减小搜索范围，提升搜索效率。
缺点：速度也还不是很快。
使用情况：相比Flat会大大增加检索的速度，建议百万级别向量可以使用。
参数：IVFx中的x是k-means聚类中心的个数

'''
def IVFxFlat():
    global xb, xq, d, nq
    print('------------------------IVFx Flat ：倒排暴力检索------------------------')
    dim, measure = 64, faiss.METRIC_L2
    param = 'IVF100,Flat'  # 代表k-means聚类中心为100,
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)  # 此时输出为False，因为倒排索引需要训练k-means，
    index.train(xb)  # 因此需要先训练index，再add向量
    index.add(xb)
    topk(index, 4)  # 上面定义的进行检索

''' 
PQx ：乘积量化
优点：利用乘积量化的方法，改进了普通检索，将一个向量的维度切成x段，每段分别进行检索，每段向量的检索结果取交集后得出最后的TopK。因此速度很快，而且占用内存较小，召回率也相对较高。
缺点：召回率相较于暴力检索，下降较多。
使用情况：内存及其稀缺，并且需要较快的检索速度，不那么在意召回率
参数：PQx中的x为将向量切分的段数，因此，x需要能被向量维度整除，且x越大，切分越细致，时间复杂度越高
'''
def PQx():
    global xb, xq, d, nq
    print('------------------------PQx ：乘积量化------------------------')
    dim, measure = 64, faiss.METRIC_L2
    param = 'PQ16'
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)  # 此时输出为False，因为倒排索引需要训练k-means，
    index.train(xb)  # 因此需要先训练index，再add向量
    index.add(xb)
    topk(index, 4)  # 上面定义的进行检索

'''
IVFxPQy 倒排乘积量化
优点：工业界大量使用此方法，各项指标都均可以接受，利用乘积量化的方法，改进了IVF的k-means，将一个向量的维度切成x段，每段分别进行k-means再检索。
缺点：集百家之长，自然也集百家之短
使用情况：一般来说，各方面没啥特殊的极端要求的话，最推荐使用该方法！
参数：IVFx，PQy，其中的x和y同上
'''
def IVFxPQy():
    global xb, xq, d, nq
    print('------------------------IVFxPQy 倒排乘积量化------------------------')
    dim, measure = 64, faiss.METRIC_L2
    param = 'IVF100,PQ16'
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)  # 此时输出为False，因为倒排索引需要训练k-means，
    index.train(xb)  # 因此需要先训练index，再add向量 index.add(xb)
    topk(index, 4)  # 上面定义的进行检索

'''
LSH 局部敏感哈希
原理：哈希对大家再熟悉不过，向量也可以采用哈希来加速查找，我们这里说的哈希指的是局部敏感哈希（Locality Sensitive Hashing，LSH），不同于传统哈希尽量不产生碰撞，局部敏感哈希依赖碰撞来查找近邻。高维空间的两点若距离很近，那么设计一种哈希函数对这两点进行哈希计算后分桶，使得他们哈希分桶值有很大的概率是一样的，若两点之间的距离较远，则他们哈希分桶值相同的概率会很小。
优点：训练非常快，支持分批导入，index占内存很小，检索也比较快
缺点：召回率非常拉垮。
使用情况：候选向量库非常大，离线检索，内存资源比较稀缺的情况
'''
def LSH():
    global xb, xq, d, nq
    print('------------------------LSH 局部敏感哈希------------------------')
    dim, measure = 64, faiss.METRIC_L2
    param = 'LSH'
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)  # 此时输出为True
    index.add(xb)
    topk(index, 4)  # 上面定义的进行检索

'''
HNSWx ：层次NSW【图检索】
优点：该方法为基于图检索的改进方法，检索速度极快，10亿级别秒出检索结果，而且召回率几乎可以媲美Flat，最高能达到惊人的97%。检索的时间复杂度为loglogn，几乎可以无视候选向量的量级了。并且支持分批导入，极其适合线上任务，毫秒级别体验。
缺点：构建索引极慢，占用内存极大（是Faiss中最大的，大于原向量占用的内存大小）
参数：HNSWx中的x为构建图时每个点最多连接多少个节点，x越大，构图越复杂，查询越精确，当然构建index时间也就越慢，x取4~64中的任何一个整数。
使用情况：不在乎内存，并且有充裕的时间来构建index
'''
def HNSWx():
    global xb, xq, d, nq
    print('------------------------HNSWx ：层次NSW【图检索】------------------------')
    dim, measure = 64, faiss.METRIC_L2
    param = 'HNSW64'
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)  # 此时输出为True
    index.add(xb)
    topk(index, 4)  # 上面定义的进行检索

IVFxPQy()