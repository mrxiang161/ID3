
from math import log2
import treePlotter
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# treePlotter2.createPlot(myTree)
'''
    ID3算法如何选择最优的特征进行划分：
   1.计算整体样本的信息熵
   2.通过特征索引，特征取值对样本集进行划分
   3.计算信息增益并获得增益最大的特征索引

'''

def createDataSet():
	# dataSet = [['青年', '否', '否', '一般', '拒绝'],
	# 			['青年', '否', '否', '好', '拒绝'],
	# 			['青年', '是', '否', '好', '同意'],
	# 			['青年', '是', '是', '一般', '同意'],
	# 			['青年', '否', '否', '一般', '拒绝'],
	# 			['中年', '否', '否', '一般', '拒绝'],
	# 			['中年', '否', '否', '好', '拒绝'],
	# 			['中年', '是', '是', '好', '同意'],
	# 			['中年', '否', '是', '非常好', '同意'],
	# 			['中年', '否', '是', '非常好', '同意'],
	# 			['老年', '否', '是', '非常好', '同意'],
	# 			['老年', '否', '是', '好', '同意'],
	# 			['老年', '是', '否', '好', '同意'],
	# 			['老年', '是', '否', '非常好', '同意'],
	# 			['老年', '否', '否', '一般', '拒绝'],
	# 			]
	# feature_index = ['年龄', '有工作', '有房子', '信贷情况']
	feature_index=[0,1,2,3]
		# ##添加了工资列
	dataSet = [['1000','青年', '否', '否', '一般', '拒绝'],
                ['2000','青年', '否', '否', '好', '拒绝'],
                ['7000','青年', '是', '否', '好', '同意'],
                ['7100','青年', '是', '是', '一般', '同意'],
                ['3000','青年', '否', '否', '一般', '拒绝'],
                ['3500','中年', '否', '否', '一般', '拒绝'],
                ['3600','中年', '否', '否', '好', '拒绝'],
                ['8000','中年', '是', '是', '好', '同意'],
                ['9000','中年', '否', '是', '非常好', '同意'],
                ['9200','中年', '否', '是', '非常好', '同意'],
                ['8600','老年', '否', '是', '非常好', '同意'],
                ['7800','老年', '否', '是', '好', '同意'],
                ['10000','老年', '是', '否', '好', '同意'],
                ['6500','老年', '是', '否', '非常好', '同意'],
                ['3000','老年', '否', '否', '一般', '拒绝'],
                ]
	return dataSet,feature_index


def calc_shannon_ent(data,col=-1):#传入样本集data，计算信息熵ent
	label_list=[d[col] for d in data]
	label_count={}#计算样本总数num，创建一个空字典label_count={}
	for label in label_list:#对于每个样本；
		label_count[label]=label_count.get(label,0)+1
		# if label not in label_count.keys():
		# 	label_count[label]=0
		# label_count[label]+=1
	N=len(data)
	ent=0.0#初始化ent=0.0;
	for value in label_count.values():#字典里的每一个值
		ent-=value/N*log2(value/N)
	return ent
data,label=createDataSet()

ent=calc_shannon_ent(data)
print(ent)

def split_data(data,index,value):#传入data,index,value,计算sub_data
#按特征的索引和特征的值获取子集，划分特征子空间
	sub_data=[]
	for d in data:
		if d[index]==value:
			d1=d.copy()
			del(d1[index])####直接del，会出错，会改动原data
							##需要将d 拷贝到d1之后,调整d1
			#删除特征原因在于建树过程中，下一个节点分裂时，就再次计算熵，这样可以减少运算量并避免该特征的影响
			sub_data.append(d1)
	return sub_data

def choose_best_index(data,):#传入data,计算best_index
	#获取信息增益最大的特征
	base_ent=calc_shannon_ent(data)
	best_IG=0.0;best_index=-1
	fea_len=len(data[0])-1
	N=len(data)
	for index in range(fea_len):#遍历每个特征
		cond_ent=0.0
		fea_space=set([d[index] for d in data])#集合的方法获取每个特征的子空间
		for value in fea_space:#内层循环：遍历该特征的每一种取值
			sub_data=split_data(data, index, value)#计算划分的子样本集
			cond_ent+=calc_shannon_ent(sub_data)*len(sub_data)/N#计算子样本的熵，通过叠加的方式计算条件熵
		IG=(base_ent-cond_ent)##/calc_shannon_ent(data,index)#计算信息增益
		if IG>best_IG:#比较得最优的信息增益获取特征
			best_IG=IG
			best_index=index
	return best_index#返回最优的特征索引
print(choose_best_index(data))

def to_leaf_node(label_list):
	label=max(set(label_list),key=label_list.count)
	return label


fea_list=[i for i in range(len(data[0])-1)]#索引列表
def create_tree(data,fea_list):
	label_list=[d[-1] for d in data]
	# if label_list.count(label_list[0])==len(label_list):
	if len(set(label_list))==1:#判断是否符合终止条件，如果是返回类别标签作为叶节点
		return label_list[0]
	if len(data[0])==1:
		return to_leaf_node(label_list)

	# if len(set(label_list))==1 or len(data[0])==1:#	如果所有样本标签相同或者只剩下一个样本，将该标签作为叶子节点；
	# 	return to_leaf_node(label_list)
	index=choose_best_index(data)#利用最优索引获得索引index

	real_index=fea_list[index]
	del(fea_list[index])#将索引从index_list中删除

	tree={'index':real_index,'child':{}}
	fea_space=set([d[index] for d in data])
	for value in fea_space:
		sub_data=split_data(data, index, value)
		tree['child'][value]=create_tree(sub_data, fea_list)
	return tree
tree=create_tree(data, fea_list)
print(tree)
treePlotter.createPlot(tree)

def predict(tree,sample):
	index=tree['index']
	value=sample[index]
	pre=tree['child'][value]
	if isinstance(tree['child'][value],dict):
		tree=tree['child'][value]
		pre=predict(tree,sample)
	return pre

print(predict(tree,[23,'老年', '否', '否', '一般', '拒绝']))