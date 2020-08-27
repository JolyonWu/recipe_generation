# recipe_generation
1.环境安装
conda install pytorch=0.4.1 cuda90 -c pytorch
pip install -r requirements.txt

2.构建词汇表 
python build_vocab.py --recipe_dir /DATACENTER/3/wjl/Recipe_generation_our/
（这个recipe_dir:存放recipe数据集的地方）

3.将图片映射到LMDB 数据库，加快图片读取速度，这个操作可要可不要，如果不要这步操作，那么接下来训练的时候要加上参数--load_jpeg ，这部操作完生成的LMDB数据库将会存放在  recipe_dir文件夹下
python utils/ims2file.py --recipe_dir /DATACENTER/3/wjl/Recipe_generation_our/

4.模型的训练分两步，第一步将训练这个成分和动作生成,(我使用的是2个GPU来跑)
python train_fomer_eos.py --model_name fomer_eos --batch_size 128  --es_metric loss --loss_weight 0 5000 1000  20 20 --learning_rate 1e-4 --log_term
model_name:模型名字       batch_size：所使用的batch大小
es_metric:决定啥时候可以提前中止训练的指标，可选项有 loss 和
iou_sample           loss_weight:分别是1) instruction, 2) ingredient, 3) action 4)ingredient_eos 5)action_eos
learning_rate:学习速率 (设置在1e-4左右，根据batch_size大小调整)     


5.第二步，训练食谱生成（使用2个GPU）
python train_full.py --model_name full --batch_size 64  --transfer_from fomer_eos 
--log_term --learning_rate 1e-4 --es_metric loss --loss_weight 1000 0 0 0 0
Transfer_from:这让模型可以去寻找第一段训练的文件所在位置，所以和第一段训练时的模型名字相同

6.打开tensorboard,在/DATACENTER/3/wjl.Recipe_generation_our/recipe_generation/输入下面代码
tensorboard --logdir='./tb_logs' --port=6006 --host 0.0.0.0







###################################################################################






preprocess文件夹：

extract.py	将recipe1m_vocab_ingrs.pkl(这是其他实验从recipe1M这个数据集中抽取出来的所有ingredient)与我们数据集的文本进行匹配抽取出我们数据集的ingredient. 和action.	

accu_metric.py	计算我们匹配生成的ingredient,action和手动标注的ingredient,action之间的准确率和召回率。

annotation_ingredient.py	用来对前50个recipe进行ingredient的打标签。	

vocabulary.py	定义了一个类vocabulary,对数据集的单词进行映射成数字，包含一个clean_data()函数，这个函数可以对数据集进行清洗，去除一些多余的符号等。	

clean_dataset.py	对数据集清除一些不需要的信息，将image的地址映射到每个步骤，然后对数据集进行随机分类成训练集和测试集。	

annotation_action.py	用来对前50个recipe进行actiont的打标签。	

accu_miss_picturec.py	计算数据集中图片缺失的步骤有多少	




src文件夹：

build_vocab.py	生成ingredient和instruction的vocab表。	

data_loader.py	数据的加载模块	数据的加载可能是导致GPU利用率偏低的原因，后期可以对这个进行加速。

args.py	这是用来接收整个实验所需要的所有参数。	

demo.ipynb	这个代码用来对结果进行可视化，输出每个step生成的ingredient,action,instruction.	




model文件夹：
model_fomer_eos.py	这个模型包含了CNN模块以及2个decoder模块，主要用来做ingredient,action的预测。	

model_full.py	这是一个完整的模型，包含model_fomer_eos.py中的3个模块，以及两个encoder模块以及最后的instruction——decoder模块，这个模型的前半部分参数将会从第一步生成的模型中加载。	

train文件夹：包含了2个train文件，用来对2个部分分开训练，首先训练ingredient和action模块，最后是caption的模块。

train_fomer_eos.py	训练模型的ingredient和action部分。
	
train_full.py	训练caption的部分。	




util文件夹：
ims2file.py	将所有的图片转换成lmdb数据库里，加快图片的读取速度	

metric.py	一些评价指标以及损失函数的定义。	

output_utils.py	一些将生成的ingredient,action,instruction从数字转换为文字的函数，在demo.py文件中会使用。	

tb_visualizer.py	用来帮助进行绘图。	

utils_all.py	包含各种常用函数	









