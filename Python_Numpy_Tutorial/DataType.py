print("---Python Basic Data Type---")
print("---number---")
x = 3
print(type(x))
print(x)
print(x ** 2)
#没有x++ x--这种操作符
y = 2.5
print(type(y))
print(x, y)
print("---boolean---")
t = True
f = False
print(type(t))
print(t and f)
print(t != f)
print(f)
print("---string---")
hello = 'hello'
world = "world"
print(hello)
print(len(world))
hw = hello + ' ' + world
print(hw)
hw233 = '%s %s %d' % (hello, world, 233)
print(hw233)
print("---string methods---")
s = "hello"
print(s.capitalize()) #首字母大写
print(s.upper()) #capitalize/uppercase
print(s.rjust(7)) #right justify右对齐，限制长度，用空格填充
print(s.center(7)) #居中，跟rjust同类型的效果
print(s.replace('l', '(ell)')) #替换所有的
print('   world '.strip()) #删除所有前导和后缀空格

#容器
print("---Container---")
print("---list---")
#python的列表list就是其他语言的数组array，但是长度可变resizeable且可存不同类型
xs = [3, 1, 2]
print(xs, xs[2])
print(xs[-1]) #从后往前数，1开始计数
xs[2] = 'foo'
print(xs)
xs.append('bar')
print(xs)
x = xs.pop() #移除并返回最后一个元素
print(x, xs)
print("---list sclicing(切片)---") #concise(brief) syntax来获取子列表sublists
nums = list(range(5)) #用list转换range对象，还不懂为什么，具体的都忘记了
print(nums)
print(nums[2:4]) #后面的总是不被包含，老外习惯[,)区间
print(nums[2:])
print(nums[:2])
print(nums[:])
print(nums[:-1])
nums[2:4] = [8, 9]
print(nums)
print("---list loops---")
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
for idx, animal in enumerate(animals): #enumerate枚举，这样可以获得index，序号从零计数(是指针？)
    print('#%d: %s' % (idx + 1, animal))
print("---list comprehensions(列表推导)---") #很强的一个syntax
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares) #用循环是这么写的
nums = [0, 1, 2 ,3 ,4]
squares = [x ** 2 for x in nums] #用列表推导就可以极度简化
print(squares)
squares = [x ** 2 for x in nums if x % 2 == 0] #可以加条件
print(squares)

print("---dictionary---")
#python的字典很像其它语言的map或者JavaScript里面的object，是(key, value)的键值对
d = {'cat': 'cute', 'dog': 'furry'}
print(d['cat']) #得到一条entry，entry理解为词条/条目
print('cat' in d) #check if dictionary d has a given key, return True of False
d['fish'] = 'wet' #C++里面有类似的写法
print(d['fish'])
#print(d['monkey']) #KeyError，所以不能这么写，最好用get
print(d.get('monkey', 'N/A')) #N/A是获取默认值，没有就打印N/A
print(d.get('fish', 'N/A'))
del d['fish'] #删除一个entry
print(d.get('fish', 'N/A'))
print("---dictionary loops---")
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print("A %s has %d legs" % (animal, legs))
for animal, legs in d.items(): #这样可以获取key和对应value(corresponding)
    print("A %s has %d legs" % (animal, legs))
    
    #这个用法没搞懂
#关于d.items()的作用：以列表形式返回可遍历的(键, 值) 元组数组。
#示例，详解：
D = {'Google': 'www.google.com', 'Runoob': 'www.runoob.com', 'taobao': 'www.taobao.com'}
print("字典值 : %s" % D.items())
print("转换为列表 : %s" % list(D.items()))
for key, value in D.items():#遍历字典列表，这里的key和value分别对应D.items()生成的元组tuple里的两个元素。
    print(key, value)

print("---dictionary comprehensions")
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0} #even number偶数，字典推导和列表推导的形式差不多
print(even_num_to_square)

print("---set---") #直观上跟其他语言中的set就很像了
animals = {'cat', 'dog'}
print('cat' in animals) #data in container的句子是返回是否在其中
print('fish' in animals)
animals.add('fish')
print('fish' in animals)
print(len(animals)) #有几个元素
animals.add('cat') #does nothing
animals.remove('cat')
print(len(animals))
print("---set loops")
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals): #enumerate(container)返回一个由(序号，元素)元组构成的列表
    print('#%d: %s' % (idx + 1, animal)) #可以发现，无序，无法做顺序的假设，无论加入的顺序如何
print("---set comprehensions---")
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)} #满足后面条件的x全都要被遍历
print(nums)

print("---tuples---")
#元组是一个值的有序列表（不可改变！）。从很多方面来说，元组和列表都很相似。
#和列表最重要的不同在于，
#1.元组可以在字典中用作键，
#2.还可以作为集合的元素，
#而列表不行。例子如下：
d = {(x, x + 1): x for x in range(10)}
t = (5, 6)
tt = (5, 6, 7)
print(tt, type(t))
print(d[t]) #是个比较神奇的用法
print(d[(1, 2)])