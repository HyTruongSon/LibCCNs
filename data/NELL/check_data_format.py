import pickle as pkl

# Check x
obj = pkl.load(open('nell_data/ind.nell.0.1.x', 'rb'), encoding = 'latin1')
print(obj)

# Check y
obj = pkl.load(open('nell_data/ind.nell.0.1.y', 'rb'), encoding = 'latin1')
print(obj)

# Check tx
obj = pkl.load(open('nell_data/ind.nell.0.1.tx', 'rb'), encoding = 'latin1')
print(obj)

# Check ty
obj = pkl.load(open('nell_data/ind.nell.0.1.ty', 'rb'), encoding = 'latin1')
print(len(obj))

# Check allx
obj = pkl.load(open('nell_data/ind.nell.0.1.allx', 'rb'), encoding = 'latin1')
print(obj)

# Check ally
obj = pkl.load(open('nell_data/ind.nell.0.1.ally', 'rb'), encoding = 'latin1')
print(obj.shape)

# Check graph
obj = pkl.load(open('nell_data/ind.nell.0.1.graph', 'rb'), encoding = 'latin1')
print(obj)