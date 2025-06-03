import pickle
cache = {}
with open(f'43412depthCache.pkl', 'rb') as f:
    cache = pickle.load(f) 

keys = list(cache.keys())
values = list(cache.values())

for i in range(100):
    for key, value in cache.items():
        reward = value['reward']
        data = value['data']
        hashKey = key

        for k, v in data.items():
            print(k, v)
        print(reward)
        print(hashKey)
        print()
        
