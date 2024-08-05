d = {}
for line in open('/ai/zyr/NWPU/ImageSets/Main/test.txt'):
    d[line] = d.get(line, 0) + 1
fd = open('b.txt', 'w')
for k, v in d.items():
    print(k)
    if v > 1:
        print(k)
fd.close()