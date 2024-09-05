def highest_frequency(arr):
    dic={}
    max=0
    max_key=None
    for i in arr:
        val=dic.get(i,-1)
        if (val>max):
            max=val
            max_key=i

        if (val==-1):
            dic[i]=1
        elif (val!=-1):
            dic[i]=val+1
    
    return max_key


arr=[12, 12, 2, 12, 12, 2, 1, 2, 2, 11, 12, 2, 6]
max_key=highest_frequency(arr)
print(max_key)            