def max_dis(arr):
    dic = {}
    max = 0
    c = 0
    val = 0
    max_key = None
    '''
    arr1[0]: first apprearence
    arr1[1]: last apprearence
    '''

    for i in arr:
        arr1 = dic.get(i, [-1, -1])

        if (arr1[0] == -1 and arr1[1] == -1):  # first appearence
            arr1[0] = c
            dic[i] = arr1
        elif (arr1[0] != -1):  # any other appearence
            arr1[1] = c
            dic[i] = arr1
            
        if (arr1[0] != -1 and arr1[1] != -1):
            val = arr1[1]-arr1[0]
            if (val >= max):
                max = val
                max_key = i
        c = c+1
    return max, max_key


arr = [1, 3, 1, 4, 5, 6, 4, 8, 3]
max, max_key = max_dis(arr)
print(max, max_key)
