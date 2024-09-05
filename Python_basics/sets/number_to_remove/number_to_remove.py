#finding the number of elements to be removed from an array to get an array of unique elements
def num_rem(arr):
    set1=set({})
    count=0
    for i in arr:
        if i in set1:
            count=count+1
        elif i not in set1:
            set1.add(i)
    return count

arr=[2,1,4,2,1,1,2,1,4,1,3,5,2]
print(num_rem(arr))
        