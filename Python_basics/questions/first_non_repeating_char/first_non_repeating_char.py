def first_non_repeating_char(s):
    dic={}
    c=0
    min= 1000
    min_char=None
    for i in s:
        print(i)
        val=dic.get(i,-1)
        if (val==-1):
            dic[i]=c
        elif (val!=-1):
            dic.pop(i)
        c+=1
    for i in dic:
        if (dic[i]<min):
            min=dic[i]
            min_char=i
    return min, min_char

s="aDcadhc"
a,b=first_non_repeating_char(s)
print(a,b)