# [-4,-2,1,3,5] to [1,-2,3,-4,5]
# you should utilize the data structure of the input list

a = [-4,-2,1,3,5]

def my_sort(a):
    if a[0] >= 0:
        return a
    
    result = []
    crit = 0
    for idx, i in enumerate(a):
        if i >= 0:
            crit = idx
            break
        
    print(crit)
    
    curr = crit
    for i in range(len(a)):
        if i == 0:
            result.append(a[crit])
            curr += 1
            crit -= 1
        else:
            if abs(a[curr]) < abs(a[crit]) or crit < 0:
                result.append(a[curr])
                curr += 1
            elif curr > len(a)-1:
                result.append(a[crit])
            else:
                result.append(a[crit])
                crit -= 1
        
        print(result, "crit = ", crit, "curr = ", curr)
    
    return result

print(my_sort(a))
            
            
            