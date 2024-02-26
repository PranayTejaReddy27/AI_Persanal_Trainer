def getMaximumMEX(n, arr):
    m = 0
    for i in arr:
        if arr[i-1] == m and arr[i-1] < m :
            m += 1
            return
        elif arr[i-1] in arr[0:i]:
            m -= 1  
        else:
            arr[i-1] = m
            m += 1
    return m
# Example usage:
n = 3
arr = [2,3,2]
print(getMaximumMEX(n, arr))  
