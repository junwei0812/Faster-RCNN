
def merge(arr,lo,mid,hi):
    aux = arr.copy()
    i = lo
    j = mid+1
    for k in range(lo,hi+1):
        if i > mid:
            arr[k] = aux[j]
            j += 1
        elif j > hi:
            arr[k] = aux[i]
            i += 1
        elif aux[i] < aux[j]:
            arr[k] = aux[i]
            i+= 1
        else:
            arr[k] = aux[j]
            j += 1
def sort(arr,i,j):
    if i >= j:
        return
    mid = i + (j-i)//2
    sort(arr,i,mid)
    sort(arr,mid+1,j)
    merge(arr,i,mid,j)

if __name__ == '__main__':
    arr = [3,5,1,42]
    sort(arr,0,len(arr)-1)
    print(arr)