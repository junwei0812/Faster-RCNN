

def partition(arr, i, j):
    lo = i
    hi = j
    pivot = arr[lo]
    while lo < hi:
        while hi > lo and arr[hi] >= pivot:
            hi -= 1
        while lo < hi and arr[lo] <= pivot:
            lo += 1
        arr[lo],arr[hi] = arr[hi],arr[lo]
    arr[i],arr[hi] = arr[hi],arr[i]
    return hi

def sort(arr, i, j):
    if i >= j:
        return
    m = partition(arr,i,j)
    sort(arr,i,m-1)
    sort(arr,m+1,j)

if __name__ == '__main__':
    arr = [3,2,5,1]
    sort(arr,0,len(arr)-1)
    print(arr)