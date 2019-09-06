def check_pick(string='({[]()(([]{}{}))})'):
    def pop_(lst):
        lst.pop()
        if len(lst) != 0:
            if lst[-1] in '})]':
                return pop_(lst)
        return lst
    pick = []
    for punc in string:
        if punc in '})]':
            pick = pop_(pick)
        else:
            pick.append(punc)
    if pick == []:
        return True
    return False

def find(M):
    count = len(M)
    circle_num = 0

    def find_(i):
        _ = 0
        for j in range(i + 1,count):
            if M[i][j] == 1:
                M[i][j] = 0
                M[j][j] = 0
                _ = 1
                find_(j)
        return _


    for i in range(count):
        if find_(i):
            circle_num += 1
        elif M[i][i] == 1:
            circle_num += 1

    print(circle_num)

def checkRecord(s):
    flag =0
    if not s.count('A') > 1:
        for _ in s:
            if _ == 'L':
                flag += 1
            else:
                flag = 0
            if flag > 2:
                return False
        return True
    return False

def Pairs(dominoes):
    length = len(dominoes)
    print(length)
    all = 0
    for i in range(length - 1):
        for j in range(i + 1, length):
            if dominoes[i] == dominoes[j] or dominoes[i][::-1] == dominoes[j]:
                all += 1
                # print(i,j)
    print(all)


def reachNumber(target):
    sum = 0
    n = 0

    while True:
        if sum >= target:
            if sum == target or (sum - target) % 2 == 0:
                return n
            elif n % 2 == 0:
                return n + 1
            else:
                return n + 2
        n += 1
        sum += n


def alphabetBoardPath(target):
    """
    :type target: str
    :rtype: str
    """
    lst = dict(enumerate(list('abcdefghijklmnopqrstuvwxyz')))
    word2id = {k:v + 1 for v,k in lst.items()}
    target = [word2id[word] for word in target]
    print(target)
    way = ''
    now = [1,1]

    def run(r,c,way):
        if c > now[1]:
            way += 'R'
            now[1] += 1
            way = run(r,c,way)
        elif c < now[1]:
            way += 'L'
            now[1] -= 1
            way = run(r,c,way)
        if r < now[0]:
            way += 'U'
            now[0] -= 1
            way = run(r,c,way)
        elif r > now[0]:
            way += 'D'
            now[0] += 1
            way =run(r,c,way)
        else:
            if way[-1] != '!':
                way += '!'
            return way
        return way

    for num in target:
        r_ = num // 5
        c_ = num % 5
        r = r_ + 1 if c_ != 0 else r_
        c = 5 if c_ == 0 else c_

        print(r,c)
        way = run(r,c,way)
    return way


def stoneGameII(piles):
    pass


def largest1BorderedSquare(grid):
    r = len(grid)
    c = len(grid[0])

    max_len = 0

    max_of = max(r,c)

    def search(i, j, max_len):
        if j + max_len > c:
            return False

        for _ in range(j + 1, j + max_len):
            if grid[i - max_len + 1][_] == 0:
                return False
        for _ in range(j + 1, j + max_len):
            if grid[i][_] == 0:
                return False
        for _ in range(i, i - max_len + 1, -1):
            if grid[_][j + max_len - 1] == 0:
                return False
        for _ in range(i, i -max_len + 1,-1):
            if grid[_][j] == 0:
                return False
        return True

    for i in range(r):
        for j in range(c):
            if grid[i][j] == 1:
                for temp_max in range(max_len,max_of+1):
                    if i >= temp_max - 1:
                        if search(i,j,temp_max):
                            max_len = temp_max


    return max_len


class TreeNode:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None

def findMaximumOR(nums):
    root = TreeNode(-1)

    for num in nums:
        cur_node = root

        for i in range(31,-1,-1):
            if num & (1 << i) == 0:
                cur_node.left = TreeNode(0)
                cur_node = cur_node.left
                print(0,end='')
            else:
                cur_node.right = TreeNode(1)
                cur_node = cur_node.right
                print(1,end='')
        cur_node.left = TreeNode(num)
        print()

    res = 0
    for num in nums:
        cur_node = root

        for i in range(31,-1,-1):
            if num & (1 << i) == 0:
                if cur_node.right:
                    cur_node = cur_node.right
                    print(1,end='')
                else:
                    cur_node = cur_node.left
                    print(0,end='')
            else:
                if cur_node.left:
                    cur_node = cur_node.left
                    print(0,end='')
                else:
                    cur_node = cur_node.right
                    print(1,end='')
        print()
        temp = cur_node.left.value
        res = max(res, num ^ temp)

    return res


dict = {1:['I','V'],2:['X','L'],3:['C','D'],4:['M','']}
def intToRoman(num):
    result = ''
    for i in range(4):
        temp_num =  (num % 10 ** (i + 1) ) // 10 ** i
        if temp_num != 0:
            if temp_num != 4  and temp_num != 9:
                result = dict[i + 1][1]* (temp_num // 5) + dict[i+1][0] * (temp_num % 5) + result
            else:
                if temp_num // 4 ==1:
                    result = dict[i + 1][0] + dict[i+1][1] +result
                else:
                    result = dict[i + 1][0] + dict[i+2][0] + result
    return result


def maxRepOpt1(text):
    '''1156'''
    start = 0
    end = 1
    longest = 1
    temp = ''

    for i in range(len(text)):
        if text[i] != temp and end - start >= longest:
            pass



class MajorityChecker:
    def __init__(self,arr):
        self.arr = arr

    def query(self, left, right, threshold) :
        count = {}
        for i in self.arr[left:right + 1]:
            if i not in count:
                count[i] = 1
            else:
                count[i] += 1

            if count[i] >= threshold:
                return i
        return -1


from collections import defaultdict
from collections import Counter

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        dp = defaultdict(int)
        N = 10 ** 9 + 7

        def dpf(k, t):
            if (k, t) in dp or k <= 0 or t <= 0:
                return dp[(k, t)]
            if k == 1 and t <= f:
                return 1
            for i in range(1, f + 1):
                dp[(k, t)] = (dpf(k - 1, t - i) + dp[(k, t)]) % N
            return dp[(k, t)]

        return dpf(d, target)

    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        a_count = defaultdict(int)
        b_count = defaultdict(int)

        for i in ransomNote:
            a_count[i] += 1
        for i in magazine:
            b_count[i] += 1

        for i in a_count:
            if b_count[i] < a_count[i]:
                return False
        return True

    def movesToMakeZigzag(self, nums):
        sum = 0
        for i in range(len(nums)):
            if  i % 2 == 0:
                if i == 0:
                    if nums[0] >= nums[1]:
                        sum += nums[0] - nums[1] + 1
                elif i == len(nums) - 1:
                    if nums[-1] >= nums[-2]:
                        sum += nums[-1] - nums[-2] + 1
                else:
                    min_ = min(nums[i-1],nums[i+1])
                    if nums[i] >= min_:
                        sum += nums[i] - min_ + 1
        temp = sum
        sum = 0
        for i in range(len(nums)):
            if  i % 2 == 1:
                if i == len(nums) - 1:
                    if nums[-1] >= nums[-2]:
                        sum += nums[-1] - nums[-2] + 1
                else:
                    min_ = max(nums[i-1],nums[i+1])
                    if nums[i] <= min_:
                        sum += min_ - nums[i] + 1
        if sum < temp :
            temp = sum


        return temp


if __name__ == '__main__':

    # find([[1,0,0],
    #       [0,1,0],
    #       [0,0,1]])

    # checkRecord("PPALLL")

    # Pairs([[4,4],[8,4],[4,1],[3,9],[9,8],[5,3],[8,9],[4,8],[1,7],[5,9]])
    # n = reachNumber(5)
    # result = alphabetBoardPath('code')
    # result = stoneGameII([])
#     result = largest1BorderedSquare(
# [[1,1,1],[1,0,1],[1,1,1]])
#     result = intToRoman(9)
#     print(result)

    solution = Solution()
    # print(solution.numRollsToTarget(4,20,40))
    print(solution.movesToMakeZigzag([2,1,2]))

