### 1. 已知一个字符串为 “hello_world_man”，如何得到一个队列 [“hello”,”world”,”man”] 

提示：使用内置字符串api

```python
test = 'hello_world_man'

```

### 2. 有个列表 [“hello”, “world”, “man”]，如何把列表里面的字符串联起来，得到字符串 “hello_world_man”？

```python
test = ['hello', 'world', 'man']

```

### 3. 实现一个函数，统计字符串“Hello, welcome to my world.” 中字母 w 出现的次数。

### 4. 实现一个函数，给定一个整数数组，找出总和最大的连续数列，并返回总和。

```python
from typing import List

def maxSubArray(nums: List[int]) -> int:
    # 初始化当前最大和和全局最大和为列表的第一个元素
    current_max = global_max = nums[0]
    
    # 从第二个元素开始迭代，因为第一个元素已作为初始值
    for num in nums[1:]:
        # 更新当前最大和，选择最大值在于加入新元素后是否仍然最大
        current_max = max(num, current_max + num)
        
        # 更新全局最大和
        if current_max > global_max:
            global_max = current_max
    
    return global_max

# 示例使用
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
result = maxSubArray(nums)
print(result)  # 输出: 6, 对应的子数组为 [4, -1, 2, 1]
```

- **Current Max 和 Global Max**：
  - `current_max` 记录以当前元素结尾的最大子数组和。
  - `global_max` 记录迄今为止找到的最大子数组和。
- **更新策略**：
  - `current_max` 通过 `max(num, current_max + num)` 更新。此表达式处理“是否开始新的子数组”还是“把当前元素加到已有子数组上”这两种情况。
- **全局更新**：
  - 将 `current_max` 和 `global_max` 进行比较，更新 `global_max` 以保持对整个数组的最大和的记录。

### 5. 实现一个函数。完成如下功能：书店店员有一张链表形式的书单，每个节点代表一本书，节点中的值表示书的编号。为更方便整理书架，店员需要将书单倒过来排列，就可以从最后一本书开始整理，逐一将书放回到书架上。请倒序返回这个书单链表。

```python
from typing import Optional, List

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseBookList(head: Optional[ListNode]) -> ListNode:
    prev = None
    current = head

    while current is not None:
        next_node = current.next  # 暂存下一个节点
        current.next = prev       # 翻转当前节点指针
        prev = current            # 移动 prev 指针到当前节点
        current = next_node       # 移动到下一个节点

    return prev

# 构建一个链表 [3,2,6,4]
def build_linked_list(vals: List[int]) -> ListNode:
    if not vals:
        return None
    
    head = ListNode(vals[0])
    current = head
    for val in vals[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head

# 打印链表以便于验证结果
def print_linked_list(head: ListNode):
    current = head
    while current is not None:
        print(current.val, end=' ')
        current = current.next
    print()

# 测试
book_list = build_linked_list([3, 2, 6, 4])
print("原链表:")
print_linked_list(book_list)

reversed_book_list = reverseBookList(book_list)
print("翻转后的链表:")
print_linked_list(reversed_book_list)
```

- **`reverseBookList` 函数**：执行链表的反转操作。利用三个指针 `prev`, `current`, `next_node` 来逐步反转链表。
  - `prev` 初始化为 `None`，因为在反转开始时，第一个节点将成为链表结尾。
  - `current` 是遍历链表的关键，用于定位当前正在处理的节点。
  - 在循环中，第一步是保存 `current.next` 到 `next_node`，接下来将 `current.next` 反向指向 `prev`，然后移动 `prev` 和 `current` 的位置。
- **`build_linked_list` 函数**：用于从一个值列表构建链表。
- **`print_linked_list` 函数**：用于打印链表，帮助测试和验证链表的正确翻转。

这种实现方法完全处理单链表的翻转问题，并确保链表各节点的次序成功反转。该方法复杂度为 𝑂(𝑛)*O*(*n*)，其中 𝑛*n* 是链表的长度。

### 6. 实现一个双端队列类，并实现pop_back, insert_front, insert_rear, size, pop_front, is_empty 以及__init__构造方法。注意方法中应用异常处理判断异常情况。

```python
class Deque:
    def __init__(self):
        self.items = []
    
    def is_empty(self):
        """Checks whether the deque is empty."""
        return len(self.items) == 0
    
    def insert_front(self, item):
        """Inserts an item at the front of the deque."""
        self.items.insert(0, item)
    
    def insert_rear(self, item):
        """Inserts an item at the end of the deque."""
        self.items.append(item)
    
    def pop_front(self):
        """Removes and returns an item from the front of the deque."""
        if self.is_empty():
            raise IndexError("pop_front from empty deque")
        return self.items.pop(0)
    
    def pop_back(self):
        """Removes and returns an item from the end of the deque."""
        if self.is_empty():
            raise IndexError("pop_back from empty deque")
        return self.items.pop()
    
    def size(self):
        """Returns the number of items in the deque."""
        return len(self.items)

# Example usage
try:
    d = Deque()
    d.insert_rear(1)
    d.insert_front(2)
    print(d.pop_front())  # Output: 2
    print(d.pop_back())   # Output: 1
    print(d.is_empty())   # Output: True
    d.pop_front()         # This will raise an error
except IndexError as e:
    print(e)
```

1. **`__init__` 方法**：初始化一个空的列表 `items` 来存储队列元素。
2. **`is_empty` 方法**：检查队列是否为空，通过检查列表长度实现。
3. **`insert_front` 和 `insert_rear` 方法**：使用 `insert` 和 `append` 来在队首和队尾插入元素。
4. **`pop_front` 和 `pop_back` 方法**：
   - 在试图从空队列中删除元素时，引发 `IndexError` 异常。
   - 使用 `list.pop()` 来从队尾移除元素。
   - 使用 `list.pop(0)` 来从队首移除元素。
5. **`size` 方法**：返回队列中元素的数量，直接取得 `len(self.items)`。

在每个方法中对于边界条件进行了异常处理，例如在试图删除不存在的元素时，抛出了 `IndexError`。这种结构有效地示范了如何在队列的两端进行高效的操作。

### 7. 实现一个简单的哈希表类来存储和查找字符串。 首先明确什么是哈希表，然后进行实现。要求要至少实现如下方法：

```python
class SimpleHashTable:
    def __init__(self, size):
        # 初始化哈希表和大小
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def simple_hash(self, s):
        # 简单的哈希函数，将字符串的字符ASCII值相加
        hash_value = sum(ord(char) for char in s)
        return hash_value % self.size
    
    def insert(self, key):
        index = self.simple_hash(key)  # 计算键的哈希值
        # 为避免重复插入，先检查是否现有项中已经存在该键
        if key not in self.table[index]:
            self.table[index].append(key)
    
    def search(self, key) -> bool:
        index = self.simple_hash(key)  # 计算键的哈希值
        return key in self.table[index]

# 示例测试
hash_table = SimpleHashTable(10)
hash_table.insert("hello")
hash_table.insert("world")
print(hash_table.search("hello"))  # 输出: True
print(hash_table.search("python")) # 输出: False
```

1. **`__init__` 方法**：初始化哈希表的大小，并创建一个包含空列表的数组，以每个列表作为一个桶用于存储元素。
2. **`simple_hash` 方法**：使用字符的 ASCII 值之和来计算哈希值，并取模以适应表大小。该哈希函数简单且可能造成较差的分布。
3. **`insert` 方法**：计算哈希值，查找相应的槽（桶），如果不存在则插入键。避免重复插入。
4. **`search` 方法**：计算哈希值，检查对应槽内的链表是否包含键。

这种实现示范了基本的哈希表功能。由于哈希函数简单，真实应用中应使用更复杂的哈希算法以减少冲突和优化性能。通过链地址法来解决简单的冲突，在同索引的槽结点后面构建一个链表来存储冲突的多个项。

## 8. 实现一个简单的账户管理系统

### 需求概述

设计一个 Python 类 `Account` 用于模拟一个简单的银行账户管理系统。该类应能存储账户信息，并提供基本的操作，例如存款、取款和查询账户余额。

### 具体要求

1. **类的属性**：
   - `account_number`：账户号码，一个字符串。
   - `owner`：账户持有者的名字，一个字符串。
   - `balance`：账户余额，一个浮点数，初始值为 0.0。
2. **类的方法**：
   - `__init__`：初始化 `Account` 对象，接收参数设置账户号码和账户持有者的名字。
   - `deposit(amount)`：接受一个金额参数，表示存款金额。更新账户余额。如果存款金额为负，应该抛出一个异常。
   - `withdraw(amount)`：接受一个金额参数，表示取款金额。更新账户余额。如果取款金额为负或者取款金额大于账户余额，则抛出一个异常。
   - `get_balance()`：返回当前账户余额。
   - `get_owner_info()`：返回账户持有者的信息（即账户号码和持有者名字）。
3. **异常处理**：
   - 如果在存款和取款操作中输入不合法的金额（如负数或超过余额的取款），应该抛出并处理自定义异常。

### 示例

```python
class InvalidTransactionError(Exception):
    """自定义异常用于无效的交易操作"""
    pass

class Account:
    def __init__(self, account_number, owner):
        """初始化账户，设置账户号码和持有者名字"""
        self.account_number = account_number
        self.owner = owner
        self.balance = 0.0  # 初始余额为 0.0

    def deposit(self, amount):
        """存款方法，更新余额"""
        if amount < 0:
            raise InvalidTransactionError("存款金额不能为负数。")
        self.balance += amount

    def withdraw(self, amount):
        """取款方法，更新余额"""
        if amount < 0:
            raise InvalidTransactionError("取款金额不能为负数。")
        if amount > self.balance:
            raise InvalidTransactionError("取款金额超过账户余额。")
        self.balance -= amount

    def get_balance(self):
        """返回当前余额"""
        return self.balance

    def get_owner_info(self):
        """返回账户持有者的信息"""
        return f"Account: {self.account_number}, Owner: {self.owner}"

# 示例代码
try:
    # 创建一个账户实例
    my_account = Account("123456", "Alice")
    my_account.deposit(200.0)
    print(my_account.get_balance())  # 输出 200.0
    my_account.withdraw(50.0)
    print(my_account.get_balance())  # 输出 150.0
    print(my_account.get_owner_info())  # 输出 "Account: 123456, Owner: Alice"
    
    # 尝试非法操作
    my_account.withdraw(200.0)  # 这行应该抛出一个异常
except InvalidTransactionError as e:
    print("Error:", e)
```

1. **`InvalidTransactionError` 类**：这是一个自定义异常类，用于处理无效的交易操作。
2. **`__init__` 方法**：初始化账户信息，设定账户号码和持有者信息，以及初始余额为0。
3. **`deposit` 方法**：允许正值的存款操作，若为负值则会抛出 `InvalidTransactionError`。
4. **`withdraw` 方法**：允许合法的取款操作，当扣款金额为负数或超过当前余额时，抛出 `InvalidTransactionError`。
5. **`get_balance` 和 `get_owner_info` 方法**：分别返回当前账户的余额和账户的基本信息。

通过这种设计，账户类能处理常见的银行账户操作，并通过自定义异常机制优雅地处理无效的操作。

## 9. 实现一个迷你优先级队列

### 需求概述

设计并实现一个简单的优先级队列类 `PriorityQueue`，用于存储具有不同优先级的任务，并能够按照优先级顺序（从高到低）进行处理。

### 具体要求

1. **类的属性**：
   - `queue`：内部使用列表或其他适合的数据结构来存储队列元素，每个元素为一个元组 `(任务, 优先级)`。
2. **类的方法**：
   - `__init__`：初始化 `PriorityQueue` 对象。
   - `insert(task, priority)`：接受任务和其对应的优先级，将任务加入到优先级队列中。
   - `pop()`：移除并返回队列中优先级最高的任务。如果多个任务有相同的优先级，按插入顺序处理。
   - `peek()`：返回队列中优先级最高的任务，但不移除它。
   - `is_empty()`：检查队列是否为空。如队列为空返回 `True`，否则返回 `False`。
3. **异常处理**：
   - 对于 `pop()` 和 `peek()` 在队列为空的情况下，均应抛出适当的异常。

### 示例

```python
class PriorityQueue:
    def __init__(self):
        # 初始化优先队列为一个空列表
        self.queue = []
    
    def insert(self, task, priority):
        # 将任务和优先级作为元组插入到队列
        self.queue.append((priority, task))
        # 根据优先级排序（优先级高的排在后面）
        self.queue.sort(reverse=True, key=lambda x: x[0])

    def pop(self):
        # 弹出队列中优先级最高的任务
        if self.is_empty():
            raise IndexError("pop from empty priority queue")
        return self.queue.pop()[1]  # 返回任务名
    
    def peek(self):
        # 查看队列中优先级最高的任务
        if self.is_empty():
            raise IndexError("peek from empty priority queue")
        return self.queue[-1][1]  # 返回任务名
    
    def is_empty(self):
        # 判断队列是否为空
        return len(self.queue) == 0

# 示例代码
try:
    pq = PriorityQueue()
    pq.insert("Task 1", 2)
    pq.insert("Task 2", 1)
    pq.insert("Task 3", 3)
    
    print(pq.peek())  # 输出 "Task 3"
    print(pq.pop())   # 输出 "Task 3"
    print(pq.pop())   # 输出 "Task 1"
    print(pq.is_empty())  # 输出 False
    print(pq.pop())   # 输出 "Task 2"
    print(pq.is_empty())  # 输出 True
    
    pq.pop()  # 这行应该抛出一个异常
except Exception as e:
    print("Error:", e)
```

1. **`__init__` 方法**：初始化一个空的列表 `queue` 以存储任务。
2. **`insert` 方法**：将任务及其优先级作为元组添加到列表中，并在每次插入后通过 `sort` 方法根据优先级进行排序，使得优先级高的排在后面以便 `pop` 和 `peek` 操作时从队列末尾取出。
3. **`pop` 方法**：移除并返回队列中优先级最高的任务（由于优先级高的任务在队尾）。如果队列为空，抛出 `IndexError`。
4. **`peek` 方法**：查看而不移除队列中优先级最高的任务。如果队列为空，抛出 `IndexError`。
5. **`is_empty` 方法**：检查队列是否为空。

此实现使用 Python 的列表及其内置方法排序来维护任务的优先级。如果需要更高效的实现（特别是在数据规模较大时），可以考虑使用堆数据结构，如 `heapq`。

## 10. 尝试理解下面的代码，有必要的话查阅资料

一个 Python 装饰器 `@timing_decorator`，用于测量任意函数的执行时间，并在函数执行完毕后打印出该函数的运行时长。

1. **装饰器功能**：
   - 能够装饰任何函数，并在函数执行之前和之后记录时间。
   - 计算并打印函数的执行时间。
   - 支持装饰有参数的函数。
2. **使用关键字参数**：
   - 提供一个可选参数 `verbose`，用于控制是否打印详细的执行时间信息。当 `verbose=True` 时，应输出详细信息；否则，仅输出简要信息。
3. **异常处理**：
   - 确保装饰的函数在出现异常时仍能输出时间信息。

### code

```python
import time
import functools
'''
import time: 导入 Python 内置的 time 模块，主要用于获取当前的时间戳，以计算函数的执行时间。
import functools: 导入 functools 模块，其中的 wraps 函数用于确保被装饰函数的元数据信息（如函数名、文档字符串等）不会被装饰器修改。
'''
# 这是一个外层函数，它接受一个可选参数 verbose，用于控制输出的详细程度。当 verbose 为 True 时，会输出更详细的信息。
# timing_decorator 返回一个内部函数 decorator
def timing_decorator(verbose=False):
    def decorator(func): # decorator(func): 接受一个参数 func，即需要被装饰的函数。
        @functools.wraps(func) # 该函数返回一个内部函数 wrapper，实际包裹并执行 func
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                end_time = time.time()
                elapsed_time = end_time - start_time
                if verbose:
                    print(f"Function '{func.__name__}' executed in {elapsed_time:.6f} seconds with args: {args}, kwargs: {kwargs}")
                else:
                    print(f"Function '{func.__name__}' executed in {elapsed_time:.6f} seconds")
            return result
        return wrapper
    return decorator
'''
使用 @functools.wraps(func)：这是一个装饰器，wraps 是 functools 提供的一个实用工具，用于用被装饰函数的元数据更新包装函数，以保持被装饰函数的元数据信息（如函数名func.__name__、函数文档字符串 func.__doc__）
def wrapper(*args, **kwargs): wrapper 函数接受任何数量的位置参数和关键字参数，这些参数将被传递给被装饰的函数 func。
start_time = time.time(): 记录当前时间戳，为函数执行的开始时间。
try...finally 结构：确保在计算时间时无论函数 func 如何返回（成功或异常），都能记录并输出执行时间。
result = func(*args, **kwargs): 调用被装饰的函数，并捕获其返回结果。
end_time = time.time(): 记录当前时间戳，为函数执行的结束时间。
elapsed_time = end_time - start_time: 计算函数执行时间。
if verbose: 检测 verbose 参数来决定输出内容的详细程度。
print(...): 如果 verbose 为 True，打印函数名、执行时间，以及函数的传入参数和返回结果；否则，仅打印函数名和执行时间。
return result: 将函数 func 的返回值传递出 wrapper，以保持与原函数的接口一致。
'''
# 示例使用
@timing_decorator(verbose=True)
def example_function(x, y):
    time.sleep(1)  # 模拟耗时操作
    return x + y
'''
@timing_decorator(verbose=True)：使用装饰器语法糖对 example_function 应用装饰器，并设置 verbose=True。
定义 example_function(x, y)：一个示例函数，用 time.sleep(1) 模拟耗时操作并返回 x + y。
result = example_function(5, 3): 调用 example_function，由于 sleep(1)，实际会延迟约1秒后返回 8。
输出结果：函数执行时，会记录并输出 example_function 的执行时间和详细调用信息（因 verbose=True），然后打印结果为 8。
通过这种方式，开发者可以很容易地为多个函数添加时间测量功能，帮助进行性能调试和优化。通过调节 verbose 参数，可以控制信息的详细程度
'''
result = example_function(5, 3)
print("Result:", result)
```