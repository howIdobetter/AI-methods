import copy

class A_star:
    #初始化目标状态
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.heap = [['top', 0, 0]]  #堆从索引1开始，索引0占位
        self.closed = set()          #已扩展节点集合
        self.cnt = 0                 #迭代计数器

    #堆下降操作
    def heap_down(self, index):
        t = index
        if index * 2 <= len(self.heap)-1 and self.heap[index*2][2] < self.heap[t][2]:
            t = index * 2
        if index * 2 + 1 <= len(self.heap)-1 and self.heap[index*2+1][2] < self.heap[t][2]:
            t = index * 2 + 1
        if t != index:
            self.heap[t], self.heap[index] = self.heap[index], self.heap[t]
            self.heap_down(t)

    #堆上升
    def heap_up(self, index):
        while index > 1 and self.heap[index][2] < self.heap[index//2][2]:
            self.heap[index], self.heap[index//2] = self.heap[index//2], self.heap[index]
            index = index // 2

    #堆插入
    def heap_insert(self, state_str, g, f):
        self.heap.append([state_str, g, f])
        self.heap_up(len(self.heap)-1)

    #堆pop
    def heap_pop(self):
        if len(self.heap) <= 1:
            return None
        res = self.heap[1]
        if len(self.heap) > 2:
            self.heap[1] = self.heap[-1]
            self.heap.pop()
            self.heap_down(1)
        else:
            self.heap.pop()
        return res

    #转换列表为字符串
    def list2str(self, list1):
        return ''.join(str(num) for row in list1 for num in row)

    #转换字符串为列表
    def str2list(self, s):
        return [[int(s[3*i + j]) for j in range(3)] for i in range(3)]

    #计算曼哈顿距离
    def heuristic(self, s):
        h = 0
        goal_str = self.list2str(self.goal)
        for idx in range(9):
            if s[idx] == '0':
                continue  #空格不计入
            x1, y1 = idx // 3, idx % 3
            x2, y2 = goal_str.find(s[idx]) // 3, goal_str.find(s[idx]) % 3
            h += abs(x1 - x2) + abs(y1 - y2)
        return h

    #拓展当前节点
    def a_expand(self, node):
        current_str, g = node[0], node[1]
        current_list = self.str2list(current_str)
        empty_pos = current_str.find('0')
        x, y = empty_pos // 3, empty_pos % 3

        #四个移动方向
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_list = copy.deepcopy(current_list)
                new_list[x][y], new_list[nx][ny] = new_list[nx][ny], new_list[x][y]
                new_str = self.list2str(new_list)

                #检查是否为目标
                if new_str == self.list2str(self.goal):
                    print(f"第{self.cnt}次迭代找到解！")
                    self.print_state(new_str)
                    return True

                #未处理过的加入堆
                if new_str not in self.closed:
                    h = self.heuristic(new_str)
                    self.heap_insert(new_str, g + 1, g + 1 + h)
        return False

    #打印当前状态
    def print_state(self, s):
        for i in range(3):
            print(' '.join(s[3*i:3*i+3]))
        print("--------")

    #主算法入口
    def a_star(self):
        start_str = self.list2str(self.start)
        h_start = self.heuristic(start_str)
        self.heap_insert(start_str, 0, h_start)

        while len(self.heap) > 1:
            self.cnt += 1
            current_node = self.heap_pop()
            if current_node is None:
                break
            current_str = current_node[0]

            #跳过已处理状态
            if current_str in self.closed:
                continue
            self.closed.add(current_str)

            print(f"第{self.cnt - 1}次迭代，当前状态：")
            self.print_state(current_str)

            if self.a_expand(current_node):
                return

        print("无解")

#测试案例
#注意我这个示例是无解的
print('严格按照如下格式输入对应八数码：')
print('''1 2 3
4 5 6
7 8 0
--------''')
print('在这里输入:')
start = []
for i in range(3):
    st = []
    a, b, c = map(int, input().split())
    st.append(a), st.append(b), st.append(c)
    start.append(st)
goal = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]
solver = A_star(start, goal)
solver.a_star()