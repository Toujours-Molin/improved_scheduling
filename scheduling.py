from flask import Flask
from flask import request
import datetime
import json
import numpy as np
from dateutil.parser import parse
import random
import collections
from typing import List
import copy
import math
from operator import itemgetter
from itertools import groupby
import pandas as pd

##TODO 当前版本考虑了设备的不可用时间 2022/10/29
##TODO 当前版本融入了人员需求清单 2022/10/30

## Sequene 初始化一个需要的数据格式
info_pr = {'process_info': [[[[1, 1, 10], [2, 1, 11]], [[3, 1, 12]], [[2, 1, 13]]],
                            [[[3,1,14], [1,1,15], [2,1,16]], [[2,1,17], [3,1,18]], [[3,1,19]]],
                            [[[1,1,1],[2,1,2]]]],
           "job_nb": 3, 'total_op_nb': 7, 'machine_nb': 3,'machine_list':[1,2,3], 'machine_aval':[0,0,0],
           'periodStartDate':"2022-11-01 00:00:00",
           'job_dict':{1:['order 1', 1], 2:['order 2', 1],3:['order 3', 1]},
           'machine_dict':{1:'machine 1',2:'machine 2',3:'machine 3'}
           }


app = Flask(__name__)

## Sequence 读取当前排产周期开始日期和不可用日期，转化为不可用时段
def time_transform(start_time: str, period: list): 
    start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    target = []
    if bool(period):
        for i in period:
            date1 = parse(i[0])
            date2 = parse(i[1])
            result1 = date1 - start_time
            result2 = date2 - start_time
            target.append([int(result1.total_seconds()//60), int(result2.total_seconds()//60)])
    return target

## Sequence 将json格式数据转化为需要的数据格式
def info_pr_with_json(data_json):
    global info_pr 
    info_pr['machine_nb'] = len(data_json['machines'])
    dict_machine = {}
    info_pr['machine_dict'] = {}
    for index, machine_name in enumerate(data_json['machines']):
        dict_machine[machine_name] = index + 1
        info_pr['machine_dict'][index+1] = machine_name
    info_pr['machine_list'] = [i for i in range(1, info_pr['machine_nb'] + 1)]
    
    job_nb = 0
    for order in data_json['orders']:
        job_nb += order['productNB']
    info_pr['job_nb'] = job_nb
    info_pr['start_date'] = data_json['periodStartDate']
    info_pr['machineNotAvalPeriod'] = []
    for machine, machine_index in dict_machine.items():
        info_pr['machineNotAvalPeriod'].append(time_transform(data_json['periodStartDate'], data_json['machineNotAvalPeriod'][machine]))
    
    info_pr['process_info'] = []
    info_pr['total_op_nb'] = 0
    job_count = 1
    for order in data_json['orders']:
        for job_index_in_order in range(order['productNB']):
            info_pr['job_dict'][job_count] = [order['orderID'] , job_index_in_order + 1]
            job_count += 1
            job_info = []
            processes = data_json['process'][order['productID']]
            info_pr['total_op_nb'] += len(processes)
            for process in processes:
                process_info = []
                for index, machine in enumerate(process['machine']):
                    temp = [0,1,0]
                    # 设备index 赋值
                    temp[0] = dict_machine[machine]
                    ##! 暂时忽略夹具
                    if bool(process['fixture']):
                        temp[1] = process['fixture'][index]
                    # 加工时间赋值
                    temp[2] = process['processTime'][index]
                    temp.append(process['isInspection'])
                    process_info.append(temp)
                job_info.append(process_info)
            info_pr['process_info'].append(job_info)
    info_pr['machine_aval'] = data_json['machineAval']
    info_pr['nonProcessingTime'] = data_json['nonProcessingTime']
    ## Sequence 加入人员需求相关的信息
    info_pr['staff'] = data_json['staff']
    info_pr['cooperatedMachine'] = data_json['cooperatedMachine']
    info_pr['staffCapacity'] = data_json['staffCapacity']
    info_pr['shiftTime'] = data_json['shiftTime']
    return info_pr

## Sequence 排产算法主体
class GA_Tool:
    ## SEQUENCE 编码
    def __init__(self, Pop_size, Pc, Pm, product, operation, machine, machine_nm, PT, OS_list, index_dic, nonProcessingTime, machineNotAvalPeriod):
        self.RS_num =  int(0.5 * Pop_size)
        self.MTWR_num = int(0.1 * Pop_size)
        self.GS_num = int(0.4 * Pop_size)
        self.product = product
        self.operation = operation
        self.machine = machine
        self.machine_nm = machine_nm
        self.PT = PT
        self.OS_list = OS_list
        self.index_dic = index_dic
        self.nonProcessingTime = nonProcessingTime
        self.machineNotAvalPeriod = machineNotAvalPeriod
        self.Pop_size = Pop_size
        self.Pc = Pc
        self.Pm = Pm
    
    def Random_initial(self):
        L = len(self.OS_list)
        CHS = np.zeros((self.RS_num, L*2), dtype=int)
        OS = CHS[:, :L]
        MA = CHS[:, L:]
        for i in range(self.RS_num):
            os_list = copy.deepcopy(self.OS_list)
            random.shuffle(os_list)
            OS[i] = np.array(os_list)
            Machine_selection = []
            for j in self.product:
                for k in self.operation[j]:
                    Machine_selection.append(random.choice(self.machine[(j,k)]))
            MA[i] = np.array(Machine_selection)
        CHS1 = np.hstack((OS, MA))
        return CHS1
    
    def GS(self):
        L = len(self.OS_list)
        machine_selection = [0 for i in range(L)]
        machine_time = np.zeros(self.machine_nm, dtype=int)
        gs_product = copy.deepcopy(self.product)
        random.shuffle(gs_product)
        for i in gs_product:
            for j in self.operation[i]:
                available_machine = self.machine[(i,j)]
                judge_list = []
                for k in available_machine:
                    judge_list.append(self.PT[(i,j,k)] + machine_time[k-1])
                index = judge_list.index(min(judge_list))
                machine_index = available_machine[index]
                machine_time[machine_index-1] += self.PT[(i,j,machine_index)]
                machine_selection[self.index_dic[(i,j)]-1] = machine_index
        return np.array(machine_selection)
        
    def GS_intial(self):
        L = len(self.OS_list)
        CHS = np.zeros((self.GS_num, L*2), dtype=int)
        OS = CHS[:, :L]
        MA = CHS[:, L:]
        for i in range(self.GS_num):
            os_list = copy.deepcopy(self.OS_list)
            random.shuffle(os_list)
            OS[i] = np.array(os_list)
            MA[i] = self.GS()
        CHS2 = np.hstack((OS, MA))
        return CHS2
    
    def MTWR(self):
        L = len(self.OS_list)
        Operation_sequence = []
        candidate_list = []
        for i in self.product:
            candidate_list.append(len(self.operation[i]))
        while len(Operation_sequence) < L:
            mtw = max(candidate_list)
            index = candidate_list.index(mtw)
            Operation_sequence.append(index +1)
            if mtw >= 1:
                candidate_list[index] -= 1
        return np.array(Operation_sequence)
    
    def MTWR_initial(self):
        L = len(self.OS_list)
        CHS = np.zeros((self.MTWR_num, L*2), dtype=int)
        OS = CHS[:, :L]
        MA = CHS[:, L:]
        for i in range(self.MTWR_num):
            OS[i] = self.MTWR()
            Machine_selection = []
            for j in self.product:
                for k in self.operation[j]:
                    Machine_selection.append(random.choice(self.machine[(j,k)]))
            MA[i] = np.array(Machine_selection)
        CHS3 = np.hstack((OS, MA))
        return CHS3
    
    ## SEQUENCE 解码
    def get_OS_index(self, OS):
        osList = OS.tolist()
        OS_copy = osList.copy()
        OS_index = [] ##! OS中各工序对应在MA中的位置, 从1开始
        OS_index_PT = [] ##! OS中各工序对应该产品的第几道工序，PT:product
        L = len(self.OS_list)
        for i in range(L):
            count = collections.Counter(OS_copy)
            operation_index  = len(self.operation[OS[i]]) - count[OS[i]]
            OS_index.append(self.index_dic[(OS[i], self.operation[OS[i]][operation_index])])
            OS_index_PT.append(self.operation[OS[i]][operation_index])
            OS_copy.pop(0)
        return OS_index, OS_index_PT
    
    def find_available_gap(self, notAvalPeriod, pt, last_op_end_time):
        ##! notAvalPeriod: 机器不可用时间段
        ##! pt: 工序加工时间
        ##! nonProcessingTime: 机器间隔时间
        ##! last_op_end_time: 上道工序结束时间
        ##! 返回值：可用时间段的起始时间
        possible_start_time = []
        time_list = notAvalPeriod.copy()
        time_list.sort() ##! 按时间先后排序
        time_list.insert(0, [0,0])
        for i in range(len(time_list)-1):
            gap = time_list[i+1][0] - max(time_list[i][1], last_op_end_time)
            if gap >= pt + self.nonProcessingTime:
                if last_op_end_time == 0:
                    possible_start_time.append(time_list[i][1])
                else:
                    possible_start_time.append(max(time_list[i][1], last_op_end_time + self.nonProcessingTime))
        if possible_start_time == []:
            if last_op_end_time == 0:
                possible_start_time.append(time_list[-1][1])
            else:
                possible_start_time.append(max(time_list[-1][1], last_op_end_time + self.nonProcessingTime))
        return possible_start_time
    
    def decode(self, OS_chrom, MA_chrom):
        L = len(self.OS_list)
        schedule = np.zeros((2, L), dtype=int)
        ## machineNotAvalPeriod = res['machineNotAvalPeriod']
        updated_machineNotAvalPeriod = copy.deepcopy(self.machineNotAvalPeriod)
        OS_index, OS_index_PT = self.get_OS_index(OS_chrom)
        for i in range(L):
            current_product = OS_chrom[i]
            current_machine = MA_chrom[OS_index[i]-1] ##! 从1开始
            if OS_index_PT[i] == 1:
                last_op_end_time = 0
            else:
                last_op_end_time = schedule[1][OS_index[i]-2]
            schedule[0][OS_index[i]-1] = self.find_available_gap(updated_machineNotAvalPeriod[current_machine-1], \
            self.PT[(current_product, OS_index_PT[i], current_machine)], last_op_end_time)[0]
            schedule[1][OS_index[i]-1] = schedule[0][OS_index[i]-1] + self.PT[(current_product, OS_index_PT[i], current_machine)]
            ## 更新机器不可用时间段
            updated_machineNotAvalPeriod[current_machine-1].append([schedule[1][OS_index[i]-1], schedule[1][OS_index[i]-1] + self.nonProcessingTime])
        return schedule
    
    ## SEUENCE 适应度计算与选择
    def fitness(self, OS_chrom, MA_chrom):
        schedule = self.decode(OS_chrom, MA_chrom)
        makespan = np.max(schedule[1])
        return makespan
    
    def selection(self, CHSs):
        row = self.Pop_size
        # row = CHSs.shape[0]
        L = len(self.OS_list)
        selection = np.zeros((row, 2*L), dtype=int)
        fitness_value = []
        for i in range(CHSs.shape[0]):
            fitness_value.append(self.fitness(CHSs[i, :L], CHSs[i, L:]))
        fraction = np.array([1/fitness_value[i] for i in range(row)])
        proba = np.array([fraction[i]/np.sum(fraction) for i in range(row)])
        selected_index = np.random.choice(np.arange(row), size = int(row), replace = True, p=proba)
        for i in range(row):
            selection[i] = CHSs[selected_index[i]]
        return selection
    
    ## SEQUENCE 交叉/变异
    def multi_point_crossover(self, CHS1, CHS2):
        L = len(self.OS_list)
        OS1 = CHS1[0:L]
        OS2 = CHS2[0:L]
        MA1 = CHS1[L:2*L]
        MA2 = CHS2[L:2*L]
        new_MA1 = np.zeros(L,dtype=int)
        new_MA2 = np.zeros(L,dtype=int)
        indicator = np.random.randint(0,2,L)
        index1 = np.where(indicator==1)
        index2 = np.where(indicator==0)
        for i in index1[0]:
                new_MA1[i] = MA1[i]
                new_MA2[i] = MA2[i]
        for j in index2[0]:
                new_MA1[j] = MA2[j]
                new_MA2[j] = MA1[j]
        CHS1 = np.hstack((OS1,new_MA1))
        CHS2 = np.hstack((OS2,new_MA2))
        return CHS1, CHS2
    
    def PP_crossover(self, CHS1, CHS2):
        L = len(self.OS_list)
        OS1 = CHS1[0:L]
        OS2 = CHS2[0:L]
        MA1 = CHS1[L:2*L]
        MA2 = CHS2[L:2*L]
        indicator = np.random.randint(1,3,L)
        list1 = OS1.tolist()
        list2 = OS2.tolist()
        new_OS = []
        for i in indicator:
                if i == 1:
                        value = list1[0]
                        new_OS.append(value)
                        list2.remove(value)
                        list1.remove(value)
                if i == 2:
                        value = list2[0]
                        new_OS.append(value)
                        list1.remove(value)
                        list2.remove(value)
        new_OS = np.array(new_OS)
        CHS1 = np.hstack((new_OS,MA1))
        CHS2 = np.hstack((new_OS,MA2))
        return CHS1, CHS2
    
    def swap(self, OS):
        L = len(self.OS_list)
        index = random.sample(range(0,L), 2)
        a, b = OS[index[0]], OS[index[1]]
        OS[index[0]], OS[index[1]] = b, a
        return OS
    
    def multi_swap_mutation(self, CHS):
        L = len(self.OS_list)
        count = random.randint(3,7)
        OS = CHS[0:L]
        MA = CHS[L:2*L]
        for i in range(count):
            OS = self.swap(OS)
        CHS = np.hstack((OS,MA))
        return CHS
    
    ## SEQUENCE 迭代
    def iteration(self, Parent):        
        Children = np.zeros((self.Pop_size, 2*len(self.OS_list)), dtype=int)
        sortedChildren = np.zeros((self.Pop_size, 2*len(self.OS_list)), dtype=int)
        value_storage = [0 for i in range(self.Pop_size)]
        selected_part = self.selection(Parent)
        index_list = [i for i in range(self.Pop_size)]
        random.shuffle(index_list)
        for i in range(int(selected_part.shape[0]/2)):
            index1 = index_list[i]
            index2 = index_list[i+int(selected_part.shape[0]/2)]
            Children_candidate = []
            if random.random() < self.Pc:
                crossover1, crossover2 = self.multi_point_crossover(selected_part[index1], selected_part[index2])
                Children_candidate.append(crossover1)
                Children_candidate.append(crossover2)
            if random.random() < self.Pm:
                mutation1 = self.multi_swap_mutation(selected_part[index1])
                mutation2 = self.multi_swap_mutation(selected_part[index2])
                Children_candidate.append(mutation1)
                Children_candidate.append(mutation2)
            Children_candidate = np.array(Children_candidate)
            if Children_candidate.shape[0] == 0:
                Children[i] = selected_part[index1]
                Children[i+int(selected_part.shape[0]/2)] = selected_part[index2]
                value_storage[i] = self.fitness(selected_part[index1, :len(self.OS_list)], selected_part[index1, len(self.OS_list):])
                value_storage[i+int(selected_part.shape[0]/2)] = self.fitness(selected_part[index2, :len(self.OS_list)], selected_part[index2, len(self.OS_list):])
            else:
                candidate_fitness_value = [[i, self.fitness(Children_candidate[i, :len(self.OS_list)], Children_candidate[i, len(self.OS_list):])] for i in range(int(Children_candidate.shape[0]))]
                candidate_fitness_value.sort(key=itemgetter(1))
                Children[i] = Children_candidate[candidate_fitness_value[0][0]]
                Children[i+int(selected_part.shape[0]/2)] = Children_candidate[candidate_fitness_value[1][0]]
                value_storage[i] = candidate_fitness_value[0][1]
                value_storage[i+int(selected_part.shape[0]/2)] = candidate_fitness_value[1][1]
        index_fitness = []
        for i in range(self.Pop_size):
            index_fitness.append([i, value_storage[i]])
        index_fitness.sort(key=itemgetter(1))
        for i in range(self.Pop_size):
            sortedChildren[i] = Children[index_fitness[i][0]]
        return sortedChildren


## Sequence 基于排产计划的人员需求清单获取
class Staff_Scheduling:
    def __init__(self, staff, cooperatedMachine, staffCapacity, shiftTime, workSchedule):
        self.stuff = staff
        self.cooperatedMachine = cooperatedMachine
        self.stuffCapacity = staffCapacity
        self.shiftTime = shiftTime
        self.workSchedule = workSchedule
        
    def find_sorted_candidate(self, machine_name, stuffCapacity): #根据排产结果找到设备名称
        candidate_list=stuffCapacity[machine_name][0]
        capacity_list=stuffCapacity[machine_name][1]
        sorted_list = []
        if len(candidate_list) > 0:
            dic = {}
            for i in range(len(candidate_list)):
                dic[candidate_list[i]] =capacity_list[i]#候选人的技能字典
            sorted_dic = sorted(dic.items(), key=lambda x:x[1], reverse=False)#按照dic的第一维排序，即掌握工序数量 sorted_dic：每一行是一个字典 候选人姓名：掌握的工序数量
            for i in range(len(sorted_dic)):
                sorted_list.append(sorted_dic[i][0])#把姓名提取出来！
        else:
            sorted_list.append('缺人')
            sorted_list.append('非常缺人')
        return sorted_list

    def divid_daynight(self, workSchedule, shiftTime):
        workSchedule_DataFrame=pd.DataFrame(workSchedule)
        day_time = workSchedule_DataFrame['start_time'].str.split(' ', expand=True)#将日期和时间拆分成两列 0:day 1:time ##! start_time
        Task_Day_Time = pd.concat([workSchedule_DataFrame['machine'],day_time],axis=1)#拼接列 ##! machine
        Task_Day_Time.columns=['Task','Day','Time'] 
        List_Day_Time=Task_Day_Time.values.tolist()
        Current_Day_Time=List_Day_Time.copy()
        Current_Day_Time.sort(key=lambda x:x[1])
        Group_Day_Time=groupby(Current_Day_Time,key=lambda x:x[1]) #对机器设备按照日期进行排序和分组
        #按照日期和机器开始时间分组
        machine_dic={}
        for key,group in Group_Day_Time:
            Day_stuff_list = []
            Night_stuff_list = []
            for each in group:
                if int(each[2][0:2])>=int(shiftTime[0][0:2]) and int(each[2][0:2])<=int(shiftTime[1][0:2]):
                    Day_stuff_list.append(each[0])
                    machine_dic[(key,"dayshift")]=np.unique(Day_stuff_list)
                else:
                    Night_stuff_list.append(each[0])
                    machine_dic[(key,"nightshift")]=np.unique(Night_stuff_list)
        machine_time=sorted(machine_dic.keys())
        machine_dic_1={}
        for i in machine_time:
            machine_dic_1[i]=machine_dic[i]
        return machine_dic_1

    ##! 排班函数（输入生产计划需要的设备集合）
    def staff_schedule_for_whole_period(self):
        cooperated_info = {}
        for i in self.cooperatedMachine:
            for j in i:
                cooperated_info[j] = self.cooperatedMachine.index(i)
        total_shift = self.divid_daynight(self.workSchedule, self.shiftTime)
        total_plan = {} ##! 各班次的排班计划
        last_shift_staff = []
        
        for date, shiftType in total_shift.keys():
            current_shift_machine_list = total_shift[(date, shiftType)].tolist()
            cooperated_info_index = [cooperated_info[i] for i in current_shift_machine_list]
            updated_index = cooperated_info_index.copy()
            current_occupied = []
            current_scheduling = {}
            while len(updated_index) > 0:
                current_target = [i for i, v in enumerate(updated_index) if v == updated_index[0]]
                current_target_machine = [current_shift_machine_list[i] for i in current_target]
                available_stuff = self.find_sorted_candidate(current_target_machine[0], self.stuffCapacity)
                available_stuff = [i for i in available_stuff if i not in current_occupied and i not in last_shift_staff]
                if len(current_target) <= 2:
                    if len(available_stuff) > 0:
                        for i in current_target_machine:
                            current_scheduling[i] = available_stuff[0]
                        current_occupied.append(available_stuff[0])
                    else:
                        for i in current_target_machine:
                            current_scheduling[i] = '缺1人'
                else:
                    if len(available_stuff) >= 2:
                        for i in current_target_machine:
                            current_scheduling[i] = [available_stuff[0] + ',' + available_stuff[1]]
                            current_occupied.append(available_stuff[0])
                            current_occupied.append(available_stuff[1])
                    elif len(available_stuff) == 1:
                        for i in current_target_machine:
                            current_scheduling[i] = [available_stuff[0] + ',' + '缺1人']
                        current_occupied.append(available_stuff[0])
                    else:
                        for i in current_target_machine:
                            current_scheduling[i] = ['缺2人'] 
                updated_index = list(filter(lambda x: x != updated_index[0], updated_index)) ##! 更新设备协同的index
                current_shift_machine_list = list(filter(lambda x: x not in current_target_machine, current_shift_machine_list)) ##! 更新一下当前班次的机器列表
            total_plan[str((date, shiftType))] = current_scheduling
            last_shift_staff = current_occupied
        return total_plan


required_data = {
    "periodStartDate": "2022-11-01 08:00:00", ##! 当前排产周期的开始时间
    "shiftTime": ["08:00", "19:00", "20:00", "07:00"], ##! 白班、夜班的开始和结束时间
    "orders": [ ##! 当前排产周期订单信息
        {   
            'orderID': '001',
            'productID': '0213配油盘',
            'productNB': 2
        },
        {
            'orderID': '002',
            'productID': '12M26机体',
            'productNB': 3
        },
        {
            'orderID': '003',
            'productID': 'PSI机体',
            'productNB': 3
        }
    ],
    "process": { ##! 产品加工工艺信息
        '0213配油盘': [
            {
                'machine': ['MC6000-7'],
                'processTime': [20],
                'fixture': ['F1'],
                'isInspection': False
            },
            {
                'machine': ['MC6000-7'],
                'processTime': [35],
                'fixture': ['F1'],
                'isInspection': False
            },
            {
                'machine': ['MC6000-7'],
                'processTime': [30],
                'fixture': ['F2'],
                'isInspection': False
            },
            {
                'machine': ['MC6000-7'],
                'processTime': [30],
                'fixture': ['F2'],
                'isInspection': False
            }
        ],
        '12M26机体': [
            {
                'machine': ['MC6000-6'],
                'processTime': [50],
                'fixture': ['F3'],
                'isInspection': False
            },
            {
                'machine': ['MC6000-7'],
                'processTime': [75],
                'fixture': ['F4'],
                'isInspection': False
            },
            {
                'machine': ['MC6000-6'],
                'processTime': [60],
                'fixture': ['F4'],
                'isInspection': False
            },
            {
                'machine': ['MC6000-6'],
                'processTime': [30],
                'fixture': ['F5'],
                'isInspection': True
            }
        ],
        'PSI机体': [
            {
                'machine': ['德玛吉五轴', '乌尼恩-1#'],
                'processTime': [276, 276],
                'fixture': ['F6', 'F7'],
                'isInspection': False
            },
            {
                'machine': ['德玛吉五轴', '英赛-1#'],
                'processTime': [360, 360],
                'fixture': ['F8', 'F9'],
                'isInspection': False
            },
            {
                'machine': ['德玛吉五轴', '英赛-1#', 'BW'],
                'processTime': [276, 276, 300],
                'fixture': ['F10', 'F11', 'F12'],
                'isInspection': False
            },
            {
                'machine': ['南MTM清洗机'],
                'processTime': [252],
                'fixture': [],
                'isInspection': False
            }
        ]
    },
    "machines": ['MC6000-6', 'MC6000-7', '德玛吉五轴', '乌尼恩-1#', '英赛-1#', 'BW', '南MTM清洗机'], ##! 当前排产周期的设备列表
    "cooperatedMachine": [['MC6000-6', 'MC6000-7'], ['德玛吉五轴', '乌尼恩-1#'], ['英赛-1#', 'BW'], ['南MTM清洗机']], ##! 当前排产周期的设备协同关系
    "staff": ['张成龙','崔焕友','刘宝强','臧志远','张晓辉','冯金明'], ##! 当前排产周期的员工列表
    "staffCapacity": {
        'MC6000-6': [['崔焕友','刘宝强','张成龙'],[4,4,6]],
        'MC6000-7': [['崔焕友','刘宝强','张成龙'],[4,4,6]],
        '德玛吉五轴': [['张成龙','臧志远','张晓辉','冯金明'],[6,4,4,8]],
        '乌尼恩-1#': [['张成龙','臧志远','张晓辉','冯金明'],[6,4,4,8]],
        '英赛-1#': [['张成龙','臧志远','张晓辉','冯金明'],[6,4,4,8]],
        'BW':[['张成龙','臧志远','张晓辉','冯金明'],[6,4,4,8]],
        '南MTM清洗机': [['张成龙','张晓辉','冯金明'],[6,4,8]]}, ##! 当前排产周期的员工能力
    "machineAval": [0, 0, 0, 0, 0, 0, 0], ##! 当前排产周期的设备可用时间
    "machineNotAvalPeriod": { ##! 当前排产周期的设备不可用时间段
        'MC6000-6': [['2022-11-01 01:00:00', '2022-11-02 08:00:00'], ['2022-11-02 19:00:00', '2022-11-03 08:00:00']], ##! 夜班不可用
        'MC6000-7': [['2022-11-01 19:00:00', '2022-11-02 08:00:00'], ['2022-11-02 19:00:00', '2022-11-03 08:00:00']],
        '德玛吉五轴': [],
        '乌尼恩-1#': [],
        '英赛-1#': [],
        'BW': [],
        '南MTM清洗机': [['2022-11-01 12:00:00', '2022-11-01 18:00:00']] ##! 维修时间
    },
    "nonProcessingTime": 15, ##! 上下料、换夹具等非加工时间
}


@app.route('/schedule', methods=['GET', 'POST'])
def schedule():
    start_temps = datetime.datetime.now()
    if request.method == 'POST':
        data = request.get_data()
        data = json.load(data)
        info_pr_with_json(data)
    else:
        info_pr_with_json(required_data)
    
    machine_nm = info_pr['machine_nb']
    nonProcessingTime = info_pr['nonProcessingTime']
    machineNotAvalPeriod = info_pr['machineNotAvalPeriod']
    
    JOB = [i for i in range(1, info_pr['job_nb'] + 1)]
    OPERATION = {}
    for i in JOB:
        OPERATION[i] = [i for i in range(1, 1+ len(info_pr['process_info'][i-1]))]
    MACHINE = {}
    pt = {}
    PT = {}
    for i in JOB:
        for j in OPERATION[i]:
            MACHINE[(i,j)] = [info_pr['process_info'][i-1][j-1][k][0] for k in range(len(info_pr['process_info'][i-1][j-1]))]
            pt[(i,j)] = [info_pr['process_info'][i-1][j-1][k][2] for k in range(len(info_pr['process_info'][i-1][j-1]))]
            dic_ref = dict(zip(MACHINE[(i,j)], pt[(i,j)]))
            for k in MACHINE[(i,j)]:
                PT[(i,j,k)] = dic_ref[k]
    INDEX_DIC = {}
    lengh = 0
    for i in JOB:
        for j in OPERATION[i]:
            INDEX_DIC[(i,j)] = lengh + OPERATION[i].index(j) + 1
        lengh += len(OPERATION[i])
        
    OS_list = []
    for i in JOB:
        for j in OPERATION[i]:
            OS_list.append(i)
    
    ## Sequence 迭代参数
    Pop_size = 20
    Pc = 0.7
    Pm = 0.2
    
    initial = GA_Tool(Pop_size=Pop_size, Pc=Pc, Pm=Pm, product=JOB ,operation=OPERATION, machine=MACHINE, \
        machine_nm=machine_nm, PT=PT, OS_list=OS_list, index_dic=INDEX_DIC, \
            nonProcessingTime=nonProcessingTime, machineNotAvalPeriod=machineNotAvalPeriod)
    
    CHS1 = initial.Random_initial()
    CHS2 = initial.GS_intial()
    CHS3 = initial.MTWR_initial()
    Parent = np.vstack((CHS1, CHS2, CHS3))
    
    history = []
    optimal_value = 100000 ##! 初始最优
    best_CHS = np.zeros(2*len(OS_list), dtype=int)
    history.append(initial.fitness(Parent[0][:len(OS_list)], Parent[0][len(OS_list):]))
    for i in range(70): ##! 迭代次数
        Parent = initial.iteration(Parent)
        opt_value = initial.fitness(Parent[0][:len(OS_list)], Parent[0][len(OS_list):])
        if opt_value < optimal_value:
            optimal_value = opt_value
            best_CHS = Parent[0]
        history.append(optimal_value)
    print(history)
    final_plan = initial.decode(best_CHS[:len(OS_list)], best_CHS[len(OS_list):])
    print(final_plan)
    
    end_temps = datetime.datetime.now()
    
    print("elapsedTime: " + str(end_temps - start_temps))
    
    df = []
    for i in JOB:
        for j in OPERATION[i]:
            periodStart = datetime.datetime.strptime(info_pr['start_date'], '%Y-%m-%d %H:%M:%S')
            begin = periodStart + datetime.timedelta(minutes=int(final_plan[0][INDEX_DIC[(i,j)]-1]))
            begin = begin.strftime('%Y-%m-%d %H:%M:%S')
            end = periodStart + datetime.timedelta(minutes=int(final_plan[1][INDEX_DIC[(i,j)]-1]))
            end = end.strftime('%Y-%m-%d %H:%M:%S')
            selected_machine = info_pr['machine_dict'][best_CHS[len(OS_list)+INDEX_DIC[(i,j)]-1]]
            job_name = info_pr['job_dict'][i]
            processID = j
            isInsepction = info_pr['process_info'][i-1][j-1][0][3]
            fixture = info_pr['process_info'][i-1][j-1][0][1]
            df.append(dict(job=job_name, processID=processID, machine=selected_machine, fixture=fixture, start_time=begin, end_time=end, isInsepction=isInsepction))
    
    df2 = df.copy()
    
    staff = info_pr['staff']
    cooperatedMachine = info_pr['cooperatedMachine']
    staffCapacity = info_pr['staffCapacity']
    shiftTime = info_pr['shiftTime']
    workSchedule = df2
    
    staff_plan = Staff_Scheduling(staff, cooperatedMachine, staffCapacity, shiftTime, workSchedule)
    total_plan = staff_plan.staff_schedule_for_whole_period()
    print(total_plan)
    
    respond = {'makespan': str(optimal_value),'elapsedTime': str(end_temps-start_temps), 'workSchedule': df, 'staffSchedule': total_plan}
    return json.dumps(respond)
            
    


if __name__ == '__main__':
    app.run()
    
    

