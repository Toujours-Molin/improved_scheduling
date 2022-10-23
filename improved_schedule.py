from flask import Flask
from flask import request
import datetime
import json
import numpy as np
import random
import collections
from typing import List
import copy
import math

# from machine_selection import GMT_select
# from info_GA import data_conversion, get_bottleneck
# from chromo_simp import chromo_simp
# from functions import get_op_to_ma
# # from algo.data_processing import get_os_list, raw_data_to_compact_data, info_pr_init
# import data_processing
# from GA_operators import GA_Tools

info_pr = {'process_info': [[[[1, 1, 10], [2, 1, 11]], [[3, 1, 12]], [[2, 1, 13]]],
                            [[[3,1,14], [1,1,15], [2,1,16]], [[2,1,17], [3,1,18]], [[3,1,19]]],
                            [[[1,1,1],[2,1,2]]]],
           "job_nb": 3, 'total_op_nb': 7, 'machine_nb': 3,'machine_list':[1,2,3], 'machine_aval':[0,0,0],
           'start_date':"2022-01-01 00:00:00",
           'job_dict':{1:['order 1', 1], 2:['order 2', 1],3:['order 3', 1]},
           'machine_dict':{1:'machine 1',2:'machine 2',3:'machine 3'}
           }

app = Flask(__name__)



def info_pr_init_with_json(data_json):
    global info_pr
    info_pr['machine_nb'] = len(data_json['machines'])
    # machine_dict key 为mechine name，value为从1开始的index
    machine_dict = {}
    for index, machine_name in enumerate(data_json['machines']):
        machine_dict[machine_name] = index + 1
        info_pr['machine_dict'][index+1] = machine_name
    info_pr['machine_list'] = [i for i in range(1, len(data_json['machines'])+1)]

    job_nb = 0
    for order in data_json['orders']:
        job_nb += order['productNB']
    info_pr['start_date'] = data_json['orders'][0]['dateBegin'] + ' 00:00:00' ##! 这里的时间需要改为periodStartDate
    info_pr['job_nb'] = job_nb

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
                    temp[0] = machine_dict[machine]
                    # 加工时间赋值
                    temp[2] = process['processTime'][index]
                    process_info.append(temp)
                job_info.append(process_info)
            info_pr['process_info'].append(job_info)
    info_pr['machine_aval'] = data_json['machineAval']


def get_os_list():
    global info_pr
    os_list = []
    for job in info_pr['process_info']:
        os_list.append(len(job))
    return os_list

def GMT_select(high_failure_machine: list = None): ##！需要考虑机器的不可用时间（求和），然后用GMT（arr矩阵：行表示工序，列表示设备）
    """
    :param high_failure_machine: a list of high failure machine NUMBER
    :return:
    """
    magic_large = math.inf ##！这里的magic_large是一个很大的数，用来表示设备不可用的情况 9999999999999
    # ms generation
    ms = [0] * info_pr['total_op_nb'] 
    arr_primal = np.array(get_all_pr_list(magic_large=magic_large), dtype=np.int64)
    arr = np.copy(arr_primal)
    # print(arr)
    for _ in range(len(ms)):
        position_array = np.where(arr == np.min(arr))
        num = position_array[0].size
        choice_list = list(range(num))
        flag = False
        while not flag:
            selected_num = random.choice(choice_list)
            selected_machine = position_array[1][selected_num]
            selected_op = position_array[0][selected_num]
            if (not bool(high_failure_machine)) or num == 1 or selected_machine+1 not in high_failure_machine: ##?含义？
                flag = True
            else:
                choice_list.remove(selected_num)
        if ms[selected_op] == 0:
            selected_column = position_array[1][selected_num]
            op_list = arr[selected_op]
            ms_code = 0
            for i in range(selected_column + 1):
                if op_list[i] < magic_large:
                    ms_code += 1
            ms[selected_op] = ms_code
            # update arr
            pr_time_prime = arr_primal[selected_op][selected_column]
            arr[:, selected_column] = arr[:, selected_column] + np.array([pr_time_prime] * arr.shape[0])
            arr[selected_op] = np.array([magic_large] * arr.shape[1])
        else:
            raise Exception('get to the ms position {} already processed'.format(selected_op))
        # print(arr)
    return ms

def get_all_pr_list(magic_large: int): ##！初始化arr矩阵
    """
    the pr time of the machine that could not be used is set to a large number 9999
    :return:
    """
    pr_list = [[magic_large] * info_pr['machine_nb'] for i in range(info_pr['total_op_nb'])]
    index = 0
    for job in info_pr['process_info']:
        for op in job:
            for alternative in op:
                column_nb = info_pr['machine_list'].index(alternative[0])
                pr_list[index][column_nb] = alternative[2]
            index += 1
    return pr_list

def data_conversion(ms: list): 
    info_ma = collections.defaultdict(list)
    ms_upper_list = []
    # ms_upper example:[ (start of info o11 (1,1 ->o11),((1,10)->machine 1 with pr time 10,(2,11)) end of info o11),...]
    for job_index, job in enumerate(info_pr['process_info']):
        for op_index, operation in enumerate(job):
            list_temp = []
            for machine in operation:
                list_temp.append((machine[0], machine[2]))
            list_temp.sort()
            ms_upper_list.append(((job_index + 1, op_index + 1), tuple(list_temp)))
    for position, code in enumerate(ms):
        machine_selected = ms_upper_list[position][1][code - 1][0]
        pr_time = ms_upper_list[position][1][code - 1][1]
        op_info = ms_upper_list[position][0]
        info_ma[machine_selected].append(op_info + (pr_time,))
    return info_ma

def get_bottleneck(info_ma: dict):
    list_machine = []
    list_occupation = []
    for machine, operations in info_ma.items():
        occupation = 0
        if operations:
            for operation in operations:
                occupation += operation[2]
        list_machine.append(int(machine))
        list_occupation.append(occupation)
    if len(list_machine) <= 2:
        list_machine.sort()
        return list_machine
    bottleneck = []
    max_ocp = max(list_occupation)
    for i, ocp in enumerate(list_occupation):
        if ocp >= 0.7*max_ocp:
            bottleneck.append(list_machine[i])
    if len(bottleneck) < 3:
        sorted_occupation = sorted(list_occupation)
        while len(bottleneck) < 3:
            occu = sorted_occupation.pop()
            position_list = []
            for i in range(len(list_occupation)):
                if list_occupation[i] == occu:
                    position_list.append(i)
            for i in position_list:
                if list_machine[i] not in bottleneck:
                    bottleneck.append(list_machine[i])
    bottleneck.sort()
    return bottleneck ##! 返回瓶颈设备,至少三台，最多0.7*max_ocp


class chromo_simp:
    __gene = None

    def __init__(self, gene=None):
        self.__gene = gene

    @classmethod
    def random_init(cls, info_ma: dict, bottleneck: tuple):
        gene = {}
        for machine in bottleneck:
            gene[machine] = list(info_ma[machine])
            random.shuffle(gene[machine])
        return cls(gene)

    def __str__(self):
        return "gene:" + str(self.__gene) + "\n"

    def get_gene(self, machine: int):
        return self.__gene[machine]

    def decode(self, info_ma: dict, os_list: list, bottleneck: tuple):##！不用此函数 当前版本
        # first tuple element is pr_time, second tuple element is machine number
        op_ma = get_op_to_ma(info_ma)

        job_nb = info_pr['job_nb']
        job_makespan = [0] * job_nb
        machine_nb = info_pr['machine_nb']
        machine_makespan = [0] * machine_nb

        # [interval start time, interval end time, interval lasting time]
        # !!!!keep it in ascending order
        machine_interval = collections.defaultdict(list)

        machine_working = collections.defaultdict(list)
        op_current = [1 for _ in range(job_nb)]
        op_over = ['over' for _ in range(job_nb)]
        while op_current != op_over:
            # construct the start and end time for each job
            # 5-element tuples with a total of job_nb, tuple:(job nb, op nb, machine_number, start time, end time)
            job_info_set = []

            # a tuple including (job num, op num, machine num, finish time)
            finish_first_record = None

            # update job info set
            for job_minus, op in enumerate(op_current):
                if op == 'over':
                    continue
                pr_time = op_ma[(job_minus + 1, op)][0]
                machine_chosen = op_ma[(job_minus + 1, op)][1]

                # balance between machine and operation sequence
                # regarding operation sequence, the earliest possible is current job_makespan
                earliest = job_makespan[job_minus]
                # regarding machine, check all possible intervals
                interval_chosen = False
                if machine_interval[machine_chosen]:
                    for interval in machine_interval[machine_chosen]:
                        if interval[1]-pr_time >= earliest and interval[2] >= pr_time:
                            earliest = max(earliest, interval[0])
                            interval_chosen = True
                            break
                if not interval_chosen:
                    index = info_pr['machine_list'].index(machine_chosen)
                    earliest = max(machine_makespan[index], earliest)

                job_info_set.append((job_minus+1, op, machine_chosen, earliest, earliest+pr_time))

                # find the earliest finish time
                if (not finish_first_record) or (finish_first_record[4] > earliest+pr_time):
                    finish_first_record = (job_minus+1, op, machine_chosen, earliest, earliest+pr_time)

            # according to finish first record, find the job in job_info_set, construct the collision set
            # job_info in collision set, list of tuples:(job nb, op nb, machine_number, start time, end time)

            # print(job_info_set)
            # print(finish_first_record)
            collision_set = [finish_first_record]
            for job_info in job_info_set:
                if job_info[2] == finish_first_record[2] and job_info[4] < finish_first_record[4]:
                    collision_set.append(job_info)
            # print(collision_set)

            # choose the one to be scheduled
            # if the machine is bottleneck machine, schedule by code order
            to_schedule = None
            if finish_first_record[2] in bottleneck:
                # get gene slice
                gene_slice = self.__gene[finish_first_record[2]]
                to_schedule_found = False
                for op_info in gene_slice:
                    for collision_op_info in collision_set:
                        if op_info[0:2] == collision_op_info[0:2]:
                            to_schedule = collision_op_info
                            to_schedule_found = True
                            break
                    if to_schedule_found:
                        break
            # if the machine is not bottleneck, find the shortest processing time and schedule
            else:
                for job_info in collision_set:
                    if not to_schedule or job_info[4] - job_info[3] < to_schedule[4] - to_schedule[3]:
                        to_schedule = job_info

            # according to to_schedule, update machine status, job status
            # to_schedule :(job nb, op nb, machine_number, start time, end time)
            # print(to_schedule)
            job_makespan[to_schedule[0] - 1] = to_schedule[4]

            # update machine_interval
            interval_filled = False
            for interval_position, interval_info in enumerate(machine_interval[to_schedule[2]]):
                # interval_info : [interval start time, interval end time, interval lasting time]
                if interval_info[0] <= to_schedule[3] and interval_info[1] >= to_schedule[4]:
                    if interval_info[0] == to_schedule[3] and interval_info[1] == to_schedule[4]:
                        del machine_interval[to_schedule[2]][interval_position]
                    elif interval_info[0] == to_schedule[3]:
                        machine_interval[to_schedule[2]][interval_position] = [to_schedule[4], interval_info[1],
                                                                               interval_info[1]-to_schedule[4]]
                    elif interval_info[1] == to_schedule[4]:
                        machine_interval[to_schedule[2]][interval_position] = [interval_info[0], to_schedule[3],
                                                                               to_schedule[3]-interval_info[0]]
                    else:
                        machine_interval[to_schedule[2]][interval_position:interval_position+1] = \
                            [[interval_info[0], to_schedule[3], to_schedule[3]-interval_info[0]],
                              [to_schedule[4],interval_info[1],interval_info[1]-to_schedule[4]]]
                    interval_filled = True
                    break
            # update_machine_makespan
            # 这里似乎有个bug，没有更新machine interval
            if not interval_filled:
                index = info_pr['machine_list'].index(to_schedule[2])
                machine_makespan[index] = to_schedule[4]
            # update op current
            op_current[to_schedule[0] - 1] += 1
            if op_current[to_schedule[0] - 1] > os_list[to_schedule[0] - 1]:
                op_current[to_schedule[0] - 1] = 'over'
        assert max(job_makespan) == max(machine_makespan)
        return max(job_makespan)

    def decode_with_response(self, info_ma: dict, os_list: list, bottleneck: tuple): ##！ 解码考虑不可用时间
        # first tuple element is pr_time, second tuple element is machine number
        op_ma = get_op_to_ma(info_ma)

        job_nb = info_pr['job_nb']
        job_makespan = [0] * job_nb
        machine_nb = info_pr['machine_nb']
        machine_makespan = info_pr['machine_aval']

        # [interval start time, interval end time, interval lasting time]
        # !!!!keep it in ascending order
        machine_interval = collections.defaultdict(list)

        machine_working = collections.defaultdict(list)
        op_current = [1 for _ in range(job_nb)]
        op_over = ['over' for _ in range(job_nb)]
        while op_current != op_over:
            # construct the start and end time for each job
            # 5-element tuples with a total of job_nb, tuple:(job nb, op nb, machine_number, start time, end time)
            job_info_set = []

            # a tuple including (job num, op num, machine num, finish time)
            finish_first_record = None

            # update job info set
            for job_minus, op in enumerate(op_current):
                if op == 'over':
                    continue
                pr_time = op_ma[(job_minus + 1, op)][0]
                machine_chosen = op_ma[(job_minus + 1, op)][1]

                # balance between machine and operation sequence
                # regarding operation sequence, the earliest possible is current job_makespan
                earliest = job_makespan[job_minus]
                # regarding machine, check all possible intervals
                interval_chosen = False
                if machine_interval[machine_chosen]:
                    for interval in machine_interval[machine_chosen]:
                        if interval[1]-pr_time >= earliest and interval[2] >= pr_time:
                            earliest = max(earliest, interval[0])
                            interval_chosen = True
                            break
                if not interval_chosen:
                    index = info_pr['machine_list'].index(machine_chosen)
                    earliest = max(machine_makespan[index], earliest)

                job_info_set.append((job_minus+1, op, machine_chosen, earliest, earliest+pr_time))

                # find the earliest finish time
                if (not finish_first_record) or (finish_first_record[4] > earliest+pr_time):
                    finish_first_record = (job_minus+1, op, machine_chosen, earliest, earliest+pr_time)

            # according to finish first record, find the job in job_info_set, construct the collision set
            # job_info in collision set, list of tuples:(job nb, op nb, machine_number, start time, end time)

            # print(job_info_set)
            # print(finish_first_record)
            collision_set = [finish_first_record] ##！ 冲突集
            for job_info in job_info_set:
                if job_info[2] == finish_first_record[2] and job_info[4] < finish_first_record[4]:
                    collision_set.append(job_info)
            # print(collision_set)

            # choose the one to be scheduled
            # if the machine is bottleneck machine, schedule by code order
            to_schedule = None
            if finish_first_record[2] in bottleneck:
                # get gene slice
                gene_slice = self.__gene[finish_first_record[2]]
                to_schedule_found = False
                for op_info in gene_slice:
                    for collision_op_info in collision_set:
                        if op_info[0:2] == collision_op_info[0:2]:
                            to_schedule = collision_op_info
                            to_schedule_found = True
                            break
                    if to_schedule_found:
                        break
            # if the machine is not bottleneck, find the shortest processing time and schedule
            else:
                for job_info in collision_set:
                    if not to_schedule or job_info[4] - job_info[3] < to_schedule[4] - to_schedule[3]:
                        to_schedule = job_info

            # according to to_schedule, update machine status, job status
            # to_schedule :(job nb, op nb, machine_number, start time, end time)
            # print(to_schedule)
            job_makespan[to_schedule[0] - 1] = to_schedule[4]

            # update machine working
            # info for every machine [job_nb, start time, end time]
            machine_working[to_schedule[2]].append([to_schedule[0], to_schedule[3], to_schedule[4], to_schedule[1]])

            # update machine_interval
            interval_filled = False
            for interval_position, interval_info in enumerate(machine_interval[to_schedule[2]]):
                # interval_info : [interval start time, interval end time, interval lasting time]
                if interval_info[0] <= to_schedule[3] and interval_info[1] >= to_schedule[4]:
                    if interval_info[0] == to_schedule[3] and interval_info[1] == to_schedule[4]:
                        del machine_interval[to_schedule[2]][interval_position]
                    elif interval_info[0] == to_schedule[3]:
                        machine_interval[to_schedule[2]][interval_position] = [to_schedule[4], interval_info[1],
                                                                               interval_info[1]-to_schedule[4]]
                    elif interval_info[1] == to_schedule[4]:
                        machine_interval[to_schedule[2]][interval_position] = [interval_info[0], to_schedule[3],
                                                                               to_schedule[3]-interval_info[0]]
                    else:
                        machine_interval[to_schedule[2]][interval_position:interval_position+1] = \
                            [[interval_info[0], to_schedule[3], to_schedule[3]-interval_info[0]],
                              [to_schedule[4],interval_info[1],interval_info[1]-to_schedule[4]]]
                    interval_filled = True
                    break
            # update_machine_makespan
            if not interval_filled:
                index = info_pr['machine_list'].index(to_schedule[2])
                machine_makespan[index] = to_schedule[4]
            # update op current
            op_current[to_schedule[0] - 1] += 1
            if op_current[to_schedule[0] - 1] > os_list[to_schedule[0] - 1]:
                op_current[to_schedule[0] - 1] = 'over'
        # assert max(job_makespan) == max(machine_makespan)
        # generate respond
        df = []
        time_0 = datetime.datetime.strptime(info_pr['start_date'], "%Y-%m-%d %H:%M:%S")
        for machine in machine_working:
            for work in machine_working[machine]:
                start = time_0 + datetime.timedelta(minutes=work[1])
                finish = time_0 + datetime.timedelta(minutes=work[2])
                machine_name = info_pr['machine_dict'][machine]
                job_name = info_pr['job_dict'][work[0]]
                processID = work[3]
                df.append(dict(machine=machine_name, startTime='{}'.format(start),
                               endTime='{}'.format(finish), job=job_name, processID=processID, fixture=None,
                               tool=None, pallet=None))
        return df


class GA_Tools:
    def __init__(self, population_size: int, crossover_rate: float, mutation_rate: float, elite_number: int,
                 info_ma: dict, os_list: list, bottleneck: tuple):
        self.__info_ma = info_ma
        self.__os_list = os_list
        self.__bottleneck = bottleneck
        self.__population_size = population_size
        self.__mutation_rate = mutation_rate
        self.__crossover_rate = crossover_rate
        if elite_number >= population_size:
            raise Exception('elite number too large, '
                            'population size is {}, while elite number is {}'.format(population_size, elite_number))
        else:
            self.__elite_number = elite_number

    def population_init(self):
        population = []
        while len(population) < self.__population_size:
            population.append(chromo_simp.random_init(self.__info_ma, self.__bottleneck))
        return population

    def crossover_operator(self, parent_1: chromo_simp, parent_2: chromo_simp):
        if random.random() > self.__crossover_rate:
            return
        machine_chosen = random.choice(self.__bottleneck)
        gene_1 = parent_1.get_gene(machine_chosen)
        gene_2 = parent_2.get_gene(machine_chosen)
        assert len(gene_1) == len(gene_2)
        if len(gene_1) <= 2:
            return
        slice_chosen = sorted(random.sample(list(range(len(gene_1))),k=2))
        # print('slice_chosen', slice_chosen)
        # print('machine_chosen', machine_chosen)
        cut_1 = gene_1[slice_chosen[0]: slice_chosen[1]]
        cut_2 = gene_2[slice_chosen[0]: slice_chosen[1]]
        for i in range(len(cut_1)):
            gene_1.remove(cut_2[i])
            gene_2.remove(cut_1[i])
        gene_1[slice_chosen[0]:slice_chosen[0]] = cut_2
        gene_2[slice_chosen[0]:slice_chosen[0]] = cut_1

    def mutation_operator(self, chromo: chromo_simp):
        if random.random() > self.__mutation_rate:
            return
        machine_chosen = random.choice(self.__bottleneck)
        gene = chromo.get_gene(machine_chosen)
        length = len(gene)
        if length > 1:
            selected_positions = sorted(random.sample(range(length), k=2))
            # print('machine_chosen', machine_chosen)
            # print('selected_positions', selected_positions)
            code = gene.pop(selected_positions[1])
            gene.insert(selected_positions[0], code)

    def population_crossover_mutation(self, population: List[chromo_simp]):
        length = len(population)
        pair_index_1, pair_index_2 = pair_up(list(range(self.__elite_number, length)))
        for i in range(len(pair_index_1)):
            self.crossover_operator(population[pair_index_1[i]], population[pair_index_2[i]])
        for i in range(self.__elite_number, length):
            self.mutation_operator(population[i])

    def generate_next_raw_generation(self, old_population: List[chromo_simp], population_fitness: list):
        """
                elite keeping with tournament selection. selection only, without crossover and mutation
                :param old_population:
                :param population_fitness:
                :return:
                """
        length = len(old_population)
        sorted_index = get_sorted_popu_index(population_fitness)
        next_generation = []
        for i in range(self.__elite_number):
            next_generation.append(copy.deepcopy(old_population[sorted_index[i]]))
        while len(next_generation) < len(old_population):
            index_chosen = random.sample(range(length), k=2)
            if population_fitness[index_chosen[0]] < population_fitness[index_chosen[1]]:
                next_generation.append(copy.deepcopy(old_population[index_chosen[0]]))
            else:
                next_generation.append(copy.deepcopy(old_population[index_chosen[1]]))
        return next_generation

    def population_decode(self, population: List[chromo_simp]):
        population_fitness = []
        for item in population:
            population_fitness.append(item.decode(info_ma=self.__info_ma, os_list=self.__os_list,
                                                  bottleneck=self.__bottleneck))
        return population_fitness


def get_op_to_ma(info_ma: dict):
    """
    :return: a dictionary with the key of (1,1)->op11, value (56, 3)-> pr time of 56 for machine 3
    """
    op_ma = {}
    for machine, ops in info_ma.items():
        for op in ops:
            op_ma[op[:2]] = (op[2], int(machine))
    return op_ma


def pair_up(index_list: list):
    """
    WARNING : this function modifies the input list directly
    :param index_list:
    :return: 2 paired list, the second one may be longer for an odd length of index_list
    """
    length = len(index_list)
    result_list_1 = random.sample(index_list, k=int(length/2))
    for i in result_list_1:
        index_list.remove(i)
    return result_list_1, index_list


def get_sorted_popu_index(population_fitness_list: list):
    """
    :param population_fitness_list: should be a list of population fitness, corresponding with the population_list
    :return: in an ascending order, the index of the chromosomes
    """
    sorted_index = sorted(range(len(population_fitness_list)), key=lambda k: population_fitness_list[k])
    return sorted_index


@app.route('/schedule', methods=['GET', 'POST'])
def home():
    iteration_limit = 5
    start_time = datetime.datetime.now()
    if request.method == 'POST':
        data = request.get_data()
        data = json.loads(data)
        info_pr_init_with_json(data)
    else:

        job_nb_list = [2, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              33, 0, 0, 0, 0, 0, 0, 0]

        # data preparation
        # data_processing.info_pr_init(data_processing.raw_data_to_compact_data(job_nb_list))

    # 传入数据主要用于处理info pr
    os_list = get_os_list()
    ms = GMT_select()
    info_ma = data_conversion(ms)
    print('info_ma', info_ma)
    bottleneck = get_bottleneck(data_conversion(ms))
    tools = GA_Tools(population_size=10, crossover_rate=0.6, mutation_rate=0.4, elite_number=1,
                     info_ma=info_ma, os_list=os_list, bottleneck=bottleneck)

    # population initialization
    initial_population = tools.population_init()
    fitness = tools.population_decode(initial_population)
    print('initialization done')
    print(fitness)

    # population iteration
    for epoch in range(iteration_limit):
        if epoch == 0:
            new_population = tools.generate_next_raw_generation(initial_population, fitness)
        else:
            new_population = tools.generate_next_raw_generation(new_population, new_fitness)
        tools.population_crossover_mutation(new_population)
        new_fitness = tools.population_decode(new_population)
        # for item in new_population:
        #     print(item)
        best_fit = min(new_fitness)
        print(new_fitness)
        print('best fit epoch' + str(epoch) + ' ' + str(best_fit))
    df = new_population[new_fitness.index(best_fit)].decode_with_response(info_ma=info_ma, os_list=os_list,
                                                                          bottleneck=bottleneck)
    now = datetime.datetime.now()
    print(start_time)
    print("elapsedTime: " + str(now - start_time))
    respond = {'makespan':str(best_fit), 'elapsedTime':str(now - start_time), 'workSchedule':df}
    return json.dumps(respond)





if __name__ == '__main__':
    app.run()


