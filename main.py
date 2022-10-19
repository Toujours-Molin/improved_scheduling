from flask import Flask
from flask import request
import datetime
import json
import numpy as np
import random
import collections
from typing import List
import copy
from copy import deepcopy
import Interface_data_self
import plotly as py
import plotly.figure_factory as ff
import math
from datetime import timedelta
import datetime as dt
import sys

info_pr = {'process_info': [[[[1, 1, 10], [2, 1, 11]], [[3, 1, 12]], [[2, 1, 13]]],
                            [[[3,1,14], [1,1,15], [2,1,16]], [[2,1,17], [3,1,18]], [[3,1,19]]],
                            [[[1,1,1],[2,1,2]]]],
           "job_nb": 3, 'total_op_nb': 7, 'machine_nb': 3,'machine_list':[1,2,3], 'machine_aval':[0,0,0],
           'start_date':"2022-01-01 00:00:00",
           'job_dict':{1:['order 1', 1], 2:['order 2', 1],3:['order 3', 1]},
           'machine_dict':{1:'machine 1',2:'machine 2',3:'machine 3'}
           }

app = Flask(__name__)


# #######################################
# 静态排产部分

def info_pr_init_with_json(data_json):
    global info_pr
    info_pr['machine_nb'] = len(data_json['machines'])
    # machine_dict key 为mechine name，value为从1开始的index
    machine_dict = {}
    for index, machine_name in enumerate(data_json['machines']):
        machine_dict[machine_name] = index + 1
        info_pr['machine_dict'][index+1] = machine_name
    info_pr['machine_list'] = [i for i in range(1, len(data_json['machines']) + 1)]

    job_nb = 0
    for order in data_json['orders']:
        job_nb += order['productNB']
    info_pr['start_date'] = data_json['orders'][0]['dateBegin'] + ' 00:00:00'
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
                    temp = [0, 1, 0]
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


def GMT_select(high_failure_machine: list = None):
    """
    :param high_failure_machine: a list of high failure machine NUMBER
    :return:
    """
    magic_large = 99999999999
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
            if (not high_failure_machine) or num == 1 or selected_machine + 1 not in high_failure_machine:
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


def get_all_pr_list(magic_large: int):
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
        if ocp >= 0.7 * max_ocp:
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
    return bottleneck


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

    def decode(self, info_ma: dict, os_list: list, bottleneck: tuple):
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
                        if interval[1] - pr_time >= earliest and interval[2] >= pr_time:
                            earliest = max(earliest, interval[0])
                            interval_chosen = True
                            break
                if not interval_chosen:
                    index = info_pr['machine_list'].index(machine_chosen)
                    earliest = max(machine_makespan[index], earliest)

                job_info_set.append((job_minus + 1, op, machine_chosen, earliest, earliest + pr_time))

                # find the earliest finish time
                if (not finish_first_record) or (finish_first_record[4] > earliest + pr_time):
                    finish_first_record = (job_minus + 1, op, machine_chosen, earliest, earliest + pr_time)

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
                                                                               interval_info[1] - to_schedule[4]]
                    elif interval_info[1] == to_schedule[4]:
                        machine_interval[to_schedule[2]][interval_position] = [interval_info[0], to_schedule[3],
                                                                               to_schedule[3] - interval_info[0]]
                    else:
                        machine_interval[to_schedule[2]][interval_position:interval_position + 1] = \
                            [[interval_info[0], to_schedule[3], to_schedule[3] - interval_info[0]],
                             [to_schedule[4], interval_info[1], interval_info[1] - to_schedule[4]]]
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

    def decode_with_response(self, info_ma: dict, os_list: list, bottleneck: tuple):
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
                        if interval[1] - pr_time >= earliest and interval[2] >= pr_time:
                            earliest = max(earliest, interval[0])
                            interval_chosen = True
                            break
                if not interval_chosen:
                    index = info_pr['machine_list'].index(machine_chosen)
                    earliest = max(machine_makespan[index], earliest)

                job_info_set.append((job_minus + 1, op, machine_chosen, earliest, earliest + pr_time))

                # find the earliest finish time
                if (not finish_first_record) or (finish_first_record[4] > earliest + pr_time):
                    finish_first_record = (job_minus + 1, op, machine_chosen, earliest, earliest + pr_time)

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

            # update machine working
            # info for every machine [job_nb, start time, end time, op_nb]
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
                                                                               interval_info[1] - to_schedule[4]]
                    elif interval_info[1] == to_schedule[4]:
                        machine_interval[to_schedule[2]][interval_position] = [interval_info[0], to_schedule[3],
                                                                               to_schedule[3] - interval_info[0]]
                    else:
                        machine_interval[to_schedule[2]][interval_position:interval_position + 1] = \
                            [[interval_info[0], to_schedule[3], to_schedule[3] - interval_info[0]],
                             [to_schedule[4], interval_info[1], interval_info[1] - to_schedule[4]]]
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
        slice_chosen = sorted(random.sample(list(range(len(gene_1))), k=2))
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
    result_list_1 = random.sample(index_list, k=int(length / 2))
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
def schedule():
    iteration_limit = 5
    start_time = datetime.datetime.now()
    if request.method == 'POST':
        data = request.get_data()
        data = json.loads(data)
        info_pr_init_with_json(data)
    else:

        job_nb_list = [2, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 43, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0,
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
    respond = {'makespan': str(best_fit), 'elapsedTime': str(now - start_time), 'workSchedule': df}
    return json.dumps(respond)


#  ##################################################
# 静态排产部分结束
# ###############################################
# 预排产部分开始
# ######################################################
@app.route('/preschedule', methods=['GET', 'POST'])
def preschedule():
    if request.method == 'POST':
        data = request.get_data()
        data = json.loads(data)
    else:
        data = Interface_data_self.data_self

    # data = Interface_data_self.data_self

    # 订单拆分函数
    # 某日订单产能校核函数
    def Capacity_Check_Day(Results_days, day_index, Machines_nb, ordersAll, Products, T_product_machine,
                           T_ava_day,
                           capacityFactor):
        T_day = np.zeros(Machines_nb)
        for i in range(len(ordersAll)):
            product_temp = ordersAll[i]['productID']
            product_index = Products.index(product_temp)
            product_nb = Results_days[i][day_index]
            for j in range(Machines_nb):
                T_day[j] += product_nb * T_product_machine[product_index][j]
        flag = True
        for j in range(Machines_nb):
            if T_day[j] > T_ava_day * capacityFactor:
                # print("在该加工能力系数下，第{}周期设备{}加工能力不足，订单前移失败！".format(i+1, machines[j]))
                flag = False
        return flag

    # 计算指定订单未完成量对设备的需求工时，在同一优先级的订单中选择需求工时最少的订单进行前移
    def Orders_left_Capacity_Required_day(Results_left, Orders_list, Machines_nb, ordersAll, Products,
                                          T_product_machine):
        capacity_required = np.zeros((len(Orders_list), Machines_nb))
        capacity_required_sum = np.zeros(len(Orders_list))
        for i in range(len(Orders_list)):
            index = Orders_list[i]
            product_temp = ordersAll[index]['productID']
            product_index = Products.index(product_temp)
            product_nb = Results_left[index]
            for j in range(Machines_nb):
                capacity_required[i][j] += product_nb * T_product_machine[product_index][j]
                capacity_required_sum[i] += capacity_required[i][j]
            if capacity_required_sum[i] == 0:
                capacity_required_sum[i] = float('inf')
        index = capacity_required_sum.argmin()
        return Orders_list[index]

    # 计算指定订单最后一天对设备的需求工时，
    def Capacity_Required_lastday(Results_days, Orders_end_day, Orders_list, Machines_nb, ordersAll,
                                  Products,
                                  T_product_machine):
        capacity_required = np.zeros((len(Orders_list), Machines_nb))
        capacity_required_sum = np.zeros(len(Orders_list))
        for i in range(len(Orders_list)):
            index = Orders_list[i]
            product_temp = ordersAll[index]['productID']
            product_index = Products.index(product_temp)
            product_nb = Results_days[index][Orders_end_day[index] - 1]
            for j in range(Machines_nb):
                capacity_required[i][j] += product_nb * T_product_machine[product_index][j]
                capacity_required_sum[i] += capacity_required[i][j]
            if capacity_required_sum[i] == 0:
                capacity_required_sum[i] = float('inf')
        index = capacity_required_sum.argmin()
        return Orders_list[index]

    # 计算某日内某订单还可增加的数量
    def NB_canPan_day(Results_days, day_index, order_index, ordersAll, Products, T_ava_day, capacity_factor,
                      Machines_nb, T_product_machine):
        machine_time_left = T_ava_day * capacity_factor * np.ones(Machines_nb)
        for i in range(len(Results_days)):
            product_nb = Results_days[i][day_index]
            if product_nb > 0:
                product_temp = ordersAll[i]['productID']
                product_index = Products.index(product_temp)
                for j in range(Machines_nb):
                    machine_time_left[j] -= product_nb * T_product_machine[product_index][j]

        product_toAdd = ordersAll[order_index]['productID']
        product_toAdd_index = Products.index(product_toAdd)
        NB_perMachine = np.zeros(Machines_nb)
        for j in range(Machines_nb):
            if T_product_machine[product_toAdd_index][j] == 0:
                NB_perMachine[j] = float('inf')
            else:
                NB_perMachine[j] = machine_time_left[j] / \
                                   T_product_machine[product_toAdd_index][j]
        NBtoPan = min(NB_perMachine)
        return NBtoPan

    # 预排产及资源齐套性检查函数
    # 对可选设备上的各工序按时间进行排序
    def sort(m_list):
        m_listsj = deepcopy(m_list)
        for i in range(len(m_listsj)):
            for j in range(i, len(m_listsj)):
                if m_listsj[j][0] < m_listsj[i][0]:
                    m_listsj[j][0], m_listsj[i][0] = m_listsj[i][0], m_listsj[j][0]
                    m_listsj[j][1], m_listsj[i][1] = m_listsj[i][1], m_listsj[j][1]

        return m_listsj

    # 获取夹具使用时段的最大重叠数量
    def Overlap_Num(TraySysj_Fixk):  # TraySysj_Fixk: [设备、开始加工时间、结束加工时间]
        if len(TraySysj_Fixk) == 1:
            TraySysj_Fixk_num = 1
        else:
            machine = []
            for i in range(len(TraySysj_Fixk)):
                if TraySysj_Fixk[i][0] in machine:
                    continue
                else:
                    machine.append(TraySysj_Fixk[i][0])
            M_num = len(machine)
            if M_num == 1:
                TraySysj_Fixk_num = 1
            else:
                Fixture_Machine_Tsf = [[] for _ in range(M_num)]
                for i in range(len(TraySysj_Fixk)):
                    for j in range(M_num):
                        if TraySysj_Fixk[i][0] == machine[j]:
                            Fixture_Machine_Tsf[j].append([TraySysj_Fixk[i][1], TraySysj_Fixk[i][2]])
                            break
                Start_T = float('inf')
                Finish_T = float('0')
                SortTsf = [[] for _ in range(M_num)]
                for j in range(M_num):
                    SortTsf[j] = sort(Fixture_Machine_Tsf[j])
                    if SortTsf[j][0][0] < Start_T:
                        Start_T = SortTsf[j][0][0]
                    if SortTsf[j][-1][1] > Finish_T:
                        Finish_T = SortTsf[j][-1][1]
                Start_T = Start_T + 0.5
                num = 0
                while (Start_T < Finish_T):
                    temp_num = 0
                    for j in range(M_num):
                        if (Start_T < SortTsf[j][0][0]) | (Start_T > SortTsf[j][-1][1]):
                            continue
                        for k in range(len(Fixture_Machine_Tsf[j])):
                            if (Start_T > Fixture_Machine_Tsf[j][k][0]) & (Start_T < Fixture_Machine_Tsf[j][k][1]):
                                temp_num = temp_num + 1
                                break
                    if temp_num > num:
                        num = temp_num
                    if num == M_num:
                        break
                    Start_T = Start_T + 3
                TraySysj_Fixk_num = num
        return TraySysj_Fixk_num

    # 在一台设备的多个托盘中选择可开始加工时间最早的
    def Find_Tray(Serial_num, T_sf, idx, idy, v, idz, T, machine, arr_tray):

        Tray_num = arr_tray[0]  # return的设备编号
        tray_lists = [[] for _ in range(len(arr_tray))]  # 各托盘上当前各工序的开始加工时间与结束加工时间
        for i in range(len(arr_tray)):
            for j in range(len(Serial_num)):
                if Serial_num[j][6] == arr_tray[i]:
                    tray_lists[i].append(deepcopy(T_sf[j]))
        for i in range(len(arr_tray)):
            arr_temp = deepcopy(tray_lists[i])
            tray_lists[i] = sort(arr_temp)

        # 判断各设备上空闲时间与该产品该工序是否契合
        # 若idz工序为该订单该产品第一道工序
        BigM = float('inf')
        T_tempa = BigM  # 工序开始时间
        T_lastq = 0  # 该产品上一道工序的结束时间
        if idz == 0:
            for i in range(len(arr_tray)):
                if len(tray_lists[i]) == 0:
                    Tray_num = arr_tray[i]
                    T_tempa = 0
                    break
                else:
                    for l in range(len(tray_lists[i]) + 1):
                        # T_a T_b为托盘空闲时间的开始与结束
                        if l == 0:
                            T_a = 0
                            T_b = tray_lists[i][l][0]
                        elif l == len(tray_lists[i]):
                            T_a = tray_lists[i][-1][1]
                            T_b = BigM
                        else:
                            T_a = tray_lists[i][l - 1][1]
                            T_b = tray_lists[i][l][0]
                        T_temp = max(T_a, T_lastq)
                        if (T_b - T_temp) > (T[idy][idz][machine] + 20):
                            if T_temp < T_tempa:
                                Tray_num = arr_tray[i]
                                T_tempa = T_temp
                                break
        else:
            for i in range(len(Serial_num)):
                if (Serial_num[i][0] == idx) & (Serial_num[i][1] == idy) & (Serial_num[i][2] == v) & (
                        Serial_num[i][3] == idz - 1):
                    T_lastq = T_sf[i][1]  # 该产品上一道工序的结束时间
            # if T_lastq == 0:
            #         print("该产品上一道工序结束时间未初始化")
            #         return -1
            for i in range(len(arr_tray)):
                if len(tray_lists[i]) == 0:
                    Tray_num = arr_tray[i]
                    T_tempa = 0
                    break
                else:
                    for l in range(len(tray_lists[i]) + 1):
                        # T_a T_b为托盘空闲时间的开始与结束
                        if l == 0:
                            T_a = 0
                            T_b = tray_lists[i][l][0]
                        elif l == len(tray_lists[i]):
                            T_a = tray_lists[i][-1][1]
                            T_b = BigM
                        else:
                            T_a = tray_lists[i][l - 1][1]
                            T_b = tray_lists[i][l][0]
                        T_temp = max(T_a, T_lastq)
                        if (T_b - T_temp) > (T[idy][idz][machine] + 20):
                            if T_temp < T_tempa:
                                Tray_num = arr_tray[i]
                                T_tempa = T_temp
                                break
        T_tempa = T_tempa + 10
        T_tempb = T_tempa + T[idy][idz][machine]
        result = [Tray_num, T_tempa, T_tempb]
        return result
        # return [m_num,T_tempa,T_tempb]

    # 在多台可选设备中选择可开始加工时间最早的
    def Func0(Serial_num, T_sf, idx, idy, v, idz, T, arr_m):
        m_ava = arr_m[0]  # m_ava可选加工设备_设备编号
        m_lists = [[] for _ in range(len(m_ava))]  # 各设备上当前工序的开始加工时间与结束加工时间
        for j in range(len(m_ava)):
            for i in range(len(Serial_num)):
                if Serial_num[i][4] == m_ava[j]:
                    m_lists[j].append(deepcopy(T_sf[i]))  # m_lists 三维列表 1可用设备编号 2该设备上已安排工序编号 3工序开始时间与结束时间

        for j in range(len(m_ava)):
            arr_temp = deepcopy(m_lists[j])
            m_lists[j] = sort(arr_temp)

        # 判断各设备上空闲时间与该产品该工序是否契合
        # 若idz工序为该订单该产品第一道工序
        m_num = -1  # return的设备编号
        BigM = float('inf')
        T_tempa = BigM  # 工序开始时间
        T_lastq = 0  # 该产品上一道工序的结束时间
        if idz == 0:
            for j in range(len(m_ava)):
                if len(m_lists[j]) == 0:
                    m_num = m_ava[j]
                    break
                else:
                    for l in range(len(m_lists[j]) + 1):
                        # T_a T_b为空闲时间的开始与结束
                        if l == 0:
                            T_a = 0
                            T_b = m_lists[j][l][0]
                        elif l == len(m_lists[j]):
                            T_a = m_lists[j][-1][1]
                            T_b = BigM
                        else:
                            T_a = m_lists[j][l - 1][1]
                            T_b = m_lists[j][l][0]
                        T_temp = max(T_a, T_lastq)
                        if (T_b - T_temp) > T[idy][idz][m_ava[j]] + 20:  # 20:idz=0，首道工序必定进行上下料
                            if T_temp < T_tempa:
                                m_num = m_ava[j]
                                T_tempa = T_temp
                                break

        else:
            for i in range(len(Serial_num)):
                if (Serial_num[i][0] == idx) & (Serial_num[i][1] == idy) & (Serial_num[i][2] == v) & (
                        Serial_num[i][3] == idz - 1):
                    T_lastq = T_sf[i][1]  # 该产品上一道工序的结束时间
            # if T_lastq == 0:
            #         print("该产品上一道工序结束时间未初始化")
            #         return -1
            for j in range(len(m_ava)):
                if len(m_lists[j]) == 0:
                    m_num = m_ava[j]
                    break
                else:
                    for l in range(len(m_lists[j]) + 1):
                        # T_a T_b为空闲时间的开始与结束
                        if l == 0:
                            T_a = 0
                            T_b = m_lists[j][l][0]
                        elif l == len(m_lists[j]):
                            T_a = m_lists[j][-1][1]
                            T_b = BigM
                        else:
                            T_a = m_lists[j][l - 1][1]
                            T_b = m_lists[j][l][0]
                        T_temp = max(T_a, T_lastq)
                        if (T_b - T_temp) > T[idy][idz][m_ava[j]] + 20:  # 20:暂定
                            if T_temp < T_tempa:
                                m_num = m_ava[j]
                                T_tempa = T_temp
                                break

        T_tempb = T_tempa + T[idy][idz][m_num]
        return m_num
        # return [m_num,T_tempa,T_tempb]

    def Func0_1(Serial_num, T_sf, idx, idy, v, idz, T, arr_m, ResCheck_Machine_TraySystem,
                ResCheck_TraySys_TrayIndex_Fixture, ResCheck_Item_Fixture):
        m_ava = arr_m[0]  # m_ava可选加工设备_设备编号
        result_m = -1
        result_tray = -1
        result_traysys = -1
        result_figure = -1
        T_lastq = 0
        lastm_num = -1
        T_tempa = T_lastq
        BigM = float('inf')
        result_T_a = BigM
        for w in range(len(Serial_num)):
            if (Serial_num[w][0] == idx) & (Serial_num[w][1] == idy) & (Serial_num[w][2] == v) & (
                    Serial_num[w][3] == idz - 1):
                T_lastq = T_sf[w][1]  # 该产品上一道工序的结束时间
                lastm_num = Serial_num[w][4]
        # 遍历每一台可用设备
        for u in range(len(m_ava)):
            m_num = m_ava[u]
            m_list = []
            TraySys_num = -1
            Fixture_num = -1

            for ud in range(len(Serial_num)):
                if Serial_num[ud][4] == m_num:
                    m_list.append(T_sf[ud])  # m_list 二维列表  1该设备上已安排工序编号 2工序开始时间与结束时间
            for ua in range(len(ResCheck_Machine_TraySystem)):
                if ResCheck_Machine_TraySystem[ua][0] == m_num:
                    TraySys_num = ResCheck_Machine_TraySystem[ua][1]
                    break
            for ub in range(len(ResCheck_Item_Fixture)):
                if (ResCheck_Item_Fixture[ub][0] == idy) and (ResCheck_Item_Fixture[ub][1] == idz):
                    Fixture_num = ResCheck_Item_Fixture[ub][2]
                    break
            arr_tray = []
            for uc in range(len(ResCheck_TraySys_TrayIndex_Fixture)):
                if (ResCheck_TraySys_TrayIndex_Fixture[uc][0] == TraySys_num) & (
                        ResCheck_TraySys_TrayIndex_Fixture[uc][2] == Fixture_num):
                    Tray_num = ResCheck_TraySys_TrayIndex_Fixture[uc][1]
                    arr_tray.append(Tray_num)
            # 遍历可用设备中的每一个可用托盘
            for vv in range(len(arr_tray)):
                tray_num = arr_tray[vv]
                tray_list = []
                T_tempa = T_lastq
                for va in range(len(Serial_num)):
                    if Serial_num[va][6] == tray_num:
                        tray_list.append(T_sf[va])  # tray_list: 二维列表  1 托盘上执行工序  2工序开始时间与结束时间

                if (len(m_list) == 0) & (len(tray_list) == 0):
                    T_tempa = T_lastq
                    result_T_a = T_tempa
                    result_m = m_num
                    result_tray = tray_num
                    result_traysys = TraySys_num
                    result_figure = Fixture_num
                    break
                else:
                    m_list = sort(m_list)
                    tray_list = sort(tray_list)
                    T_total = T[idy][idz][m_num] + 20

                    for i in range(len(m_list) + 1):
                        if len(m_list) == 0:
                            temp_m_a = 0
                            temp_m_b = BigM
                        elif i == 0:
                            temp_m_a = 0
                            temp_m_b = m_list[i][0]
                        elif i == len(m_list):
                            temp_m_a = m_list[-1][1]
                            temp_m_b = BigM
                        else:
                            temp_m_a = m_list[i - 1][1]
                            temp_m_b = m_list[i][0]

                        for j in range(len(tray_list) + 1):
                            flag1 = 0
                            if len(tray_list) == 0:
                                temp_t_a = 0
                                temp_t_b = BigM
                            elif j == 0:
                                temp_t_a = 0
                                temp_t_b = tray_list[j][0]
                            elif j == len(tray_list):
                                temp_t_a = tray_list[-1][1]
                                temp_t_b = BigM
                            else:
                                temp_t_a = tray_list[j - 1][1]
                                temp_t_b = tray_list[j][0]
                            if min(temp_m_b, temp_t_b) - max(temp_m_a, temp_t_a, T_lastq) > T_total:
                                T_tempa = max(temp_m_a, temp_t_a, T_lastq)
                                flag1 = 1
                                break
                        if flag1 > 0:
                            break  # 此时u设备v托盘组合的最早加工时间即为T_tempa
                    if (T_tempa < result_T_a):
                        result_T_a = T_tempa
                        result_m = m_num
                        result_tray = tray_num
                        result_traysys = TraySys_num
                        result_figure = Fixture_num
                if (result_T_a == T_lastq):
                    break

            if (result_T_a == T_lastq):
                break

        if (lastm_num == result_m) & (result_T_a == T_lastq):
            result_T_a = result_T_a
        else:
            result_T_a = result_T_a + 10  # 上下料时间10min
        result_T_b = result_T_a + T[idy][idz][result_m]
        return [result_m, result_traysys, result_tray, result_figure, result_T_a, result_T_b]

    # 确定[订单、产品、产品编号、工序号、设备号、托盘系统编号、托盘编号、夹具类型编号]
    def Func1(Serial_num, T_sf, idx, idy, v, idz, T, ResCheck_Machine_TraySystem, ResCheck_TraySys_TrayIndex_Fixture,
              ResCheck_Item_Fixture):

        arr_m = np.nonzero(T[idy][idz])
        result = Func0_1(Serial_num, T_sf, idx, idy, v, idz, T, arr_m, ResCheck_Machine_TraySystem,
                         ResCheck_TraySys_TrayIndex_Fixture, ResCheck_Item_Fixture)
        m_num = result[0]
        TraySys_num = result[1]
        Tray_num = result[2]
        Fixture_num = result[3]
        T_a = result[4]
        T_b = result[5]
        return [idx, idy, v, idz, m_num, TraySys_num, Tray_num, Fixture_num, T_a, T_b]

    # 确定[某一产品某一工序开始加工时间、结束加工时间]
    def Func2(Serial_num, T_sf, idx, idy, v, idz, T):
        m_num = Serial_num[-1][4]
        m_list = []
        tray_num = Serial_num[-1][6]
        tray_list = []
        T_lastq = 0
        T_tempa = T_lastq
        lastm_num = -1
        BigM = float('inf')
        for i in range(len(Serial_num)):
            if (Serial_num[i][0] == idx) & (Serial_num[i][1] == idy) & (Serial_num[i][2] == v) & (
                    Serial_num[i][3] == idz - 1):
                T_lastq = T_sf[i][1]  # 该产品上一道工序的结束时间
                lastm_num = Serial_num[i][4]
        for i in range(len(Serial_num) - 1):
            if Serial_num[i][4] == m_num:
                m_list.append(T_sf[i])  # m_list 二维列表  1该设备上已安排工序编号 2工序开始时间与结束时间
            if Serial_num[i][6] == tray_num:
                tray_list.append(T_sf[i])  # tray_list: 二维列表  1 托盘上执行工序  2工序开始时间与结束时间

        if len(m_list) == 0 & len(tray_list) == 0:
            T_tempa = T_lastq
        else:
            m_list = sort(m_list)
            tray_list = sort(tray_list)
            T_total = T[idy][idz][m_num] + 20

            for i in range(len(m_list) + 1):
                if len(m_list) == 0:
                    temp_m_a = 0
                    temp_m_b = BigM
                elif i == 0:
                    temp_m_a = 0
                    temp_m_b = m_list[i][0]
                elif i == len(m_list):
                    temp_m_a = m_list[-1][1]
                    temp_m_b = BigM
                else:
                    temp_m_a = m_list[i - 1][1]
                    temp_m_b = m_list[i][0]

                for j in range(len(tray_list) + 1):
                    if len(tray_list) == 0:
                        temp_t_a = 0
                        temp_t_b = BigM
                    elif j == 0:
                        temp_t_a = 0
                        temp_t_b = tray_list[j][0]
                    elif j == len(tray_list):
                        temp_t_a = tray_list[-1][1]
                        temp_t_b = BigM
                    else:
                        temp_t_a = tray_list[j - 1][1]
                        temp_t_b = tray_list[j][0]
                    if min(temp_m_b, temp_t_b) - max(temp_m_a, temp_t_a, T_lastq) > T_total:
                        T_tempa = max(temp_m_a, temp_t_a, T_lastq)
                        break
                if T_tempa > 0:
                    break

        if (lastm_num == m_num) & (T_tempa == T_lastq):
            T_tempa = T_tempa
        else:
            T_tempa = T_tempa + 10  # 上下料时间10min
        T_tempb = T_tempa + T[idy][idz][m_num]
        return [T_tempa, T_tempb]

    # 确定[设备、托盘系统、采用夹具、开始使用时间、结束使用时间]
    def Func3(Serial_num, T_sf, ResCheck_Item_Fixture, ResCheck_Machine_TraySystem):
        Fixture_Tsf = []
        for i in range(len(Serial_num)):
            for j in range(len(ResCheck_Item_Fixture)):
                if (Serial_num[i][1] == ResCheck_Item_Fixture[j][0]) and (
                        Serial_num[i][3] == ResCheck_Item_Fixture[j][1]):
                    for k in range(len(ResCheck_Machine_TraySystem)):
                        if Serial_num[i][4] == ResCheck_Machine_TraySystem[k][0]:
                            Fixture_Tsf.append(
                                [Serial_num[i][4], ResCheck_Machine_TraySystem[k][1], ResCheck_Item_Fixture[j][2],
                                 T_sf[i][0], T_sf[i][1]])
                            break
                    break
        return Fixture_Tsf

    # 确定各托盘系统对各类夹具的最大需求量
    def Func4(Fixture_Tsf, TraySystem_Num, Fixture_Num):
        Result_TraySys_Fixture = np.zeros((TraySystem_Num, Fixture_Num))  # 返回值：各托盘系统对各类夹具的最大需求量
        TraySys_Fixture = [[] for _ in range(TraySystem_Num)]
        for i in range(len(Fixture_Tsf)):
            for j in range(TraySystem_Num):
                if Fixture_Tsf[i][1] == j:
                    # [夹具类别、设备、开始加工时间、结束加工时间]
                    TraySys_Fixture[j].append(
                        [Fixture_Tsf[i][2], Fixture_Tsf[i][0], Fixture_Tsf[i][3], Fixture_Tsf[i][4]])
                    break
        for j in range(TraySystem_Num):
            if len(TraySys_Fixture[j]) == 0:
                continue
            TraySysj_Fixture = [[] for _ in range(Fixture_Num)]
            for u in range(len(TraySys_Fixture[j])):
                TraySysj_Fixture[TraySys_Fixture[j][u][0]].append(
                    [TraySys_Fixture[j][u][1], TraySys_Fixture[j][u][2], TraySys_Fixture[j][u][3]])
            for k in range(Fixture_Num):
                if len(TraySysj_Fixture[k]) == 0:
                    continue
                Result_TraySys_Fixture[j][k] = Overlap_Num(TraySysj_Fixture[k])

        return Result_TraySys_Fixture

    # 确定各设备上各类刀具的需求量
    def Func5(Serial_num, Tool_Time, M, Tool_Num):
        Tool_time_require = np.zeros((M, Tool_Num))
        for i in range(len(Serial_num)):
            item = Serial_num[i][1]
            process = Serial_num[i][3]
            machine = Serial_num[i][4]
            for j in range(Tool_Num):
                Tool_time_require[machine][j] = Tool_Time[item][process][machine][j] + Tool_time_require[machine][j]

        return Tool_time_require

    # 确定各类外协件的需求量
    def Func6(Serial_num, Outpart_Num, Outpart_Type):
        Outpart_num_require = np.zeros((Outpart_Type))
        for i in range(len(Serial_num)):
            item = Serial_num[i][1]
            process = Serial_num[i][3]
            for j in range(Outpart_Type):
                Outpart_num_require[j] = Outpart_Num[item][process][j] + Outpart_num_require[j]

        return Outpart_num_require

    # 确定各周期各班次需要人员操作的设备
    def Func7(Serial_num, T_sf, T_shift, Num_shift, M):
        Worker_machine_condition = np.zeros((Num_shift, M))
        for i in range(len(Serial_num)):
            machine = Serial_num[i][4]
            Shift_start = int(T_sf[i][0] / T_shift)
            Shift_end = math.ceil(T_sf[i][1] / T_shift)
            Shift_end = min(Shift_end, Num_shift)
            for j in range(Shift_start, Shift_end):
                Worker_machine_condition[j][machine] = 1

        return Worker_machine_condition

    def GANTT(Serial_num, T_sf, M, T_begin, w):  # T
        m_lists = [[] for _ in range(M)]  # 各设备上工序的开始加工时间与结束加工时间
        Job_lists = [[] for _ in range(M)]  # 记录m_lists各时间段对应的加工产品
        for j in range(M):
            for i in range(len(Serial_num)):
                if Serial_num[i][4] == j:
                    m_lists[j].append(T_sf[i])
                    Job_lists[j].append(Serial_num[i])
        pyplt = py.offline.plot
        df = []
        colors = {}
        # 设定甘特图颜色参数
        temp0 = []
        temp1 = []
        temp2 = []
        for i in range(len(Serial_num)):
            temp0.append(Serial_num[i][0])
            temp1.append(Serial_num[i][1])
            temp2.append(Serial_num[i][2])
        color0 = int(250 / (max(temp0) + 1))
        color1 = int(250 / (max(temp1) + 1))
        color2 = int(250 / (max(temp2) + 1))

        # 画甘特图
        for j in range(M):
            for i in range(len(m_lists[j])):
                a = '设备' + str(j + 1)
                b = T_begin + timedelta(minutes=m_lists[j][i][0])
                c = T_begin + timedelta(minutes=m_lists[j][i][1])
                r = 'Order' + str(Job_lists[j][i][0] + 1) + ' Job' + str(Job_lists[j][i][1] + 1) + ' Num' + str(
                    Job_lists[j][i][2] + 1)
                d = dict(Task=a, Start=b, Finish=c, Resource=r)
                df.append(d)
                colors[str(r)] = 'rgb(' + str(color1 * (Job_lists[j][i][1] + 1)) + ',' + str(
                    color2 * (Job_lists[j][i][2] + 1)) \
                                 + ',' + str(color0 * (Job_lists[j][i][0] + 1)) + ')'

        # print(df)
        # fig = ff.create_gantt(df,showgrid_x=True, show_colorbar=True, group_tasks=True)
        fig = ff.create_gantt(df, colors=colors, index_col='Resource', showgrid_x=True, show_colorbar=True,
                              group_tasks=True)
        # w=input("请输入要保存的甘特图路径：")
        # w='E:/R4_C2_1.html'
        pyplt(fig, filename=w)

    def GANTT_ResCheck_Fixture(Fixture_Tsf, M, T_begin, w):
        m_lists = [[] for _ in range(M)]  # 各设备上的设备编号、托盘系统编号、采用夹具、工序开始加工时间、结束加工时间
        for j in range(M):
            for i in range(len(Fixture_Tsf)):
                if Fixture_Tsf[i][0] == j:
                    m_lists[j].append(Fixture_Tsf[i])
        pyplt = py.offline.plot
        df = []
        colors = {}
        # 设定甘特图颜色参数
        temp0 = []
        temp1 = []
        for i in range(len(Fixture_Tsf)):
            temp0.append(Fixture_Tsf[i][1])
            temp1.append(Fixture_Tsf[i][2])

        color0 = int(250 / (max(temp0) + 1))
        color1 = int(250 / (max(temp1) + 1))
        # 画甘特图
        for j in range(M):
            for i in range(len(m_lists[j])):
                a = 'Machine' + str(j + 1)
                b = T_begin + timedelta(minutes=m_lists[j][i][3])
                c = T_begin + timedelta(minutes=m_lists[j][i][4])
                r = 'TraySys' + str(m_lists[j][i][1] + 1) + 'Fixture' + str(m_lists[j][i][2] + 1)
                d = dict(Task=a, Start=b, Finish=c, Resource=r)
                df.append(d)
                colors[str(r)] = 'rgb(' + str(color0 * (m_lists[j][i][1] + 1)) + ',' + str(
                    color1 * (m_lists[j][i][2] + 1)) \
                                 + ',' + str(color0 * (m_lists[j][i][1] + 1)) + ')'
        fig = ff.create_gantt(df, colors=colors, index_col='Resource', showgrid_x=True, show_colorbar=True,
                              group_tasks=True)
        pyplt(fig, filename=w)

    # 数据预处理
    # 订单信息
    ordersAll = data['ordersAll']

    # 工艺信息
    process = data['process']

    # 设备信息
    machines = data['machines']

    # 刀具库信息
    machineTools = data['machineTools']

    # 托盘站信息
    traySystems = data['traySystems']

    # 托盘站设备信息
    traySystemMachine = data['traySystemMachine']

    # 各类夹具信息
    fixtures = data['fixtures']

    # 产品毛坯信息
    blanks = data['blanks']

    # 毛坯库存信息
    blankInventory = data['blankInventory']

    # 外协件库存信息
    cooperationPartInventory = data['cooperationPartInventory']

    # 排产参数
    periodNB = data['periodNB']
    periodLength = data['periodLength']
    hours = data['hours']
    T_ava = periodLength * hours * 60  # 可用工时(min)
    T_ava_day = hours * 60
    capacityFactor = data['capacityFactor']
    shiftNB = data['shiftNB']
    dayNB = periodNB * periodLength  # 排产天数
    T_begin = data['dateStart'] #排产开始日期
    T_begin = dt.datetime.strptime(T_begin, '%Y-%m-%d')

    # 订单拆分程序
    Machines_nb = len(machines)  # 设备数量

    # 订单信息处理
    Orders = []  # 包含全部订单id
    Orders_nb = len(ordersAll)  # 订单数量
    Orders_level = np.zeros(Orders_nb)  # 各订单优先级
    Orders_begin = np.zeros(Orders_nb, dtype=int)  # 订单要求开始周期
    Orders_end = np.zeros(Orders_nb, dtype=int)  # 订单要求结束周期
    Orders_begin_day = np.zeros(Orders_nb, dtype=int)  # 订单要求开始日期是所有周期（4*7）的第几天
    Orders_end_day = np.zeros(Orders_nb, dtype=int)  # 订单要求结束日期是所有周期（4*7）的第几天
    Last_days_period = np.zeros((Orders_nb, periodNB), dtype=int)  # 各订单在各周期的持续天数，作为订单在各周期排产数量的上限依据

    for i in range(Orders_nb):
        Orders.append(ordersAll[i]['orderID'])
        Orders_level[i] = ordersAll[i]['orderLevel']
        o_begin = ordersAll[i]['dateBegin']
        o_end = ordersAll[i]['dateEnd']
        o_begin = dt.datetime.strptime(o_begin, "%Y-%m-%d")
        o_end = dt.datetime.strptime(o_end, "%Y-%m-%d")
        days_b = max(0, (o_begin - T_begin).days) + 1
        days_e = max(0, (o_end - T_begin).days) + 1
        Orders_begin_day[i] = days_b
        Orders_end_day[i] = days_e
        for j in range(periodNB):
            days_1 = max(Orders_begin_day[i] - periodLength * j - 1, 0)
            days_2 = max(periodLength * (j + 1) - Orders_end_day[i], 0)
            Last_days_period[i][j] = max(periodLength - days_1 - days_2, 0)

        productionBool = np.nonzero(Last_days_period[i])
        Orders_begin[i] = productionBool[0][0]
        Orders_end[i] = productionBool[0][-1]

        # Orders_begin[i] = round(days_b / periodLength)
        # Orders_end[i] = round(days_e / periodLength)

    # 产品信息获取
    Products = []
    for i in process.keys():
        Products.append(i)
    Products_nb = len(Products)

    # 工艺信息处理
    T_product_machine = np.zeros((Products_nb, Machines_nb))
    for i in range(Products_nb):
        product_temp = Products[i]
        process_temp = process[product_temp]
        process_nb = len(process_temp)
        for j in range(process_nb):
            machine_ava = process_temp[j]['machineID']
            machine_time = process_temp[j]['processTime']
            for k in range(len(machine_ava)):
                machine_index = machines.index(machine_ava[k])
                T_product_machine[i][machine_index] += machine_time[k] / len(machine_ava)

    # 拆分结果数组
    Results = np.zeros((Orders_nb, periodNB))
    Results_days = np.zeros((Orders_nb, dayNB))  # 每日拆分订单数量
    Results_left = np.zeros(Orders_nb)  # 无法完成的订单数量

    orders_toSplit = deepcopy(Orders)
    orders_toSplit_level = deepcopy(Orders_level)

    # 订单初始拆分，同时考虑每日产能校核
    while (len(orders_toSplit) > 0):
        order_index = orders_toSplit_level.argmax()
        if ordersAll[order_index]['productNB'] <= 5:
            flag_1 = False  # 判断该订单是否分配到某一天
            for d in range(Orders_begin_day[order_index], Orders_end_day[order_index] + 1):
                Results_days_temp = deepcopy(Results_days)
                Results_days_temp[order_index][d - 1] += ordersAll[order_index]['productNB']
                if Capacity_Check_Day(Results_days_temp, d - 1, Machines_nb, ordersAll, Products,
                                      T_product_machine, T_ava_day, capacityFactor):
                    Results_days[order_index][d - 1] += ordersAll[order_index]['productNB']
                    flag_1 = True
                    break
            if not flag_1:
                Results_left[order_index] += ordersAll[order_index]['productNB']
        else:
            for d in range(Orders_begin_day[order_index], Orders_end_day[order_index] + 1):
                Results_days_temp = deepcopy(Results_days)
                NB_temp = ordersAll[order_index]['productNB'] / (
                        Orders_end_day[order_index] - Orders_begin_day[order_index] + 1)
                Results_days_temp[order_index][d - 1] += NB_temp
                if Capacity_Check_Day(Results_days_temp, d - 1, Machines_nb, ordersAll, Products,
                                      T_product_machine, T_ava_day, capacityFactor):
                    Results_days[order_index][d - 1] += NB_temp
                else:
                    Results_left[order_index] += NB_temp
        orders_toSplit.remove(ordersAll[order_index]['orderID'])
        orders_toSplit_level[order_index] = -1
    # print(Results_days)
    # 无法完成的订单数量  前移
    orders_left_toPan = deepcopy(Orders)
    orders_left_toPan_level = deepcopy(Orders_level)

    while (len(orders_left_toPan) > 0):
        Orders_list = np.where(orders_left_toPan_level == max(orders_left_toPan_level))[0]
        order_index = Orders_left_Capacity_Required_day(Results_left, Orders_list, Machines_nb, ordersAll,
                                                        Products, T_product_machine)
        # order_index = orders_left_toPan_level.argmax()
        if Results_left[order_index] > 0:
            for d in range(Orders_begin_day[order_index], Orders_end_day[order_index] + 1):
                # NB_toPan = NB_canPan_day(d-1)  # 返回：可以某订单该日的最大空余数量
                NB_toPan = NB_canPan_day(Results_days, d - 1, order_index, ordersAll, Products,
                                         T_ava_day, capacityFactor, Machines_nb, T_product_machine)
                if NB_toPan >= Results_left[order_index]:
                    Results_days[order_index][d - 1] += Results_left[order_index]
                    Results_left[order_index] = 0
                    break
                else:
                    Results_days[order_index][d - 1] += NB_toPan
                    Results_left[order_index] -= NB_toPan

        orders_left_toPan.remove(ordersAll[order_index]['orderID'])
        orders_left_toPan_level[order_index] = -1
    # print(Results_days)
    # 靠后的订单数量  前移
    orders_toPan = deepcopy(Orders)
    orders_toPan_level = deepcopy(Orders_level)

    while (len(orders_toPan) > 0):
        Orders_list = np.where(orders_toPan_level == max(orders_toPan_level))[0]
        order_index = Capacity_Required_lastday(Results_days, Orders_end_day, Orders_list, Machines_nb,
                                                ordersAll, Products, T_product_machine)
        if Results_left[order_index] == 0:
            flag_2 = False
            for d1 in range(Orders_begin_day[order_index], Orders_end_day[order_index] + 1):
                for d2 in range(Orders_end_day[order_index] + 1 - Orders_begin_day[order_index]):
                    # NB_toPan = NB_canPan_day(d1-1,)  # 返回：可以某订单该日的最大空余数量
                    NB_toPan = NB_canPan_day(Results_days, d1 - 1, order_index, ordersAll, Products,
                                             T_ava_day, capacityFactor, Machines_nb,
                                             T_product_machine)
                    day_index = Orders_end_day[order_index] - 1 - d2
                    if day_index <= d1:
                        flag_2 = True
                        break
                    if NB_toPan >= Results_days[order_index][day_index]:
                        Results_days[order_index][d1 - 1] += Results_days[order_index][
                            day_index]
                        Results_days[order_index][day_index] = 0
                    else:
                        Results_days[order_index][d1 - 1] += NB_toPan
                        Results_days[order_index][day_index] -= NB_toPan
                        break
                if flag_2:
                    break

        orders_toPan.remove(ordersAll[order_index]['orderID'])
        orders_toPan_level[order_index] = -1
    # print(Results_days)
    for i in range(dayNB):
        period_index = int(i / periodLength)
        for k in range(Orders_nb):
            Results[k][period_index] += Results_days[k][i]

    for k in range(Orders_nb):
        for r in range(periodNB):
            Results[k][r] = round(Results[k][r])
        Results_left[k] = round(Results_left[k])
        if np.sum(Results[k]) + Results_left[k] != ordersAll[k]['productNB']:
            if Results_left[k] > 0:
                Results_left[k] = ordersAll[k]['productNB'] - np.sum(Results[k])
            else:
                for rr in range(periodNB):
                    period_index = periodNB - 1 - rr
                    if Results[k][period_index] > 0:
                        Results[k][period_index] = ordersAll[k]['productNB'] - np.sum(
                            Results[k][0:period_index])
                        break

    OEE_period_machine = np.zeros((periodNB, Machines_nb))
    for i in range(periodNB):
        time_consumed = np.zeros(Machines_nb)
        for k in range(Orders_nb):
            product_temp = ordersAll[k]['productID']
            product_index = Products.index(product_temp)
            product_nb = Results[k][i]
            for j in range(Machines_nb):
                time_consumed[j] += product_nb * T_product_machine[product_index][j]
        for j in range(Machines_nb):
            OEE_period_machine[i][j] = time_consumed[j] / T_ava

    ordersSplit = []  # 拆分后订单信息
    for i in range(periodNB):
        for j in range(Orders_nb):
            order_detail = {}
            order_detail['periodID'] = i + 1
            order_detail['orderID'] = ordersAll[j]['orderID']
            order_detail['productID'] = ordersAll[j]['productID']
            order_detail['productNB'] = Results[j][i]
            ordersSplit.append(order_detail)

    ordersDelay = []  # 需延期或取消 订单信息
    for i in range(Orders_nb):
        detail = {}
        detail['orderID'] = ordersAll[i]['orderID']
        detail['productID'] = ordersAll[i]['productID']
        detail['delayNB'] = Results_left[i]
        ordersDelay.append(detail)

    # 预排产及资源齐套性检查程序
    # 排产参数获取
    Orders = []
    for i in range(len(ordersAll)):
        Orders.append(ordersAll[i]['orderID'])

    Products = []  # 产品信息
    Q_max = 0
    for i in process.keys():
        Products.append(i)
        if len(process[i]) > Q_max:
            Q_max = len(process[i])

    Products_nb = len(Products)
    N = Products_nb  # 产品数量

    Machines_nb = len(machines)  # 设备数量
    M = Machines_nb  # 设备数量

    T = np.zeros((N, Q_max, M))  # i产品的q工序在j设备上的加工时间
    Q = np.zeros(N, dtype=int)  # 各产品所需工序数

    K = len(ordersAll)  # 订单数量
    L = periodNB  # 周期个数
    T_cycleday = periodLength  # 排产周期天数
    T_hpd = hours  # 每日有效工作时长
    T_avr = T_ava  # 每周期可用工时数（min）
    T_shift = T_hpd * 60 / shiftNB  # 每班次时间（min）
    Num_shift = T_cycleday * shiftNB  # 每周期班次个数
    LargeM = float('inf')  # 极大值

    T_begin = data['dateStart'] + ' 0:0:0' #排产开始时间
    T_begin = dt.datetime.strptime(T_begin, '%Y-%m-%d %H:%M:%S')

    for i in range(Products_nb):
        item = Products[i]
        Q[i] = len(process[item])
        for q in range(Q[i]):
            machine = process[item][q]['machineID']
            time = process[item][q]['processTime']
            for j in range(len(machine)):
                machine_index = machines.index(machine[j])
                T[i][q][machine_index] = time[j]

    # 排产规则：若某产品存在在同一设备上进行的相邻工序，则合并进行
    Q_continuous = [[] for _ in range(N)]  # 各产品可以进行连续加工的工序集合
    Start_continuous = np.zeros(N, dtype=int)
    Max_continuous = np.ones(N, dtype=int)

    # 获取各产品可以进行连续加工的工序集合
    for i in range(N):
        temp_max = 1
        if Q[i] == 1:
            continue
        for q in range(Q[i]):
            if q == 0:
                temp_w0 = np.where(T[i][q] > 0)
                temp_w1 = np.where(T[i][q + 1] > 0)
                if temp_w0[0][0] == temp_w1[0][0]:
                    Start_continuous[i] = q
            elif q == Q[i] - 1:
                temp_w_1 = np.where(T[i][q - 1] > 0)
                temp_w0 = np.where(T[i][q] > 0)
                # temp_w1 = np.where(T[i][q + 1] > 0)
                if temp_w_1[0][0] == temp_w0[0][0]:
                    temp_max = q - Start_continuous[i] + 1
                    temp_q = []
                    for w in range(temp_max):
                        temp_q.append(Start_continuous[i] + w)
                    Q_continuous[i].append(temp_q)

            else:
                temp_w_1 = np.where(T[i][q - 1] > 0)
                temp_w0 = np.where(T[i][q] > 0)
                temp_w1 = np.where(T[i][q + 1] > 0)
                if (temp_w_1[0][0] != temp_w0[0][0]) & (temp_w0[0][0] == temp_w1[0][0]):
                    Start_continuous[i] = q
                elif (temp_w_1[0][0] == temp_w0[0][0]) & (temp_w0[0][0] != temp_w1[0][0]):
                    temp_max = q - Start_continuous[i] + 1
                    temp_q = []
                    for w in range(temp_max):
                        temp_q.append(Start_continuous[i] + w)
                    Q_continuous[i].append(temp_q)

    # 参数获取_设备-托盘系统+产品-工序-夹具
    TraySystem_Num = len(traySystems)  # 托盘系统个数
    Fixture_Num = len(fixtures)  # 夹具类别数量

    fixtures_list = []
    for i in fixtures.keys():
        fixtures_list.append(i)

    traySystems_list = []
    for i in traySystems.keys():
        traySystems_list.append(i)

    ResCheck_Item_Fixture = []
    ResCheck_Machine_TraySystem = []
    ResCheck_TraySys_TrayIndex_Fixture = []

    # 产品-工序-夹具对应关系列表
    for i in range(Products_nb):
        item = Products[i]
        Q[i] = len(process[item])
        for q in range(Q[i]):
            fixture_type_id = process[item][q]['fixtureTypeID']
            fixture_type_index = fixtures_list.index(fixture_type_id)
            ResCheck_Item_Fixture.append([i, q, fixture_type_index])

    # 设备-托盘站对应关系列表
    for i in range(Machines_nb):
        for j in traySystemMachine.keys():
            if machines[i] in traySystemMachine[j]['machineID']:
                traySystem_index = traySystems_list.index(j)
                ResCheck_Machine_TraySystem.append([i, traySystem_index])

    # 托盘站-托盘-夹具对应关系列表
    for i in range(len(traySystems_list)):
        traySystem = traySystems_list[i]
        for j in range(len(traySystems[traySystem])):
            if traySystems[traySystem][j]['isOccupied']:
                tray_id = traySystems[traySystem][j]['trayID']
                fixture_type_id = traySystems[traySystem][j]['fixtureTypeID']
                fixture_type_index = fixtures_list.index(fixture_type_id)
                ResCheck_TraySys_TrayIndex_Fixture.append([i, tray_id, fixture_type_index])

    # 资源齐套性检查信息获取
    # 毛坯参数获取
    blanks_list = []  # 记录毛坯类别字符的列表
    for i in range(len(blanks)):
        blanks_list.append(blanks[i]['blankTypeID'])
    Blank_N = Products_nb  # 产品数量
    Blank_class_max = len(blanks_list)  # 毛坯类别总数

    Blank = np.zeros((Blank_N, Blank_class_max))  # 某产品对某类别毛坯的需求数量
    for i in range(len(blanks)):
        product_index = Products.index(blanks[i]['productID'])
        blank_index = blanks_list.index(blanks[i]['blankTypeID'])
        Blank[product_index][blank_index] = blanks[i]['blankNB']

    # 毛坯库存参数获取
    Blank_inventory = np.zeros((1, Blank_class_max))
    for i in range(Blank_class_max):
        Blank_inventory[0][i] = blankInventory[blanks_list[i]]

    Blank_require = np.zeros((L, Blank_class_max))
    blankCheck = []  # 记录毛坯检查结果，作为输出

    # 资源齐套性检查参数获取_刀具
    tools_list = []
    for i in machineTools.keys():
        for j in range(len(machineTools[i])):
            if machineTools[i][j]['toolTypeID'] not in tools_list:
                tools_list.append(machineTools[i][j]['toolTypeID'])

    Tool_Num = len(tools_list)  # 刀具种类数
    Tool_Time = np.zeros((N, Q_max, M, Tool_Num))  # 各产品各工序在不同设备上加工时，对各类型刀具的需求时间
    for i in process.keys():
        product_index = Products.index(i)
        for q in range(len(process[i])):
            for j in process[i][q]['machineID']:
                machine_index = machines.index(j)
                for t in range(len(process[i][q]['toolTypeID'])):
                    tool = process[i][q]['toolTypeID'][t]
                    tool_index = tools_list.index(tool)
                    Tool_Time[product_index][q][machine_index][tool_index] = process[i][q]['toolTime'][t] + \
                                                                             Tool_Time[product_index][q][machine_index][
                                                                                 tool_index]

    # 刀具库参数获取
    Tool_inventory = np.zeros((M, Tool_Num))
    for i in machineTools.keys():
        machine_index = machines.index(i)
        for j in range(len(machineTools[i])):
            tool = machineTools[i][j]['toolTypeID']
            tool_index = tools_list.index(tool)
            Tool_inventory[machine_index][tool_index] = machineTools[i][j]['toolAvailableTime'] + \
                                                        Tool_inventory[machine_index][tool_index]

    Tool_require = []
    toolCheck = []  # 记录刀具检查结果，作为输出

    # 资源齐套性检查参数获取_外协件
    Outpart_Type_list = []  # 统计所有用到的的外协件
    for i in process.keys():
        for q in range(len(process[i])):
            if process[i][q]['cooperationPartTypeID'] != 0:
                for c in range(len(process[i][q]['cooperationPartTypeID'])):
                    if process[i][q]['cooperationPartTypeID'][c] not in Outpart_Type_list:
                        Outpart_Type_list.append(process[i][q]['cooperationPartTypeID'][c])

    Outpart_type_nb = len(Outpart_Type_list)  # 外协件种类数
    Outpart_Num = np.zeros((N, Q_max, Outpart_type_nb))  # 各产品各工序对各类型外协件的需求
    for i in process.keys():
        product_index = Products.index(i)
        for q in range(len(process[i])):
            if process[i][q]['cooperationPartTypeID'] != 0:
                for c in range(len(process[i][q]['cooperationPartTypeID'])):
                    Outpart_type = process[i][q]['cooperationPartTypeID'][c]
                    Outpart_type_index = Outpart_Type_list.index(Outpart_type)
                    Outpart_Num[product_index][q][Outpart_type_index] = process[i][q]['cooperationPartNB'][c]

    # 外协件库存信息获取
    Outpart_inventory = np.zeros((1, Outpart_type_nb))
    for i in cooperationPartInventory.keys():
        Outpart_type_index = Outpart_Type_list.index(i)
        Outpart_inventory[0][Outpart_type_index] = cooperationPartInventory[i]

    Outpart_require = []
    cooperationPartCheck = []  # 记录外协件检查结果，作为输出

    # 人员齐套资源检查
    laborCheck = []  # 记录下一周期需要人员操作的设备id，作为输出

    # 获取拆分后的订单信息
    X_val = np.zeros((L, K, N))  # 订单内各产品需求数量
    for i in range(len(ordersSplit)):
        period_index = ordersSplit[i]['periodID'] - 1
        order_index = Orders.index(ordersSplit[i]['orderID'])
        product_index = Products.index(ordersSplit[i]['productID'])
        product_NB = ordersSplit[i]['productNB']
        X_val[period_index][order_index][product_index] = product_NB

    preschedule = []  # 记录预排产信息
    capacity_check = []  # 记录各周期产能校核情况

    for l in range(L):
        if np.sum(X_val[l]) == 0:
            continue
        T_begin = T_begin + timedelta(days=l * T_cycleday)
        Q_now = np.zeros((K, N))  # 某订单某产品当前执行的工序
        T_rp = np.zeros((K, N))  # 某订单某产品当前剩余工序所需时间
        Q_judge = np.zeros((K, N))  # 某订单某产品当前剩余工序数量
        fig_path = 'E:/CKX（研究生）/潍柴智能排产项目/Python代码/ResourceCompleteness_Check/R4_C2_' + str(l + 1) + '.html'
        ResCheck_fig1 = 'E:/CKX（研究生）/潍柴智能排产项目/Python代码/ResourceCompleteness_Check/ResCheck_Fixture_' + str(
            l + 1) + '.html'
        txt_path = 'E:/CKX（研究生）/潍柴智能排产项目/Python代码/ResourceCompleteness_Check/R4_C2_datail' + str(l + 1) + '.txt'
        ResCheck_txt1 = 'E:/CKX（研究生）/潍柴智能排产项目/Python代码/ResourceCompleteness_Check/ResCheck_Blank_' + str(l + 1) + '.txt'
        ResCheck_txt2 = 'E:/CKX（研究生）/潍柴智能排产项目/Python代码/ResourceCompleteness_Check/ResCheck_Fixture_' + str(
            l + 1) + '.txt'
        ResCheck_txt3 = 'E:/CKX（研究生）/潍柴智能排产项目/Python代码/ResourceCompleteness_Check/ResCheck_Fixture_Require' + str(
            l + 1) + '.txt'
        ResCheck_txt4 = 'E:/CKX（研究生）/潍柴智能排产项目/Python代码/ResourceCompleteness_Check/ResCheck_Tool_Require' + str(
            l + 1) + '.txt'
        ResCheck_txt5 = 'E:/CKX（研究生）/潍柴智能排产项目/Python代码/ResourceCompleteness_Check/ResCheck_Outpart_Require' + str(
            l + 1) + '.txt'
        ResCheck_txt6 = 'E:/CKX（研究生）/潍柴智能排产项目/Python代码/ResourceCompleteness_Check/ResCheck_Machine_Condition' + str(
            l + 1) + '.txt'

        # 计算各订单各产品剩余工序数量
        for k in range(K):
            for i in range(N):
                if X_val[l][k][i] != 0:
                    Q_judge[k][i] = Q[i]

        Priority = np.zeros((K, N))  # 订单内各产品优先级
        # 计算各订单各产品首道工序加工时间
        for k in range(K):
            for i in range(N):
                if X_val[l][k][i] == 0:
                    Priority[k][i] = LargeM
                else:
                    Priority[k][i] = X_val[l][k][i] * np.sum(T[i][0]) / np.sum(T[i][0] > 0)
        # [订单、产品、产品编号、工序号、设备号、托盘系统编号、托盘编号、夹具类型编号]
        Serial_num = []  # [订单、产品、产品编号、工序号、设备号]
        T_sf = []  # [某一产品某一工序开始加工时间、结束加工时间]

        while sum(sum(Q_judge)) > 0.0001:
            TempMin = LargeM
            for u in range(K):
                for v in range(N):
                    if (Priority[u][v] < TempMin) & (Q_judge[u][v] > 0):
                        TempMin = Priority[u][v]
                        idx = u  # 订单编号
                        idy = v  # 产品类别
            idz = int(Q_now[idx][idy])  # 最紧急订单的最紧急产品 当前的生产工序
            if np.sum(T[idy][idz] > 0) == 0:
                print(l)
                print(idx)
                print(idy)
                print(idz)
                exit(-2)

            flag = 0
            for v in range(int(X_val[l][idx][idy])):
                if len(Q_continuous[idy]) > 0:
                    for w in range(len(Q_continuous[idy])):
                        if idz == Q_continuous[idy][w][0]:
                            flag = len(Q_continuous[idy][w])
                            for ww in range(len(Q_continuous[idy][w])):
                                Result = Func1(Serial_num, T_sf, idx, idy, v, idz + ww, T,
                                               ResCheck_Machine_TraySystem,
                                               ResCheck_TraySys_TrayIndex_Fixture,
                                               ResCheck_Item_Fixture)
                                Result1 = [Result[0], Result[1], Result[2], Result[3], Result[4], Result[5], Result[6],
                                           Result[7]]
                                Serial_num.append(deepcopy(Result1))
                                Result2 = [Result[8], Result[9]]
                                T_sf.append(deepcopy(Result2))
                if flag == 0:
                    Result = Func1(Serial_num, T_sf, idx, idy, v, idz, T,
                                   ResCheck_Machine_TraySystem,
                                   ResCheck_TraySys_TrayIndex_Fixture,
                                   ResCheck_Item_Fixture)
                    Result1 = [Result[0], Result[1], Result[2], Result[3], Result[4], Result[5], Result[6],
                               Result[7]]
                    Serial_num.append(deepcopy(Result1))
                    Result2 = [Result[8], Result[9]]
                    T_sf.append(deepcopy(Result2))
            if flag == 0:
                Q_now[idx][idy] = Q_now[idx][idy] + 1
            else:
                Q_now[idx][idy] = Q_now[idx][idy] + flag

            if Q_now[idx][idy] == Q[idy]:
                Priority[idx][idy] = LargeM
            else:
                Priority[idx][idy] = X_val[l][idx][idy] * np.sum(T[idy][int(Q_now[idx][idy])]) / np.sum(
                    T[idy][int(Q_now[idx][idy])] > 0)
            Q_judge[idx][idy] = Q[idy] - Q_now[idx][idy]

        # doc = open(txt_path,'w')
        # for i in range(len(Serial_num)):
        #         print("{}  {}".format(Serial_num[i],T_sf[i]),file=doc)
        # doc.close()

        # 画甘特图
        # GANTT(Serial_num, T_sf, M, T_begin, fig_path)

        # 产能校核
        temp_max = max(max(row) for row in T_sf)  # 获取二维列表的最大值
        if temp_max < T_avr * 0.9:
            print("{} 周期产能校核通过".format(l + 1))
            capacity_check.append(True)
        else:
            capacity_check.append(False)

        # 将生产甘特图转换为资源使用甘特图
        Fixture_Tsf = Func3(Serial_num, T_sf, ResCheck_Item_Fixture, ResCheck_Machine_TraySystem)
        # doc3 = open(ResCheck_txt2, 'w')
        # for i in range(len(Fixture_Tsf)):
        #         print(Fixture_Tsf[i], file=doc3)
        # doc3.close()
        # GANTT_ResCheck_Fixture(Fixture_Tsf, M, T_begin, ResCheck_fig1)

        # 根据资源使用情况获取夹具最大同时使用数量
        Fixture_require = Func4(Fixture_Tsf, TraySystem_Num, Fixture_Num)
        # doc4 = open(ResCheck_txt3, 'w')
        # for i in range(TraySystem_Num):
        #         print(Fixture_require[i], file=doc4)
        # doc4.close()

        # 资源齐套检查
        # 资源齐套性检查_毛坯
        X_val_l = np.zeros((1, N))  # 该周期内各产品的生产数量
        for k in range(K):
            for i in range(N):
                X_val_l[0][i] = X_val_l[0][i] + X_val[l][k][i]
        # 获取各类型毛坯的需求量
        # doc2 = open(ResCheck_txt1, 'w')
        for B_class in range(Blank_class_max):
            for i in range(N):
                Blank_require[l][B_class] = Blank_require[l][B_class] + Blank[i][B_class] * X_val_l[0][i]
            # print("{}周期{}类毛坯需求量为{}".format(l + 1, B_class + 1, Blank_require[0][B_class]), file=doc2)
            if l == 0:
                if Blank_require[0][B_class] > Blank_inventory[0][B_class]:
                    lackNB = Blank_require[0][B_class] - Blank_inventory[0][B_class]
                    detail = {'periodID': l+1, 'blankType': blanks_list[B_class], 'requireNB': Blank_require[l][B_class], 'lackNB': lackNB}
                    blankCheck.append(detail)
                    # print("{}周期{}类毛坯资源齐套性检查不通过，缺少{}件".format(l + 1, B_class + 1, lackNB), file=doc2)
                else:
                    detail = {'periodID': l+1, 'blankType': blanks_list[B_class], 'requireNB': Blank_require[l][B_class], 'lackNB': 0}
                    blankCheck.append(detail)
                    # print("{}周期{}类毛坯资源齐套性检查通过".format(l + 1, B_class + 1), file=doc2)
            else:
                detail = {'periodID': l+1, 'blankType': blanks_list[B_class], 'requireNB': Blank_require[l][B_class]}
                blankCheck.append(detail)
        # doc2.close()

        # 资源检查_刀具
        # 计算获取各设备上各类型刀具的使用量
        Tool_Time_Require = Func5(Serial_num, Tool_Time, M, Tool_Num)
        Tool_require.append(Tool_Time_Require)
        # doc5 = open(ResCheck_txt4, 'w')
        for i in range(len(Tool_Time_Require)):
            for j in range(len(Tool_Time_Require[i])):
                if Tool_Time_Require[i][j] > 0:
                    # print("{}周期{}设备{}类刀具需求量为{}分钟".format(l + 1, i + 1, j + 1, Tool_Time_Require[i][j]), file=doc5)
                    if l == 0:
                        if Tool_Time_Require[i][j] > Tool_inventory[i][j]:
                            lackTime = Tool_Time_Require[i][j] - Tool_inventory[i][j]
                            detail = {'periodID': l+1, 'machineID': machines[i], 'toolTypeID': tools_list[j], 'requireTime': Tool_Time_Require[i][j], 'lackTime': lackTime}
                            toolCheck.append(detail)
                        else:
                            detail = {'periodID': l+1, 'machineID': machines[i], 'toolTypeID': tools_list[j], 'requireTime': Tool_Time_Require[i][j], 'lackTime': 0}
                            toolCheck.append(detail)
                    else:
                        detail = {'periodID': l+1, 'machineID': machines[i], 'toolTypeID': tools_list[j], 'requireTime': Tool_Time_Require[i][j]}
                        toolCheck.append(detail)
        # doc5.close()

        # 计算获取各种外协件的使用数量
        Outpart_Num_Require = Func6(Serial_num, Outpart_Num, Outpart_type_nb)
        # doc6 = open(ResCheck_txt5, 'w')
        Outpart_require.append(Outpart_Num_Require)  # 记录各周期的外协件需求数量
        for i in range(len(Outpart_Num_Require)):
            # print("{}周期{}类外协件需求量为{}件".format(l + 1, i + 1, Outpart_Num_Require[i]), file=doc6)
            if l == 0:
                if Outpart_Num_Require[i] > Outpart_inventory[0][i]:
                    lackNB = Outpart_Num_Require[i] - Outpart_inventory[0][i]
                    detail = {'periodID': l+1, 'cooperationPartTypeID': Outpart_Type_list[i], 'requireNB': Outpart_Num_Require[i], 'lackNB': lackNB}
                    cooperationPartCheck.append(detail)
                else:
                    detail = {'periodID': l+1, 'cooperationPartTypeID': Outpart_Type_list[i], 'requireNB': Outpart_Num_Require[i], 'lackNB': 0}
                    cooperationPartCheck.append(detail)
            else:
                detail = {'periodID': l+1, 'cooperationPartTypeID': Outpart_Type_list[i], 'requireNB': Outpart_Num_Require[i]}
                cooperationPartCheck.append(detail)
        # doc6.close()

        # 获取各周期内设备运行状况
        Worker_Machine_Condition = Func7(Serial_num, T_sf, T_shift, Num_shift, M)
        # doc7 = open(ResCheck_txt6, 'w')
        # print("{}周期人员操作设备情况为：".format(l + 1), file=doc7)
        for i in range(Num_shift):
            # print("班次{}需要人员操作设备为:".format(i + 1), file=doc7)
            machines_to_operate = []
            for j in range(M):
                if Worker_Machine_Condition[i][j] > 0:
                    # print("设备{}".format(j + 1), '\t', end='', file=doc7)
                    if l == 0:
                        machines_to_operate.append(machines[j])
            # print("\n", file=doc7)
            laborCheck.append({'shiftID': i + 1, 'machineID': machines_to_operate})
        # doc7.close()

        for i in range(len(Serial_num)):
            order_index = Serial_num[i][0]
            order_id = Orders[order_index]
            job_type_index = Serial_num[i][1]
            job_type = Products[job_type_index]
            job_serial_nb = Serial_num[i][2] + 1
            process_id = Serial_num[i][3] + 1
            machine_index = Serial_num[i][4]
            machine_id = machines[machine_index]
            start_time = T_sf[i][0]
            end_time = T_sf[i][1]
            detail = {'periodID': l+1, 'orderID': order_id, 'jobType': job_type, 'jobSerialNB': job_serial_nb, 'processID': process_id,
                      'machine': machine_id, 'startTime': start_time, 'endTime': end_time}
            preschedule.append(detail)

    capacityCheck = True
    for i in range(len(capacity_check)):
        if not capacity_check[i]:
            capacityCheck = False
            break

    # print(preschedule)
    # print(capacityCheck)
    # print(blankCheck)
    # print(toolCheck)
    # print(cooperationPartCheck)
    # print(laborCheck)

    response = {'ordersSplit': ordersSplit,
                'ordersDelay': ordersDelay,
                'preschedule': preschedule,
                'capacityCheck': capacityCheck,
                'blankCheck': blankCheck,
                'toolCheck': toolCheck,
                'cooperationPartCheck': cooperationPartCheck,
                'laborCheck': laborCheck
                }
    return json.dumps(response)


# #############################################################
# 预排产结束
# ############################################################
# 紧急插单开始
# ##################################################
@app.route('/insert', methods=['GET', 'POST'])
def insert():
    if request.method == 'POST':
        data = request.get_data()
        data = json.loads(data)
    else:
        data = {'hours': 22,
                'periodLength': 7,
                'planStart': '2022-06-01 00:00:00',
                'process': {'P0001': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0002': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0003': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0004': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0005': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0006': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0007': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0008': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0009': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0010': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0011': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0012': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0013': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0014': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0015': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0016': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0017': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0018': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0019': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0020': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0021': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0022': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0023': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0024': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0025': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0026': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0027': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0028': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0029': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0030': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0031': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0032': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0033': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0034': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0035': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0036': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0037': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0038': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0039': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0040': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0041': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0042': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0043': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0044': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0045': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0046': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0047': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0048': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0049': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0050': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0051': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0052': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0053': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0054': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0055': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0056': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0057': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0058': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0059': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0060': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0061': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0062': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0063': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0064': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0065': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0066': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0067': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0068': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0069': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0070': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0071': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0072': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0073': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0074': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0075': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0076': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0077': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0078': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0079': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0080': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0081': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0082': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0083': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0084': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0085': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0086': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0087': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0088': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0089': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0090': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0091': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0092': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0093': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0094': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0095': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0096': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0097': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0098': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0099': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0100': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]}},
                'replanTime': '2022-06-02 00:00:00',
                'pendingProcessMachine': {'M01': [],
                                          'M02': [],
                                          'M03': [],
                                          'M04': [],
                                          'M05': [],
                                          'M06': ['P0011p001',
                                                  'P0012p001',
                                                  'P0013p001',
                                                  'P0014p001',
                                                  'P0015p001',
                                                  'P0016p001',
                                                  'P0017p001',
                                                  'P0041p001',
                                                  'P0042p001',
                                                  'P0043p001',
                                                  'P0044p001',
                                                  'P0045p001',
                                                  'P0046p001',
                                                  'P0047p001',
                                                  'P0048p001'],
                                          'M07': ['P0051p006',
                                                  'P0051p007',
                                                  'P0052p006',
                                                  'P0052p007',
                                                  'P0061p006',
                                                  'P0061p007',
                                                  'P0071p001',
                                                  'P0072p001',
                                                  'P0073p001',
                                                  'P0074p001',
                                                  'P0075p001',
                                                  'P0076p001',
                                                  'P0081p004',
                                                  'P0081p010',
                                                  'P0082p004',
                                                  'P0082p010',
                                                  'P0083p004',
                                                  'P0084p004',
                                                  'P0085p004',
                                                  'P0086p004',
                                                  'P0087p004',
                                                  'P0088p004',
                                                  'P0089p004',
                                                  'P0090p004'],
                                          'M08': [],
                                          'M09': [],
                                          'M10': [],
                                          'M11': [],
                                          'M12': ['P0011p004',
                                                  'P0011p007',
                                                  'P0012p004',
                                                  'P0013p004',
                                                  'P0041p002',
                                                  'P0041p003',
                                                  'P0042p002',
                                                  'P0042p003',
                                                  'P0043p002',
                                                  'P0043p003',
                                                  'P0044p002',
                                                  'P0044p003',
                                                  'P0045p002',
                                                  'P0045p003',
                                                  'P0046p002',
                                                  'P0046p003',
                                                  'P0047p002',
                                                  'P0047p003'],
                                          'M13': ['P0001p001',
                                                  'P0002p001',
                                                  'P0003p001',
                                                  'P0004p001',
                                                  'P0005p001',
                                                  'P0006p001',
                                                  'P0007p001',
                                                  'P0008p001',
                                                  'P0009p001',
                                                  'P0010p001',
                                                  'P0081p002',
                                                  'P0081p006',
                                                  'P0082p002',
                                                  'P0082p006',
                                                  'P0083p002',
                                                  'P0083p006',
                                                  'P0084p002',
                                                  'P0084p006',
                                                  'P0085p002',
                                                  'P0085p006',
                                                  'P0086p002',
                                                  'P0086p006',
                                                  'P0087p002',
                                                  'P0087p006',
                                                  'P0088p002',
                                                  'P0088p006',
                                                  'P0089p002',
                                                  'P0089p006',
                                                  'P0090p002'],
                                          'M14': ['P0011p006',
                                                  'P0041p004',
                                                  'P0042p004',
                                                  'P0043p004',
                                                  'P0044p004',
                                                  'P0045p004',
                                                  'P0051p005',
                                                  'P0052p005',
                                                  'P0053p005',
                                                  'P0061p005',
                                                  'P0081p003',
                                                  'P0081p007',
                                                  'P0082p003',
                                                  'P0082p007',
                                                  'P0083p003',
                                                  'P0083p007',
                                                  'P0084p003',
                                                  'P0084p007',
                                                  'P0085p003',
                                                  'P0085p007',
                                                  'P0086p003',
                                                  'P0086p007',
                                                  'P0087p003',
                                                  'P0088p003',
                                                  'P0089p003',
                                                  'P0090p003'],
                                          'M15': ['P0011p005',
                                                  'P0012p005',
                                                  'P0041p005',
                                                  'P0042p005',
                                                  'P0043p005',
                                                  'P0044p005',
                                                  'P0045p005',
                                                  'P0051p004',
                                                  'P0052p004',
                                                  'P0053p004',
                                                  'P0054p004',
                                                  'P0061p004',
                                                  'P0062p004',
                                                  'P0063p004',
                                                  'P0064p004',
                                                  'P0081p008',
                                                  'P0081p009',
                                                  'P0082p008',
                                                  'P0082p009',
                                                  'P0083p008',
                                                  'P0083p009',
                                                  'P0084p008',
                                                  'P0084p009',
                                                  'P0085p008',
                                                  'P0085p009',
                                                  'P0086p008'],
                                          'M16': ['P0011p002',
                                                  'P0011p003',
                                                  'P0012p002',
                                                  'P0012p003',
                                                  'P0013p002',
                                                  'P0013p003',
                                                  'P0014p002',
                                                  'P0014p003',
                                                  'P0015p002',
                                                  'P0015p003',
                                                  'P0016p002',
                                                  'P0016p003',
                                                  'P0041p006',
                                                  'P0041p007',
                                                  'P0042p006',
                                                  'P0043p006',
                                                  'P0051p001',
                                                  'P0051p002',
                                                  'P0051p003',
                                                  'P0052p001',
                                                  'P0052p002',
                                                  'P0052p003',
                                                  'P0053p001',
                                                  'P0053p002',
                                                  'P0053p003',
                                                  'P0054p001',
                                                  'P0054p002',
                                                  'P0054p003',
                                                  'P0055p001',
                                                  'P0055p002',
                                                  'P0055p003',
                                                  'P0056p001',
                                                  'P0056p002',
                                                  'P0057p001',
                                                  'P0061p001',
                                                  'P0061p002',
                                                  'P0061p003',
                                                  'P0062p001',
                                                  'P0062p002',
                                                  'P0062p003',
                                                  'P0063p001',
                                                  'P0063p002',
                                                  'P0063p003',
                                                  'P0064p001',
                                                  'P0064p002',
                                                  'P0064p003',
                                                  'P0065p001',
                                                  'P0065p002',
                                                  'P0065p003',
                                                  'P0066p001',
                                                  'P0066p002',
                                                  'P0067p001',
                                                  'P0068p001',
                                                  'P0069p001',
                                                  'P0070p001',
                                                  'P0071p002',
                                                  'P0071p003',
                                                  'P0072p002',
                                                  'P0072p003',
                                                  'P0073p002',
                                                  'P0074p002'],
                                          'M17': ['P0031p001',
                                                  'P0038p001',
                                                  'P0081p001',
                                                  'P0081p005',
                                                  'P0082p001',
                                                  'P0082p005',
                                                  'P0083p001',
                                                  'P0083p005',
                                                  'P0084p001',
                                                  'P0084p005',
                                                  'P0085p001',
                                                  'P0085p005',
                                                  'P0086p001',
                                                  'P0086p005',
                                                  'P0087p001',
                                                  'P0087p005',
                                                  'P0088p001',
                                                  'P0088p005',
                                                  'P0089p001',
                                                  'P0089p005',
                                                  'P0090p001',
                                                  'P0090p005'],
                                          'M18': [],
                                          'M19': [],
                                          'M20': [],
                                          'M21': [],
                                          'M22': [],
                                          'M23': [],
                                          'M24': [],
                                          'M25': [],
                                          'M26': ['P0021p001', 'P0021p002', 'P0022p001', 'P0023p001'],
                                          'M27': [],
                                          'M28': [],
                                          'M29': [],
                                          'M30': [],
                                          'M31': [],
                                          'M32': [],
                                          'M33': [],
                                          'M34': [],
                                          'M35': [],
                                          'M36': [],
                                          'M37': [],
                                          'M38': [],
                                          'M39': [],
                                          'M40': [],
                                          'M41': [],
                                          'M42': [],
                                          'M43': [],
                                          'M44': [],
                                          'M45': [],
                                          'M46': ['P0032p001',
                                                  'P0033p001',
                                                  'P0034p001',
                                                  'P0035p001',
                                                  'P0036p001',
                                                  'P0037p001',
                                                  'P0039p001',
                                                  'P0040p001'],
                                          'M47': []},
                'pendingProcessOriginalPlan': {'M01': [], 'M02': [], 'M03': [], 'M04': [], 'M05': [],
                                               'M06': [['2022-06-01 00:00:00', '2022-06-01 02:22:00'],
                                                       ['2022-06-01 07:58:00', '2022-06-01 10:20:00'],
                                                       ['2022-06-01 10:20:00', '2022-06-01 12:42:00'],
                                                       ['2022-06-01 18:17:00', '2022-06-01 20:39:00'],
                                                       ['2022-06-01 23:27:00', '2022-06-02 01:49:00'],
                                                       ['2022-06-02 04:37:00', '2022-06-02 06:59:00'],
                                                       ['2022-06-02 09:47:00', '2022-06-02 12:09:00'],
                                                       ['2022-06-01 02:22:00', '2022-06-01 05:10:00'],
                                                       ['2022-06-01 05:10:00', '2022-06-01 07:58:00'],
                                                       ['2022-06-01 12:41:00', '2022-06-01 15:29:00'],
                                                       ['2022-06-01 15:29:00', '2022-06-01 18:17:00'],
                                                       ['2022-06-01 20:39:00', '2022-06-01 23:27:00'],
                                                       ['2022-06-02 01:49:00', '2022-06-02 04:37:00'],
                                                       ['2022-06-02 06:59:00', '2022-06-02 09:47:00'],
                                                       ['2022-06-02 12:09:00', '2022-06-02 14:57:00']],
                                               'M07': [['2022-06-03 19:56:00', '2022-06-03 22:25:00'],
                                                       ['2022-06-04 01:39:00', '2022-06-04 04:08:00'],
                                                       ['2022-06-04 14:19:00', '2022-06-04 16:48:00'],
                                                       ['2022-06-05 00:08:00', '2022-06-05 02:37:00'],
                                                       ['2022-06-04 08:15:00', '2022-06-04 10:44:00'],
                                                       ['2022-06-04 18:25:00', '2022-06-04 20:54:00'],
                                                       ['2022-06-01 00:00:00', '2022-06-01 03:35:00'],
                                                       ['2022-06-01 06:08:00', '2022-06-01 09:43:00'],
                                                       ['2022-06-01 11:20:00', '2022-06-01 14:55:00'],
                                                       ['2022-06-01 16:32:00', '2022-06-01 20:07:00'],
                                                       ['2022-06-01 21:44:00', '2022-06-02 01:19:00'],
                                                       ['2022-06-04 10:44:00', '2022-06-04 14:19:00'],
                                                       ['2022-06-01 04:31:00', '2022-06-01 06:08:00'],
                                                       ['2022-06-04 05:01:00', '2022-06-04 06:38:00'],
                                                       ['2022-06-01 09:43:00', '2022-06-01 11:20:00'],
                                                       ['2022-06-04 20:54:00', '2022-06-04 22:31:00'],
                                                       ['2022-06-01 14:55:00', '2022-06-01 16:32:00'],
                                                       ['2022-06-01 20:07:00', '2022-06-01 21:44:00'],
                                                       ['2022-06-03 22:25:00', '2022-06-04 00:02:00'],
                                                       ['2022-06-04 00:02:00', '2022-06-04 01:39:00'],
                                                       ['2022-06-04 06:38:00', '2022-06-04 08:15:00'],
                                                       ['2022-06-04 16:48:00', '2022-06-04 18:25:00'],
                                                       ['2022-06-04 22:31:00', '2022-06-05 00:08:00'],
                                                       ['2022-06-05 02:37:00', '2022-06-05 04:14:00']], 'M08': [],
                                               'M09': [],
                                               'M10': [], 'M11': [],
                                               'M12': [['2022-06-03 07:23:00', '2022-06-03 09:47:00'],
                                                       ['2022-06-06 04:15:00', '2022-06-06 06:39:00'],
                                                       ['2022-06-04 08:07:00', '2022-06-04 10:31:00'],
                                                       ['2022-06-06 08:25:00', '2022-06-06 10:49:00'],
                                                       ['2022-06-01 05:10:00', '2022-06-01 06:56:00'],
                                                       ['2022-06-01 06:56:00', '2022-06-01 08:42:00'],
                                                       ['2022-06-01 08:41:00', '2022-06-01 10:27:00'],
                                                       ['2022-06-01 10:27:00', '2022-06-01 12:13:00'],
                                                       ['2022-06-01 15:29:00', '2022-06-01 17:15:00'],
                                                       ['2022-06-01 17:15:00', '2022-06-01 19:01:00'],
                                                       ['2022-06-01 19:01:00', '2022-06-01 20:47:00'],
                                                       ['2022-06-03 09:47:00', '2022-06-03 11:33:00'],
                                                       ['2022-06-03 11:33:00', '2022-06-03 13:19:00'],
                                                       ['2022-06-04 10:31:00', '2022-06-04 12:17:00'],
                                                       ['2022-06-03 13:19:00', '2022-06-03 15:05:00'],
                                                       ['2022-06-04 12:17:00', '2022-06-04 14:03:00'],
                                                       ['2022-06-04 14:03:00', '2022-06-04 15:49:00'],
                                                       ['2022-06-06 06:39:00', '2022-06-06 08:25:00']],
                                               'M13': [['2022-06-01 00:00:00', '2022-06-01 00:18:00'],
                                                       ['2022-06-01 00:18:00', '2022-06-01 00:36:00'],
                                                       ['2022-06-01 00:36:00', '2022-06-01 00:54:00'],
                                                       ['2022-06-01 00:54:00', '2022-06-01 01:12:00'],
                                                       ['2022-06-01 01:12:00', '2022-06-01 01:30:00'],
                                                       ['2022-06-01 01:30:00', '2022-06-01 01:48:00'],
                                                       ['2022-06-01 03:17:00', '2022-06-01 03:35:00'],
                                                       ['2022-06-01 03:35:00', '2022-06-01 03:53:00'],
                                                       ['2022-06-03 11:24:00', '2022-06-03 11:42:00'],
                                                       ['2022-06-03 11:42:00', '2022-06-03 12:00:00'],
                                                       ['2022-06-01 01:48:00', '2022-06-01 03:17:00'],
                                                       ['2022-06-03 12:00:00', '2022-06-03 13:29:00'],
                                                       ['2022-06-01 03:53:00', '2022-06-01 05:22:00'],
                                                       ['2022-06-03 19:25:00', '2022-06-03 20:54:00'],
                                                       ['2022-06-01 05:59:00', '2022-06-01 07:28:00'],
                                                       ['2022-06-03 22:23:00', '2022-06-03 23:52:00'],
                                                       ['2022-06-01 07:28:00', '2022-06-01 08:57:00'],
                                                       ['2022-06-03 23:52:00', '2022-06-04 01:21:00'],
                                                       ['2022-06-03 13:29:00', '2022-06-03 14:58:00'],
                                                       ['2022-06-04 01:21:00', '2022-06-04 02:50:00'],
                                                       ['2022-06-03 14:58:00', '2022-06-03 16:27:00'],
                                                       ['2022-06-04 21:00:00', '2022-06-04 22:29:00'],
                                                       ['2022-06-03 16:27:00', '2022-06-03 17:56:00'],
                                                       ['2022-06-04 22:29:00', '2022-06-04 23:58:00'],
                                                       ['2022-06-03 17:56:00', '2022-06-03 19:25:00'],
                                                       ['2022-06-04 23:58:00', '2022-06-05 01:27:00'],
                                                       ['2022-06-03 20:54:00', '2022-06-03 22:23:00'],
                                                       ['2022-06-05 01:27:00', '2022-06-05 02:56:00'],
                                                       ['2022-06-04 02:50:00', '2022-06-04 04:19:00']],
                                               'M14': [['2022-06-04 12:05:00', '2022-06-04 14:20:00'],
                                                       ['2022-06-01 08:42:00', '2022-06-01 11:18:00'],
                                                       ['2022-06-03 16:22:00', '2022-06-03 18:58:00'],
                                                       ['2022-06-03 20:12:00', '2022-06-03 22:48:00'],
                                                       ['2022-06-04 06:38:00', '2022-06-04 09:14:00'],
                                                       ['2022-06-04 16:48:00', '2022-06-04 19:24:00'],
                                                       ['2022-06-03 18:58:00', '2022-06-03 19:56:00'],
                                                       ['2022-06-04 02:30:00', '2022-06-04 03:28:00'],
                                                       ['2022-06-04 04:26:00', '2022-06-04 05:24:00'],
                                                       ['2022-06-04 03:28:00', '2022-06-04 04:26:00'],
                                                       ['2022-06-01 03:17:00', '2022-06-01 04:31:00'],
                                                       ['2022-06-03 13:29:00', '2022-06-03 14:43:00'],
                                                       ['2022-06-01 05:22:00', '2022-06-01 06:36:00'],
                                                       ['2022-06-04 00:02:00', '2022-06-04 01:16:00'],
                                                       ['2022-06-01 07:28:00', '2022-06-01 08:42:00'],
                                                       ['2022-06-04 01:16:00', '2022-06-04 02:30:00'],
                                                       ['2022-06-01 11:18:00', '2022-06-01 12:32:00'],
                                                       ['2022-06-04 14:20:00', '2022-06-04 15:34:00'],
                                                       ['2022-06-03 14:58:00', '2022-06-03 16:12:00'],
                                                       ['2022-06-04 19:24:00', '2022-06-04 20:38:00'],
                                                       ['2022-06-03 22:48:00', '2022-06-04 00:02:00'],
                                                       ['2022-06-04 22:29:00', '2022-06-04 23:43:00'],
                                                       ['2022-06-04 05:24:00', '2022-06-04 06:38:00'],
                                                       ['2022-06-04 15:34:00', '2022-06-04 16:48:00'],
                                                       ['2022-06-04 20:38:00', '2022-06-04 21:52:00'],
                                                       ['2022-06-04 23:43:00', '2022-06-05 00:57:00']],
                                               'M15': [['2022-06-03 09:47:00', '2022-06-03 12:47:00'],
                                                       ['2022-06-04 10:31:00', '2022-06-04 13:31:00'],
                                                       ['2022-06-02 18:29:00', '2022-06-02 19:55:00'],
                                                       ['2022-06-04 03:28:00', '2022-06-04 04:54:00'],
                                                       ['2022-06-04 13:31:00', '2022-06-04 14:57:00'],
                                                       ['2022-06-04 15:48:00', '2022-06-04 17:14:00'],
                                                       ['2022-06-06 08:03:00', '2022-06-06 09:29:00'],
                                                       ['2022-06-01 13:41:00', '2022-06-01 16:17:00'],
                                                       ['2022-06-02 19:55:00', '2022-06-02 22:31:00'],
                                                       ['2022-06-02 22:31:00', '2022-06-03 01:07:00'],
                                                       ['2022-06-03 17:56:00', '2022-06-03 20:32:00'],
                                                       ['2022-06-04 00:52:00', '2022-06-04 03:28:00'],
                                                       ['2022-06-04 04:54:00', '2022-06-04 07:30:00'],
                                                       ['2022-06-05 00:34:00', '2022-06-05 03:10:00'],
                                                       ['2022-06-06 05:27:00', '2022-06-06 08:03:00'],
                                                       ['2022-06-03 23:10:00', '2022-06-04 00:01:00'],
                                                       ['2022-06-04 00:01:00', '2022-06-04 00:52:00'],
                                                       ['2022-06-04 09:40:00', '2022-06-04 10:31:00'],
                                                       ['2022-06-04 17:14:00', '2022-06-04 18:05:00'],
                                                       ['2022-06-04 14:57:00', '2022-06-04 15:48:00'],
                                                       ['2022-06-04 18:56:00', '2022-06-04 19:47:00'],
                                                       ['2022-06-04 18:05:00', '2022-06-04 18:56:00'],
                                                       ['2022-06-04 19:47:00', '2022-06-04 20:38:00'],
                                                       ['2022-06-04 20:38:00', '2022-06-04 21:29:00'],
                                                       ['2022-06-04 21:29:00', '2022-06-04 22:20:00'],
                                                       ['2022-06-04 23:43:00', '2022-06-05 00:34:00']],
                                               'M16': [['2022-06-01 07:56:00', '2022-06-01 11:05:00'],
                                                       ['2022-06-03 04:25:00', '2022-06-03 07:23:00'],
                                                       ['2022-06-02 22:31:00', '2022-06-03 01:40:00'],
                                                       ['2022-06-04 03:37:00', '2022-06-04 06:35:00'],
                                                       ['2022-06-05 00:51:00', '2022-06-05 04:00:00'],
                                                       ['2022-06-06 05:27:00', '2022-06-06 08:25:00'],
                                                       ['2022-06-05 04:00:00', '2022-06-05 07:09:00'],
                                                       ['2022-06-06 14:10:00', '2022-06-06 17:08:00'],
                                                       ['2022-06-06 08:25:00', '2022-06-06 11:34:00'],
                                                       ['2022-06-07 07:37:00', '2022-06-07 10:35:00'],
                                                       ['2022-06-07 12:10:00', '2022-06-07 15:19:00'],
                                                       ['2022-06-07 22:24:00', '2022-06-08 01:22:00'],
                                                       ['2022-06-03 17:56:00', '2022-06-03 19:31:00'],
                                                       ['2022-06-05 12:30:00', '2022-06-05 14:05:00'],
                                                       ['2022-06-07 10:35:00', '2022-06-07 12:10:00'],
                                                       ['2022-06-07 18:04:00', '2022-06-07 19:39:00'],
                                                       ['2022-06-01 00:00:00', '2022-06-01 02:36:00'],
                                                       ['2022-06-01 05:20:00', '2022-06-01 07:56:00'],
                                                       ['2022-06-01 11:05:00', '2022-06-01 13:41:00'],
                                                       ['2022-06-01 16:26:00', '2022-06-01 19:02:00'],
                                                       ['2022-06-01 21:38:00', '2022-06-02 00:14:00'],
                                                       ['2022-06-02 17:19:00', '2022-06-02 19:55:00'],
                                                       ['2022-06-01 19:02:00', '2022-06-01 21:38:00'],
                                                       ['2022-06-02 11:05:00', '2022-06-02 13:41:00'],
                                                       ['2022-06-02 19:55:00', '2022-06-02 22:31:00'],
                                                       ['2022-06-02 08:29:00', '2022-06-02 11:05:00'],
                                                       ['2022-06-03 07:23:00', '2022-06-03 09:59:00'],
                                                       ['2022-06-03 15:20:00', '2022-06-03 17:56:00'],
                                                       ['2022-06-03 09:59:00', '2022-06-03 12:35:00'],
                                                       ['2022-06-05 07:09:00', '2022-06-05 09:45:00'],
                                                       ['2022-06-05 14:05:00', '2022-06-05 16:41:00'],
                                                       ['2022-06-03 19:31:00', '2022-06-03 22:07:00'],
                                                       ['2022-06-06 11:34:00', '2022-06-06 14:10:00'],
                                                       ['2022-06-06 19:53:00', '2022-06-06 22:29:00'],
                                                       ['2022-06-01 02:35:00', '2022-06-01 05:20:00'],
                                                       ['2022-06-02 00:14:00', '2022-06-02 02:59:00'],
                                                       ['2022-06-03 22:07:00', '2022-06-04 00:52:00'],
                                                       ['2022-06-01 13:41:00', '2022-06-01 16:26:00'],
                                                       ['2022-06-02 05:44:00', '2022-06-02 08:29:00'],
                                                       ['2022-06-04 00:52:00', '2022-06-04 03:37:00'],
                                                       ['2022-06-02 02:59:00', '2022-06-02 05:44:00'],
                                                       ['2022-06-03 01:40:00', '2022-06-03 04:25:00'],
                                                       ['2022-06-04 09:20:00', '2022-06-04 12:05:00'],
                                                       ['2022-06-03 12:35:00', '2022-06-03 15:20:00'],
                                                       ['2022-06-05 09:45:00', '2022-06-05 12:30:00'],
                                                       ['2022-06-06 02:42:00', '2022-06-06 05:27:00'],
                                                       ['2022-06-04 06:35:00', '2022-06-04 09:20:00'],
                                                       ['2022-06-05 23:57:00', '2022-06-06 02:42:00'],
                                                       ['2022-06-06 17:08:00', '2022-06-06 19:53:00'],
                                                       ['2022-06-04 15:43:00', '2022-06-04 18:28:00'],
                                                       ['2022-06-07 15:19:00', '2022-06-07 18:04:00'],
                                                       ['2022-06-04 18:28:00', '2022-06-04 21:13:00'],
                                                       ['2022-06-07 02:07:00', '2022-06-07 04:52:00'],
                                                       ['2022-06-07 04:52:00', '2022-06-07 07:37:00'],
                                                       ['2022-06-07 19:39:00', '2022-06-07 22:24:00'],
                                                       ['2022-06-02 13:41:00', '2022-06-02 17:19:00'],
                                                       ['2022-06-04 21:13:00', '2022-06-05 00:51:00'],
                                                       ['2022-06-04 12:05:00', '2022-06-04 15:43:00'],
                                                       ['2022-06-06 22:29:00', '2022-06-07 02:07:00'],
                                                       ['2022-06-05 16:41:00', '2022-06-05 20:19:00'],
                                                       ['2022-06-05 20:19:00', '2022-06-05 23:57:00']],
                                               'M17': [['2022-06-01 01:19:00', '2022-06-01 02:00:00'],
                                                       ['2022-06-01 05:57:00', '2022-06-01 06:38:00'],
                                                       ['2022-06-01 00:00:00', '2022-06-01 01:19:00'],
                                                       ['2022-06-01 07:57:00', '2022-06-01 09:16:00'],
                                                       ['2022-06-01 02:00:00', '2022-06-01 03:19:00'],
                                                       ['2022-06-03 18:06:00', '2022-06-03 19:25:00'],
                                                       ['2022-06-01 03:19:00', '2022-06-01 04:38:00'],
                                                       ['2022-06-03 20:44:00', '2022-06-03 22:03:00'],
                                                       ['2022-06-01 04:38:00', '2022-06-01 05:57:00'],
                                                       ['2022-06-03 22:03:00', '2022-06-03 23:22:00'],
                                                       ['2022-06-01 06:38:00', '2022-06-01 07:57:00'],
                                                       ['2022-06-04 00:02:00', '2022-06-04 01:21:00'],
                                                       ['2022-06-01 09:16:00', '2022-06-01 10:35:00'],
                                                       ['2022-06-04 01:39:00', '2022-06-04 02:58:00'],
                                                       ['2022-06-01 10:35:00', '2022-06-01 11:54:00'],
                                                       ['2022-06-04 08:15:00', '2022-06-04 09:34:00'],
                                                       ['2022-06-01 11:54:00', '2022-06-01 13:13:00'],
                                                       ['2022-06-04 22:39:00', '2022-06-04 23:58:00'],
                                                       ['2022-06-01 13:13:00', '2022-06-01 14:32:00'],
                                                       ['2022-06-05 00:08:00', '2022-06-05 01:27:00'],
                                                       ['2022-06-03 19:25:00', '2022-06-03 20:44:00'],
                                                       ['2022-06-05 04:14:00', '2022-06-05 05:33:00']], 'M18': [],
                                               'M19': [],
                                               'M20': [], 'M21': [], 'M22': [], 'M23': [], 'M24': [], 'M25': [],
                                               'M26': [['2022-06-01 00:00:00', '2022-06-01 01:09:00'],
                                                       ['2022-06-01 02:17:00', '2022-06-01 03:26:00'],
                                                       ['2022-06-01 01:09:00', '2022-06-01 02:18:00'],
                                                       ['2022-06-01 03:26:00', '2022-06-01 04:35:00']], 'M27': [],
                                               'M28': [],
                                               'M29': [], 'M30': [], 'M31': [], 'M32': [], 'M33': [], 'M34': [],
                                               'M35': [],
                                               'M36': [],
                                               'M37': [], 'M38': [], 'M39': [], 'M40': [], 'M41': [], 'M42': [],
                                               'M43': [],
                                               'M44': [],
                                               'M45': [], 'M46': [['2022-06-01 00:00:00', '2022-06-01 00:31:00'],
                                                                  ['2022-06-01 00:31:00', '2022-06-01 01:02:00'],
                                                                  ['2022-06-01 01:02:00', '2022-06-01 01:33:00'],
                                                                  ['2022-06-01 01:33:00', '2022-06-01 02:04:00'],
                                                                  ['2022-06-01 02:04:00', '2022-06-01 02:35:00'],
                                                                  ['2022-06-01 02:35:00', '2022-06-01 03:06:00'],
                                                                  ['2022-06-01 03:06:00', '2022-06-01 03:37:00'],
                                                                  ['2022-06-01 03:37:00', '2022-06-01 04:08:00']],
                                               'M47': []},
                'insertOrder': {'P0101': {
                    'specificProcessID': ['P0101p001', 'P0101p002', 'P0101p003', 'P0101p004', 'P0101p005', 'P0101p006',
                                          'P0101p007', 'P0101p008'],
                    'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                    'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                    'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                    [144.0]],
                    'orderLevel': 1, 'dateEnd': '2022-06-08 00:00:00'},
                    'P0102': {
                        'specificProcessID': ['P0102p001', 'P0102p002', 'P0102p003', 'P0102p004',
                                              'P0102p005', 'P0102p006',
                                              'P0102p007', 'P0102p008'],
                        'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'],
                                      ['M12']],
                        'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                        'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                        [144.0]],
                        'orderLevel': 1, 'dateEnd': '2022-06-08 00:00:00'},
                    'P0103': {
                        'specificProcessID': ['P0103p001', 'P0103p002', 'P0103p003', 'P0103p004',
                                              'P0103p005', 'P0103p006',
                                              'P0103p007', 'P0103p008'],
                        'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'],
                                      ['M12']],
                        'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                        'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                        [144.0]],
                        'orderLevel': 1, 'dateEnd': '2022-06-08 00:00:00'}},
                'processProductBelong': {'P0001': ['P0001p001'], 'P0002': ['P0002p001'], 'P0003': ['P0003p001'],
                                         'P0004': ['P0004p001'], 'P0005': ['P0005p001'], 'P0006': ['P0006p001'],
                                         'P0007': ['P0007p001'], 'P0008': ['P0008p001'], 'P0009': ['P0009p001'],
                                         'P0010': ['P0010p001'],
                                         'P0011': ['P0011p001', 'P0011p002', 'P0011p003', 'P0011p004', 'P0011p005',
                                                   'P0011p006',
                                                   'P0011p007', 'P0011p008'],
                                         'P0012': ['P0012p001', 'P0012p002', 'P0012p003', 'P0012p004', 'P0012p005',
                                                   'P0012p006',
                                                   'P0012p007', 'P0012p008'],
                                         'P0013': ['P0013p001', 'P0013p002', 'P0013p003', 'P0013p004', 'P0013p005',
                                                   'P0013p006',
                                                   'P0013p007', 'P0013p008'],
                                         'P0014': ['P0014p001', 'P0014p002', 'P0014p003', 'P0014p004', 'P0014p005',
                                                   'P0014p006',
                                                   'P0014p007', 'P0014p008'],
                                         'P0015': ['P0015p001', 'P0015p002', 'P0015p003', 'P0015p004', 'P0015p005',
                                                   'P0015p006',
                                                   'P0015p007', 'P0015p008'],
                                         'P0016': ['P0016p001', 'P0016p002', 'P0016p003', 'P0016p004', 'P0016p005',
                                                   'P0016p006',
                                                   'P0016p007', 'P0016p008'],
                                         'P0017': ['P0017p001', 'P0017p002', 'P0017p003', 'P0017p004', 'P0017p005',
                                                   'P0017p006',
                                                   'P0017p007', 'P0017p008'],
                                         'P0018': ['P0018p001', 'P0018p002', 'P0018p003', 'P0018p004', 'P0018p005',
                                                   'P0018p006',
                                                   'P0018p007', 'P0018p008'],
                                         'P0019': ['P0019p001', 'P0019p002', 'P0019p003', 'P0019p004', 'P0019p005',
                                                   'P0019p006',
                                                   'P0019p007', 'P0019p008'],
                                         'P0020': ['P0020p001', 'P0020p002', 'P0020p003', 'P0020p004', 'P0020p005',
                                                   'P0020p006',
                                                   'P0020p007', 'P0020p008'],
                                         'P0021': ['P0021p001', 'P0021p002', 'P0021p003', 'P0021p004', 'P0021p005',
                                                   'P0021p006'],
                                         'P0022': ['P0022p001', 'P0022p002', 'P0022p003', 'P0022p004', 'P0022p005',
                                                   'P0022p006'],
                                         'P0023': ['P0023p001', 'P0023p002', 'P0023p003', 'P0023p004', 'P0023p005',
                                                   'P0023p006'],
                                         'P0024': ['P0024p001', 'P0024p002', 'P0024p003', 'P0024p004', 'P0024p005',
                                                   'P0024p006'],
                                         'P0025': ['P0025p001', 'P0025p002', 'P0025p003', 'P0025p004', 'P0025p005',
                                                   'P0025p006'],
                                         'P0026': ['P0026p001', 'P0026p002', 'P0026p003', 'P0026p004', 'P0026p005',
                                                   'P0026p006'],
                                         'P0027': ['P0027p001', 'P0027p002', 'P0027p003', 'P0027p004', 'P0027p005',
                                                   'P0027p006'],
                                         'P0028': ['P0028p001', 'P0028p002', 'P0028p003', 'P0028p004', 'P0028p005',
                                                   'P0028p006'],
                                         'P0029': ['P0029p001', 'P0029p002', 'P0029p003', 'P0029p004', 'P0029p005',
                                                   'P0029p006'],
                                         'P0030': ['P0030p001', 'P0030p002', 'P0030p003', 'P0030p004', 'P0030p005',
                                                   'P0030p006'], 'P0031': ['P0031p001'], 'P0032': ['P0032p001'],
                                         'P0033': ['P0033p001'], 'P0034': ['P0034p001'], 'P0035': ['P0035p001'],
                                         'P0036': ['P0036p001'], 'P0037': ['P0037p001'], 'P0038': ['P0038p001'],
                                         'P0039': ['P0039p001'], 'P0040': ['P0040p001'],
                                         'P0041': ['P0041p001', 'P0041p002', 'P0041p003', 'P0041p004', 'P0041p005',
                                                   'P0041p006',
                                                   'P0041p007'],
                                         'P0042': ['P0042p001', 'P0042p002', 'P0042p003', 'P0042p004', 'P0042p005',
                                                   'P0042p006',
                                                   'P0042p007'],
                                         'P0043': ['P0043p001', 'P0043p002', 'P0043p003', 'P0043p004', 'P0043p005',
                                                   'P0043p006',
                                                   'P0043p007'],
                                         'P0044': ['P0044p001', 'P0044p002', 'P0044p003', 'P0044p004', 'P0044p005',
                                                   'P0044p006',
                                                   'P0044p007'],
                                         'P0045': ['P0045p001', 'P0045p002', 'P0045p003', 'P0045p004', 'P0045p005',
                                                   'P0045p006',
                                                   'P0045p007'],
                                         'P0046': ['P0046p001', 'P0046p002', 'P0046p003', 'P0046p004', 'P0046p005',
                                                   'P0046p006',
                                                   'P0046p007'],
                                         'P0047': ['P0047p001', 'P0047p002', 'P0047p003', 'P0047p004', 'P0047p005',
                                                   'P0047p006',
                                                   'P0047p007'],
                                         'P0048': ['P0048p001', 'P0048p002', 'P0048p003', 'P0048p004', 'P0048p005',
                                                   'P0048p006',
                                                   'P0048p007'],
                                         'P0049': ['P0049p001', 'P0049p002', 'P0049p003', 'P0049p004', 'P0049p005',
                                                   'P0049p006',
                                                   'P0049p007'],
                                         'P0050': ['P0050p001', 'P0050p002', 'P0050p003', 'P0050p004', 'P0050p005',
                                                   'P0050p006',
                                                   'P0050p007'],
                                         'P0051': ['P0051p001', 'P0051p002', 'P0051p003', 'P0051p004', 'P0051p005',
                                                   'P0051p006',
                                                   'P0051p007'],
                                         'P0052': ['P0052p001', 'P0052p002', 'P0052p003', 'P0052p004', 'P0052p005',
                                                   'P0052p006',
                                                   'P0052p007'],
                                         'P0053': ['P0053p001', 'P0053p002', 'P0053p003', 'P0053p004', 'P0053p005',
                                                   'P0053p006',
                                                   'P0053p007'],
                                         'P0054': ['P0054p001', 'P0054p002', 'P0054p003', 'P0054p004', 'P0054p005',
                                                   'P0054p006',
                                                   'P0054p007'],
                                         'P0055': ['P0055p001', 'P0055p002', 'P0055p003', 'P0055p004', 'P0055p005',
                                                   'P0055p006',
                                                   'P0055p007'],
                                         'P0056': ['P0056p001', 'P0056p002', 'P0056p003', 'P0056p004', 'P0056p005',
                                                   'P0056p006',
                                                   'P0056p007'],
                                         'P0057': ['P0057p001', 'P0057p002', 'P0057p003', 'P0057p004', 'P0057p005',
                                                   'P0057p006',
                                                   'P0057p007'],
                                         'P0058': ['P0058p001', 'P0058p002', 'P0058p003', 'P0058p004', 'P0058p005',
                                                   'P0058p006',
                                                   'P0058p007'],
                                         'P0059': ['P0059p001', 'P0059p002', 'P0059p003', 'P0059p004', 'P0059p005',
                                                   'P0059p006',
                                                   'P0059p007'],
                                         'P0060': ['P0060p001', 'P0060p002', 'P0060p003', 'P0060p004', 'P0060p005',
                                                   'P0060p006',
                                                   'P0060p007'],
                                         'P0061': ['P0061p001', 'P0061p002', 'P0061p003', 'P0061p004', 'P0061p005',
                                                   'P0061p006',
                                                   'P0061p007'],
                                         'P0062': ['P0062p001', 'P0062p002', 'P0062p003', 'P0062p004', 'P0062p005',
                                                   'P0062p006',
                                                   'P0062p007'],
                                         'P0063': ['P0063p001', 'P0063p002', 'P0063p003', 'P0063p004', 'P0063p005',
                                                   'P0063p006',
                                                   'P0063p007'],
                                         'P0064': ['P0064p001', 'P0064p002', 'P0064p003', 'P0064p004', 'P0064p005',
                                                   'P0064p006',
                                                   'P0064p007'],
                                         'P0065': ['P0065p001', 'P0065p002', 'P0065p003', 'P0065p004', 'P0065p005',
                                                   'P0065p006',
                                                   'P0065p007'],
                                         'P0066': ['P0066p001', 'P0066p002', 'P0066p003', 'P0066p004', 'P0066p005',
                                                   'P0066p006',
                                                   'P0066p007'],
                                         'P0067': ['P0067p001', 'P0067p002', 'P0067p003', 'P0067p004', 'P0067p005',
                                                   'P0067p006',
                                                   'P0067p007'],
                                         'P0068': ['P0068p001', 'P0068p002', 'P0068p003', 'P0068p004', 'P0068p005',
                                                   'P0068p006',
                                                   'P0068p007'],
                                         'P0069': ['P0069p001', 'P0069p002', 'P0069p003', 'P0069p004', 'P0069p005',
                                                   'P0069p006',
                                                   'P0069p007'],
                                         'P0070': ['P0070p001', 'P0070p002', 'P0070p003', 'P0070p004', 'P0070p005',
                                                   'P0070p006',
                                                   'P0070p007'],
                                         'P0071': ['P0071p001', 'P0071p002', 'P0071p003', 'P0071p004', 'P0071p005',
                                                   'P0071p006',
                                                   'P0071p007'],
                                         'P0072': ['P0072p001', 'P0072p002', 'P0072p003', 'P0072p004', 'P0072p005',
                                                   'P0072p006',
                                                   'P0072p007'],
                                         'P0073': ['P0073p001', 'P0073p002', 'P0073p003', 'P0073p004', 'P0073p005',
                                                   'P0073p006',
                                                   'P0073p007'],
                                         'P0074': ['P0074p001', 'P0074p002', 'P0074p003', 'P0074p004', 'P0074p005',
                                                   'P0074p006',
                                                   'P0074p007'],
                                         'P0075': ['P0075p001', 'P0075p002', 'P0075p003', 'P0075p004', 'P0075p005',
                                                   'P0075p006',
                                                   'P0075p007'],
                                         'P0076': ['P0076p001', 'P0076p002', 'P0076p003', 'P0076p004', 'P0076p005',
                                                   'P0076p006',
                                                   'P0076p007'],
                                         'P0077': ['P0077p001', 'P0077p002', 'P0077p003', 'P0077p004', 'P0077p005',
                                                   'P0077p006',
                                                   'P0077p007'],
                                         'P0078': ['P0078p001', 'P0078p002', 'P0078p003', 'P0078p004', 'P0078p005',
                                                   'P0078p006',
                                                   'P0078p007'],
                                         'P0079': ['P0079p001', 'P0079p002', 'P0079p003', 'P0079p004', 'P0079p005',
                                                   'P0079p006',
                                                   'P0079p007'],
                                         'P0080': ['P0080p001', 'P0080p002', 'P0080p003', 'P0080p004', 'P0080p005',
                                                   'P0080p006',
                                                   'P0080p007'],
                                         'P0081': ['P0081p001', 'P0081p002', 'P0081p003', 'P0081p004', 'P0081p005',
                                                   'P0081p006',
                                                   'P0081p007', 'P0081p008', 'P0081p009', 'P0081p010'],
                                         'P0082': ['P0082p001', 'P0082p002', 'P0082p003', 'P0082p004', 'P0082p005',
                                                   'P0082p006',
                                                   'P0082p007', 'P0082p008', 'P0082p009', 'P0082p010'],
                                         'P0083': ['P0083p001', 'P0083p002', 'P0083p003', 'P0083p004', 'P0083p005',
                                                   'P0083p006',
                                                   'P0083p007', 'P0083p008', 'P0083p009', 'P0083p010'],
                                         'P0084': ['P0084p001', 'P0084p002', 'P0084p003', 'P0084p004', 'P0084p005',
                                                   'P0084p006',
                                                   'P0084p007', 'P0084p008', 'P0084p009', 'P0084p010'],
                                         'P0085': ['P0085p001', 'P0085p002', 'P0085p003', 'P0085p004', 'P0085p005',
                                                   'P0085p006',
                                                   'P0085p007', 'P0085p008', 'P0085p009', 'P0085p010'],
                                         'P0086': ['P0086p001', 'P0086p002', 'P0086p003', 'P0086p004', 'P0086p005',
                                                   'P0086p006',
                                                   'P0086p007', 'P0086p008', 'P0086p009', 'P0086p010'],
                                         'P0087': ['P0087p001', 'P0087p002', 'P0087p003', 'P0087p004', 'P0087p005',
                                                   'P0087p006',
                                                   'P0087p007', 'P0087p008', 'P0087p009', 'P0087p010'],
                                         'P0088': ['P0088p001', 'P0088p002', 'P0088p003', 'P0088p004', 'P0088p005',
                                                   'P0088p006',
                                                   'P0088p007', 'P0088p008', 'P0088p009', 'P0088p010'],
                                         'P0089': ['P0089p001', 'P0089p002', 'P0089p003', 'P0089p004', 'P0089p005',
                                                   'P0089p006',
                                                   'P0089p007', 'P0089p008', 'P0089p009', 'P0089p010'],
                                         'P0090': ['P0090p001', 'P0090p002', 'P0090p003', 'P0090p004', 'P0090p005',
                                                   'P0090p006',
                                                   'P0090p007', 'P0090p008', 'P0090p009', 'P0090p010'],
                                         'P0091': ['P0091p001', 'P0091p002', 'P0091p003', 'P0091p004', 'P0091p005',
                                                   'P0091p006',
                                                   'P0091p007', 'P0091p008', 'P0091p009', 'P0091p010'],
                                         'P0092': ['P0092p001', 'P0092p002', 'P0092p003', 'P0092p004', 'P0092p005',
                                                   'P0092p006',
                                                   'P0092p007', 'P0092p008', 'P0092p009', 'P0092p010'],
                                         'P0093': ['P0093p001', 'P0093p002', 'P0093p003', 'P0093p004', 'P0093p005',
                                                   'P0093p006',
                                                   'P0093p007', 'P0093p008', 'P0093p009', 'P0093p010'],
                                         'P0094': ['P0094p001', 'P0094p002', 'P0094p003', 'P0094p004', 'P0094p005',
                                                   'P0094p006',
                                                   'P0094p007', 'P0094p008', 'P0094p009', 'P0094p010'],
                                         'P0095': ['P0095p001', 'P0095p002', 'P0095p003', 'P0095p004', 'P0095p005',
                                                   'P0095p006',
                                                   'P0095p007', 'P0095p008', 'P0095p009', 'P0095p010'],
                                         'P0096': ['P0096p001', 'P0096p002', 'P0096p003', 'P0096p004', 'P0096p005',
                                                   'P0096p006',
                                                   'P0096p007', 'P0096p008', 'P0096p009', 'P0096p010'],
                                         'P0097': ['P0097p001', 'P0097p002', 'P0097p003', 'P0097p004', 'P0097p005',
                                                   'P0097p006',
                                                   'P0097p007', 'P0097p008', 'P0097p009', 'P0097p010'],
                                         'P0098': ['P0098p001', 'P0098p002', 'P0098p003', 'P0098p004', 'P0098p005',
                                                   'P0098p006',
                                                   'P0098p007', 'P0098p008', 'P0098p009', 'P0098p010'],
                                         'P0099': ['P0099p001', 'P0099p002', 'P0099p003', 'P0099p004', 'P0099p005',
                                                   'P0099p006',
                                                   'P0099p007', 'P0099p008', 'P0099p009', 'P0099p010'],
                                         'P0100': ['P0100p001', 'P0100p002', 'P0100p003', 'P0100p004', 'P0100p005',
                                                   'P0100p006',
                                                   'P0100p007', 'P0100p008', 'P0100p009', 'P0100p010']},
                'productDateEnd': {'P0001': '2022-06-08 00:00:00',
                                   'P0002': '2022-06-08 00:00:00',
                                   'P0003': '2022-06-08 00:00:00',
                                   'P0004': '2022-06-08 00:00:00',
                                   'P0005': '2022-06-08 00:00:00',
                                   'P0006': '2022-06-08 00:00:00',
                                   'P0007': '2022-06-08 00:00:00',
                                   'P0008': '2022-06-08 00:00:00',
                                   'P0009': '2022-06-08 00:00:00',
                                   'P0010': '2022-06-08 00:00:00',
                                   'P0011': '2022-06-08 00:00:00',
                                   'P0012': '2022-06-08 00:00:00',
                                   'P0013': '2022-06-08 00:00:00',
                                   'P0014': '2022-06-08 00:00:00',
                                   'P0015': '2022-06-08 00:00:00',
                                   'P0016': '2022-06-08 00:00:00',
                                   'P0017': '2022-06-08 00:00:00',
                                   'P0018': '2022-06-08 00:00:00',
                                   'P0019': '2022-06-08 00:00:00',
                                   'P0020': '2022-06-08 00:00:00',
                                   'P0021': '2022-06-08 00:00:00',
                                   'P0022': '2022-06-08 00:00:00',
                                   'P0023': '2022-06-08 00:00:00',
                                   'P0024': '2022-06-08 00:00:00',
                                   'P0025': '2022-06-08 00:00:00',
                                   'P0026': '2022-06-08 00:00:00',
                                   'P0027': '2022-06-08 00:00:00',
                                   'P0028': '2022-06-08 00:00:00',
                                   'P0029': '2022-06-08 00:00:00',
                                   'P0030': '2022-06-08 00:00:00',
                                   'P0031': '2022-06-08 00:00:00',
                                   'P0032': '2022-06-08 00:00:00',
                                   'P0033': '2022-06-08 00:00:00',
                                   'P0034': '2022-06-08 00:00:00',
                                   'P0035': '2022-06-08 00:00:00',
                                   'P0036': '2022-06-08 00:00:00',
                                   'P0037': '2022-06-08 00:00:00',
                                   'P0038': '2022-06-08 00:00:00',
                                   'P0039': '2022-06-08 00:00:00',
                                   'P0040': '2022-06-08 00:00:00',
                                   'P0041': '2022-06-08 00:00:00',
                                   'P0042': '2022-06-08 00:00:00',
                                   'P0043': '2022-06-08 00:00:00',
                                   'P0044': '2022-06-08 00:00:00',
                                   'P0045': '2022-06-08 00:00:00',
                                   'P0046': '2022-06-08 00:00:00',
                                   'P0047': '2022-06-08 00:00:00',
                                   'P0048': '2022-06-08 00:00:00',
                                   'P0049': '2022-06-08 00:00:00',
                                   'P0050': '2022-06-08 00:00:00',
                                   'P0051': '2022-06-08 00:00:00',
                                   'P0052': '2022-06-08 00:00:00',
                                   'P0053': '2022-06-08 00:00:00',
                                   'P0054': '2022-06-08 00:00:00',
                                   'P0055': '2022-06-08 00:00:00',
                                   'P0056': '2022-06-08 00:00:00',
                                   'P0057': '2022-06-08 00:00:00',
                                   'P0058': '2022-06-08 00:00:00',
                                   'P0059': '2022-06-08 00:00:00',
                                   'P0060': '2022-06-08 00:00:00',
                                   'P0061': '2022-06-08 00:00:00',
                                   'P0062': '2022-06-08 00:00:00',
                                   'P0063': '2022-06-08 00:00:00',
                                   'P0064': '2022-06-08 00:00:00',
                                   'P0065': '2022-06-08 00:00:00',
                                   'P0066': '2022-06-08 00:00:00',
                                   'P0067': '2022-06-08 00:00:00',
                                   'P0068': '2022-06-08 00:00:00',
                                   'P0069': '2022-06-08 00:00:00',
                                   'P0070': '2022-06-08 00:00:00',
                                   'P0071': '2022-06-08 00:00:00',
                                   'P0072': '2022-06-08 00:00:00',
                                   'P0073': '2022-06-08 00:00:00',
                                   'P0074': '2022-06-08 00:00:00',
                                   'P0075': '2022-06-08 00:00:00',
                                   'P0076': '2022-06-08 00:00:00',
                                   'P0077': '2022-06-08 00:00:00',
                                   'P0078': '2022-06-08 00:00:00',
                                   'P0079': '2022-06-08 00:00:00',
                                   'P0080': '2022-06-08 00:00:00',
                                   'P0081': '2022-06-08 00:00:00',
                                   'P0082': '2022-06-08 00:00:00',
                                   'P0083': '2022-06-08 00:00:00',
                                   'P0084': '2022-06-08 00:00:00',
                                   'P0085': '2022-06-08 00:00:00',
                                   'P0086': '2022-06-08 00:00:00',
                                   'P0087': '2022-06-08 00:00:00',
                                   'P0088': '2022-06-08 00:00:00',
                                   'P0089': '2022-06-08 00:00:00',
                                   'P0090': '2022-06-08 00:00:00',
                                   'P0091': '2022-06-08 00:00:00',
                                   'P0092': '2022-06-08 00:00:00',
                                   'P0093': '2022-06-08 00:00:00',
                                   'P0094': '2022-06-08 00:00:00',
                                   'P0095': '2022-06-08 00:00:00',
                                   'P0096': '2022-06-08 00:00:00',
                                   'P0097': '2022-06-08 00:00:00',
                                   'P0098': '2022-06-08 00:00:00',
                                   'P0099': '2022-06-08 00:00:00',
                                   'P0100': '2022-06-08 00:00:00'}
                }

    # 输入排产周期起始时刻、时长、每日休息时段
    def information(data):
        planstart = data['planStart']
        planstart = dt.datetime.strptime(planstart, "%Y-%m-%d %H:%M:%S")
        span = data['periodLength']
        planspan = dt.timedelta(days=span)
        planend = planstart + planspan
        reststart = '22:00'  # 暂定为22:00
        [a, b] = reststart.split(':')
        a = int(a)
        b = int(b)
        reststart = dt.timedelta(minutes=a * 60 + b)
        restend = '24:00'  # 暂定为24:00
        [a, b] = restend.split(':')
        a = int(a)
        b = int(b)
        restend = dt.timedelta(minutes=a * 60 + b)
        # 输出每日工作时间段
        restduration = []
        for i in range(0, span):
            a = planstart + dt.timedelta(days=i) + reststart
            b = planstart + dt.timedelta(days=i) + restend
            restduration.append([a, b])
        T = data['replanTime']
        T = dt.datetime.strptime(T, "%Y-%m-%d %H:%M:%S")
        for i in range(0, span):  # 若插单时刻发生在休息时段，将其移至该休息时段末尾
            if (restduration[i][0] <= T) & (restduration[i][1] >= T):
                T = restduration[i][1]
        return planstart, planend, planspan, restduration, T

    # 按时间顺序对Q和QQ重排
    def adjust(Q, QQ):
        for key in Q.keys():
            if len(Q[key]) <= 1:
                continue
            for k in range(0, len(Q[key]) - 1):
                for j in range(k + 1, len(Q[key])):
                    a = dt.datetime.strptime(Q[key][k][0], "%Y-%m-%d %H:%M:%S")
                    b = dt.datetime.strptime(Q[key][j][0], "%Y-%m-%d %H:%M:%S")
                    if a > b:
                        temp1 = Q[key][k]
                        Q[key][k] = Q[key][j]
                        Q[key][j] = temp1
                        temp2 = QQ[key][k]
                        QQ[key][k] = QQ[key][j]
                        QQ[key][j] = temp2
        return Q, QQ

    def urgentinformation(urgent, J):
        # n=[工序编号‘PXXXXpXXX’,[可选设备列表]，[加工时间列表]，[机器优先级列表]]
        NN = []
        for key in urgent.keys():
            N = []
            specificProcessID = urgent[key]['specificProcessID']
            machineID = urgent[key]['machineID']
            processTime = urgent[key]['processTime']
            machinePriority = urgent[key]['machinePriority']
            J[key] = urgent[key]['specificProcessID']  # 将插单的归属关系也放入到J中
            # orderlevel=urgent[key]['orderLevel']#暂时不管
            for i in range(len(machineID)):
                n = []
                n.append(specificProcessID[i])
                n.append(machineID[i])
                n.append(processTime[i])
                n.append(machinePriority[i])
                N.append(n)
            NN.append(N)
        return NN

    # 计算设备从紧急插单点到排产周期末的占用时间
    def caculation(T, Q, planend):  # T为发生紧急插单的时刻点，Q为原排产计划的加工时间集合
        t = {}
        for key in Q.keys():
            if Q[key] == []:  # case1
                t[key] = dt.timedelta(minutes=0)
            else:
                between = 0
                for j in range(0, len(Q[key])):
                    start = dt.datetime.strptime(Q[key][j][0], "%Y-%m-%d %H:%M:%S")
                    end = dt.datetime.strptime(Q[key][j][1], "%Y-%m-%d %H:%M:%S")
                    if (start <= T) & (end >= T):  # case2订单插入时设备在加工
                        t1 = end - T
                        # print(t1)#正在加工工序还需占用的设备时间
                        t2 = dt.timedelta(minutes=0)
                        for k in range(j + 1, len(Q[key])):
                            start1 = dt.datetime.strptime(Q[key][k][0], "%Y-%m-%d %H:%M:%S")
                            end1 = dt.datetime.strptime(Q[key][k][1], "%Y-%m-%d %H:%M:%S")
                            if end1 <= planend:
                                t2 = t2 + end1 - start1  # 重拍后超出24小时排产周期的不应当计入设备占用时间
                            elif (start1 <= planend) & (end1 > planend):  # 对于生产时间跨越排产周期结点的情况
                                t2 = t2 + planend - start1
                            else:
                                break
                        t0 = t1 + t2
                        t[key] = t0
                        between = 1
                        break
                if between == 0:
                    # 设置一个临时变量busy，对紧急插单时刻原排产计划未安排工序的设为0
                    busy = 0
                    for j in range(0, len(Q[key])):
                        start = dt.datetime.strptime(Q[key][j][0], "%Y-%m-%d %H:%M:%S")
                        if start > T:
                            t2 = dt.timedelta(minutes=0)
                            for k in range(j, len(Q[key])):
                                start1 = dt.datetime.strptime(Q[key][k][0], "%Y-%m-%d %H:%M:%S")
                                end1 = dt.datetime.strptime(Q[key][k][1], "%Y-%m-%d %H:%M:%S")
                                if end1 <= planend:
                                    t2 = t2 + end1 - start1  # 重排后超出24小时排产周期的不应当计入设备占用时间
                                else:
                                    t2 = t2 + planend - start1
                                    break
                            t[key] = t2
                            # print(t2)
                            busy = 1
                            break
                    if busy == 0:
                        t[key] = dt.timedelta(minutes=0)
        return t

    # 更新插入工序后的设备占用时间
    def rearrange(T, Q, a, CQ, N,
                  QQ):  # TT当前时刻，Q当前加工时间集合，a选中设备编号，记录插入的工序在集合N中的位置索引，N待加工工序相关信息（这里主要用到工序耗时），QQ当前加工工序名称顺序集合
        # print("当前插单工序（粗排）：", N[CQ[0]][CQ[1]][0])
        CQT = []
        between = 0
        if Q[a] == []:  # 对原排产计划上没有安排工序的情况
            CQT.append(T.strftime("%Y-%m-%d %H:%M:%S"))
            CQT.append((T + dt.timedelta(seconds=round(N[CQ[0]][CQ[1]][2][N[CQ[0]][CQ[1]][1].index(a)]))).strftime(
                "%Y-%m-%d %H:%M:%S"))  # 添加round保证不会出现微秒的情况
            Q[a].append(CQT)
            QQ[a].append(N[CQ[0]][CQ[1]][0])
        else:
            for i in range(0, len(Q[a])):
                start = dt.datetime.strptime(Q[a][i][0], "%Y-%m-%d %H:%M:%S")
                end = dt.datetime.strptime(Q[a][i][1], "%Y-%m-%d %H:%M:%S")
                if (start <= T) & (end >= T):  # 订单插入时设备在加工
                    CQT.append(end.strftime("%Y-%m-%d %H:%M:%S"))  # 正常来说这里应该要加上准备时间
                    CQT.append(
                        (end + dt.timedelta(seconds=round(N[CQ[0]][CQ[1]][2][N[CQ[0]][CQ[1]][1].index(a)]))).strftime(
                            "%Y-%m-%d %H:%M:%S"))
                    # 在这之后的工序加工时间顺延
                    for k in range(i + 1, len(Q[a])):
                        start1 = dt.datetime.strptime(Q[a][k][0], "%Y-%m-%d %H:%M:%S")
                        end1 = dt.datetime.strptime(Q[a][k][1], "%Y-%m-%d %H:%M:%S")
                        restart = start1 + dt.timedelta(
                            seconds=round(N[CQ[0]][CQ[1]][2][N[CQ[0]][CQ[1]][1].index(a)]))  # 正常来说这里应该加上插入工序后的设备整理时间
                        reend = end1 + dt.timedelta(seconds=round(N[CQ[0]][CQ[1]][2][N[CQ[0]][CQ[1]][1].index(a)]))
                        Q[a][k][0] = restart.strftime("%Y-%m-%d %H:%M:%S")
                        Q[a][k][1] = reend.strftime("%Y-%m-%d %H:%M:%S")
                    Q[a].append(CQT)
                    QQ[a].append(N[CQ[0]][CQ[1]][0])
                    between = 1
                    break
            if between == 0:
                busy = 0  # 临时变量，若当前紧急插单时刻在所有已安排工序的开始时间和结束时间之后（此时很有可能紧急插单时刻已经超出排产周期），则busy=0
                for j in range(0, len(Q[a])):
                    start = dt.datetime.strptime(Q[a][j][0], "%Y-%m-%d %H:%M:%S")
                    if start > T:
                        CQT.append(T.strftime("%Y-%m-%d %H:%M:%S"))  # 正常来说这里应该要加上准备时间
                        CQT.append(
                            (T + dt.timedelta(seconds=round(N[CQ[0]][CQ[1]][2][N[CQ[0]][CQ[1]][1].index(a)]))).strftime(
                                "%Y-%m-%d %H:%M:%S"))
                        for k in range(j, len(Q[a])):
                            start2 = dt.datetime.strptime(Q[a][k][0], "%Y-%m-%d %H:%M:%S")
                            end2 = dt.datetime.strptime(Q[a][k][1], "%Y-%m-%d %H:%M:%S")
                            restart = start2 + dt.timedelta(
                                seconds=round(
                                    N[CQ[0]][CQ[1]][2][N[CQ[0]][CQ[1]][1].index(a)]))  # 正常来说这里应该加上插入工序后的设备整理时间
                            reend = end2 + dt.timedelta(seconds=round(N[CQ[0]][CQ[1]][2][N[CQ[0]][CQ[1]][1].index(a)]))
                            Q[a][k][0] = restart.strftime("%Y-%m-%d %H:%M:%S")
                            Q[a][k][1] = reend.strftime("%Y-%m-%d %H:%M:%S")
                        Q[a].append(CQT)
                        QQ[a].append(N[CQ[0]][CQ[1]][0])
                        busy = 1
                        break
                if busy == 0:
                    CQT.append(T.strftime("%Y-%m-%d %H:%M:%S"))
                    CQT.append(
                        (T + dt.timedelta(seconds=round(N[CQ[0]][CQ[1]][2][N[CQ[0]][CQ[1]][1].index(a)]))).strftime(
                            "%Y-%m-%d %H:%M:%S"))
                    Q[a].append(CQT)
                    QQ[a].append(N[CQ[0]][CQ[1]][0])
        return Q, QQ, CQT

    # 对重拍后引起的后一工序的开始时间超过前一工序结束时间的情况进行调整
    def renew(Q, QQ, J, N, restduration, T, planstart, planend, productDateEnd):  # Q为粗排的加工时间集合，QQ为粗排的加工工序集合
        QW = []  # 按设备分类的待加工工序集合
        # QK=[]#不区分设备的待加工工序集合
        QQW = []  # 按设备分类的待加工工序名称集合
        QQ2 = []  # QQW备份，检查环节使用
        Q1 = []  # 存储插单时刻前的工序时间
        QQ1 = []  # 存储插单时刻前的工序名称
        NW = []  # 按(具体)产品编号分类的待加工工序集合
        for key in Q.keys():
            qw = []
            qqw = []
            qq2 = []
            if Q[key] == []:  # 若某一设备未安排工序，则插单时刻后的加工工序、时间集合为空
                q1 = []
                qw = []
                qqw = []
                qq2 = []
                qq1 = []
                Q1.append(q1)
                QW.append(qw)
                QQW.append(qqw)
                QQ2.append(qq2)
                QQ1.append(qq1)
            else:
                q1 = []
                qq1 = []
                for j in range(0, len(Q[key])):
                    start = dt.datetime.strptime(Q[key][j][0], "%Y-%m-%d %H:%M:%S")
                    if start >= T:
                        qw.append(Q[key][j])
                        qqw.append(QQ[key][j])
                        qq2.append(QQ[key][j])
                    else:
                        q1.append(Q[key][j])
                        qq1.append(QQ[key][j])
                Q1.append(q1)
                QQ1.append(qq1)
                QQ2.append(qq2)
                QW.append(qw)
                QQW.append(qqw)
        '''
        print("插单时刻前生产计划部分：", Q1)
        print("插单时刻前生产计划部分（名称）", QQ1)
        '''
        for key in J.keys():
            nw = []
            for j in range(0, len(J[key])):
                for k in range(0, len(QQW)):
                    if J[key][j] in QQW[k]:
                        nw.append(J[key][j])
            if nw != []:
                NW.append(nw)
        # QK.extend(qqw)
        for i in range(0, len(N)):
            nw = []
            for j in range(0, len(N[i])):
                for k in range(0, len(QQW)):
                    if N[i][j][0] in QQW[k]:
                        nw.append(N[i][j][0])
            NW.append(nw)
        print("插单时刻后的产品待加工工序名称集合（按产品分类）：", NW)
        print("插单时刻后的产品待加工工序名称集合（按设备分类）：", QQW)
        print("插单时刻后的产品待加工工序时间集合（按设备分类）：", QW)
        print("插单时刻后工序名称：", QQ2)
        # 初始化设备空闲时间（可以开始加工的最早时间）
        TW = []
        for i in range(0, len(QW)):
            if QW[i] == []:  # 若某一设备未安排工序，设置工序的最早加工时间这一排产周期的起点
                TW.append(planstart)
            else:
                TW.append(QW[i][0][0])
        # 初始化产品上一工序时间
        l = len(NW)
        TTW = [T.strftime("%Y-%m-%d %H:%M:%S") for _ in range(l)]  # 初始化时间为紧急插单的时刻
        # 正式更新排产计划
        z = len(QQW)
        finalQ = [[] for _ in range(z)]  # 用来存储最后各设备各工序的加工时间
        while (QQW != [[] for _ in range(z)]):
            for i in range(0, len(QQW)):
                j = 0
                while (j < len(NW)):  # 如果产品集合未遍历完
                    if QQW[i] != []:
                        if NW[j] != []:
                            if QQW[i][0] == NW[j][0]:
                                print("当前排产工序", QQW[i][0])
                                tcost = dt.datetime.strptime(QW[i][0][1],
                                                             "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(
                                    QW[i][0][0], "%Y-%m-%d %H:%M:%S")
                                a = dt.datetime.strptime(TW[i], "%Y-%m-%d %H:%M:%S")  # 设备最早可加工时间
                                b = dt.datetime.strptime(TTW[j], "%Y-%m-%d %H:%M:%S")  # 上一工序结束时间
                                if a <= b:
                                    temp1 = b
                                else:
                                    temp1 = a
                                # 判断当前安排的工序时间是否侵占了休息时间
                                x = temp1 + tcost
                                for p in range(0, len(restduration)):
                                    if restduration[p][0].day == temp1.day:
                                        if (temp1 > restduration[p][0]) & (
                                                temp1 < restduration[p][1]):  # 休息时段内才开始的，一律推迟到休息时段末
                                            temp1 = restduration[p][1]
                                            x = temp1 + tcost
                                        elif (temp1 <= restduration[p][0]) & (
                                                x > restduration[p][1]):  # 休息时段前开始，但耗完休息时段仍为加工完的，推迟至休息时段末
                                            temp1 = restduration[p][1]
                                            x = temp1 + tcost
                                        break
                                QW[i][0][0] = temp1.strftime("%Y-%m-%d %H:%M:%S")
                                QW[i][0][1] = x.strftime("%Y-%m-%d %H:%M:%S")
                                TW[i] = QW[i][0][1]  # 更新设备的最早可加工时间
                                TTW[j] = QW[i][0][1]  # 更新产品上一工序结束时间
                                finalQ[i].append(QW[i][0])  # 将安排好的工序的开始时间和结束时间放入最终顺序中
                                del QQW[i][0]
                                del QW[i][0]
                                del NW[j][0]
                                print(QQW)
                                print(QW)
                                print(NW)
                                print(finalQ)
                                j = 0  # 如果找到某一件产品的当前最前工序为当前设备的最前工序，那么当前设备的第二个工序成为最前工序，同样需要对所有产品种类进行遍历，重置产品序号为1
                            else:
                                j = j + 1  # 如果当前产品的最前工序与当前设备的最前工序不同，产品序号加1，判断下一产品的最前工序是否为当前设备的最前工序
                        else:
                            j = j + 1  # 如果当前产品的工序已经安排完，则去比对下一产品的最前工序
                    else:
                        break  # 如果某一设备的工序已被安排完，则安排下一个设备的工序
        print("插单时刻后工序名称：", QQ2)

        # 按产品检查，能否将某些工序移动至设备时间线的空闲处
        for i in range(0, len(finalQ)):
            if finalQ[i] != []:
                for j in range(1, len(finalQ[i])):  # 从设备安排的第二道工序开始，第一道工序已为最前，无法调整
                    for key_3 in J.keys():
                        if QQ2[i][j] in J[key_3]:
                            break
                    aaa = key_3
                    bbb = seperate(aaa, QQ2[i][j])
                    print("当前检查工序：", QQ2[i][j])
                    if J[aaa].index(QQ2[i][j]) != 0:
                        ab = J[aaa][J[aaa].index(QQ2[i][j]) - 1]
                        print("紧前工序名称", ab)
                        Qspan = dt.datetime.strptime(finalQ[i][j][1],
                                                     "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(
                            finalQ[i][j][0], "%Y-%m-%d %H:%M:%S")
                        print("工序耗时", Qspan)
                        lastend = '0'
                        find = '0'
                        for m in range(0, len(QQ2)):  #
                            for n in range(0, len(QQ2[m])):
                                print(m)
                                print(n)
                                if QQ2[m][n] == ab:
                                    lastend = dt.datetime.strptime(finalQ[m][n][1],
                                                                   "%Y-%m-%d %H:%M:%S")  # 记录紧前工序的结束加工时间，从这个时刻点开始向后检查空当
                                    print("紧前工序结束时间", lastend)
                                    find = '1'
                                    break
                            if (find == '1'):
                                break
                        if (lastend != '0'):  # 找到紧前工序
                            for jj in range(0, j):
                                c = dt.datetime.strptime(finalQ[i][jj][1], "%Y-%m-%d %H:%M:%S")
                                print("空当检查工序", QQ2[i][jj])
                                if c >= lastend:  # 只对该设备该工序前、紧前工序结束时刻后的工序进行检查
                                    cspan = dt.datetime.strptime(finalQ[i][jj + 1][0],
                                                                 "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(
                                        finalQ[i][jj][1], "%Y-%m-%d %H:%M:%S")
                                    print("空当起始工序", QQ2[i][jj])
                                    print("空当时间", cspan)
                                    if cspan >= Qspan:
                                        print("有空位，可前移", QQ2[i][j])
                                        finalQ[i][j][0] = finalQ[i][jj][1]
                                        cc = dt.datetime.strptime(finalQ[i][j][0], "%Y-%m-%d %H:%M:%S") + Qspan
                                        finalQ[i][j][1] = cc.strftime("%Y-%m-%d %H:%M:%S")
                                        TEMP1 = finalQ[i][j][0]
                                        TEMP2 = finalQ[i][j][1]
                                        TEMP = QQ2[i][j]
                                        for kk in range(j - 1, jj, -1):  # 将该工序移动至空当
                                            a = finalQ[i][kk][0]
                                            b = finalQ[i][kk][1]
                                            finalQ[i][kk + 1][0] = a
                                            finalQ[i][kk + 1][1] = b
                                            QQ2[i][kk + 1] = QQ2[i][kk]
                                        finalQ[i][jj + 1][0] = TEMP1
                                        finalQ[i][jj + 1][1] = TEMP2
                                        QQ2[i][jj + 1] = TEMP
                                        print("移动后的工序顺序", QQ2[i])
                                        print("移动后的工序时间集合", finalQ[i])
                                        break
        # 输出
        for i in range(len(Q1)):
            Q1[i].extend(finalQ[i])
        print("重排完成后的加工时间集合", Q1)
        for i in range(len(QQ1)):
            QQ1[i].extend(QQ2[i])
        print("重排完成后的工序名称集合", QQ1)

        # 反馈无法在交付日期前完成的产品件号信息
        unableDelieverOnTime = {}  # 存储不能按时交付产品件号、原定交货时间、当前安排计划下的完工时间
        for key in productDateEnd.keys():
            finalprocessID = J[key][-1]  # 最后一道工序
            for i in range(len(QQ1)):
                if finalprocessID in QQ1[i]:
                    finalprocessfinishtime = dt.datetime.strptime(Q1[i][QQ1[i].index(finalprocessID)][-1],
                                                                  "%Y-%m-%d %H:%M:%S")  # 当前安排计划下的完工时间
                    dateend = dt.datetime.strptime(productDateEnd[key], "%Y-%m-%d %H:%M:%S")  # 原定交货时间
                    if finalprocessfinishtime > dateend:
                        unableDelieverOnTime[key] = {}
                        unableDelieverOnTime[key]['dateEnd'] = productDateEnd[key]
                        unableDelieverOnTime[key]['planedFinishTime'] = Q1[i][QQ1[i].index(finalprocessID)][-1]
                    break
        '''
            # 输出超出排产周期的工序名称及下一阶段设备的最早可加工时间
        TNEXT = [dt.timedelta(minutes=0) for i in range(z)]  # 存储下一阶段设备的最早可加工时间
        QNEXT = [[] for _ in range(z)]  # 存储推迟到下一排产周期的工序
        for i in range(0, len(Q1)):
            print("当前设备", i)
            removeQ = []
            removeQQ = []  # 储存需要移除的工序时间及名称
            if Q1[i] != []:
                t = dt.timedelta(minutes=0)
                for j in range(0, len(Q1[i])):
                    a = dt.datetime.strptime(Q1[i][j][0], "%Y-%m-%d %H:%M:%S")
                    b = dt.datetime.strptime(Q1[i][j][1], "%Y-%m-%d %H:%M:%S")
                    if a >= planend:
                        QNEXT[i].append(QQ1[i][j])
                        removeQ.append(Q1[i][j])
                        removeQQ.append(QQ1[i][j])
                    if (a < planend) & (b > planend):
                        t = t + b - planend
                TNEXT[i] = t
                for j in range(0, len(removeQ)):
                    print(j)
                    Q1[i].remove(removeQ[j])
                    QQ1[i].remove(removeQQ[j])
        '''

        return Q1, QQ1, unableDelieverOnTime

    # 分离计划中specificprocessID包含的productID和processID
    def seperate(a, b):
        if b.startswith(a):
            return b.replace(a, '', 1)

    # 判断specificprocessID（b）包含的productID是否为productID（a）
    def containjudge(a, b):
        if b.startswith(a):
            return True

    def home():
        planstart, planend, planspan, restduration, T = information(data)
        QQ = data['pendingProcessMachine']
        Q = data['pendingProcessOriginalPlan']
        J = data['processProductBelong']
        productDateEnd = data['productDateEnd']
        # QQ, Q, J, data = planinput(planstart)
        Q, QQ = adjust(Q, QQ)

        urgent = data['insertOrder']
        #将插入订单的交付时间也更新到prcductDateEnd中
        for key in urgent.keys():
            productDateEnd[key]=urgent[key]['dateEnd']
        N = urgentinformation(urgent, J)  # [[[工序编号‘PXXXXpXXX’,[可选设备列表]，[加工时间列表]，[机器优先级列表]],...],...]
        for i in range(0, len(N)):
            TT = T
            for j in range(0, len(N[i])):
                t = caculation(TT, Q, planend)
                lasta = 'M01'  # 初始化临时变量，记录上一次选中的设备编号
                lastt = planend - T  # 初始化临时变量，记录上一次选中的设备的占用时间为全部被占用
                for k in N[i][j][1]:  # 选设备
                    if t[k] <= lastt:
                        lasta = k
                        lastt = t[k]
                CQ = [i, j]  # 记录插入的工序在集合N中的位置索引
                Q, QQ, CQT = rearrange(TT, Q, lasta, CQ, N,
                                       QQ)  # TT当前时刻，Q当前加工时间集合，a选中设备下标，CQ记录插入的工序在集合N中的位置索引，N待加工工序相关信息（这里主要用到工序耗时），QQ当前加工工序名称顺序集合
                print('工序%s的插入时刻%s' % (N[i][j][0], CQT))
                TT = dt.datetime.strptime(CQT[1], "%Y-%m-%d %H:%M:%S")  # 更新当前时间
                Q, QQ = adjust(Q, QQ)
        finalQ, finalQQ,unableDelieverOnTime= renew(Q, QQ, J, N, restduration, T, planstart, planend,productDateEnd)  # finalQ加工时间集合，finalQQ加工名称集合
        replanProcessName = {}
        replanProcessTime = {}
        j = -1
        for key in Q.keys():
            j = j + 1
            replanProcessTime[key] = finalQ[j]
            replanProcessName[key] = finalQQ[j]
        respond = {'replanProcessName': replanProcessName,
                   'replanProcessTime': replanProcessTime,
                   'replanProcessOthers': {},
                   'unableDelieverOnTime':unableDelieverOnTime}  # 暂时定为空集
        return json.dumps(respond)
    return home()

# ##############################################
# 紧急插单结束
# ###################################################
# 紧急撤单部分
##########################################
@app.route('/revoke', methods=['GET', 'POST'])
def revoke():
    if request.method == 'POST':
        data = request.get_data()
        data = json.loads(data)
    else:
        data = {'hours': 22,
                'periodLength': 7,
                'planStart': '2022-06-01 00:00:00',
                'process': {'P0001': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0002': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0003': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0004': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0005': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0006': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0007': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0008': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0009': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0010': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0011': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0012': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0013': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0014': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0015': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0016': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0017': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0018': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0019': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0020': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0021': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0022': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0023': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0024': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0025': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0026': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0027': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0028': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0029': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0030': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0031': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0032': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0033': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0034': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0035': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0036': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0037': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0038': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0039': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0040': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0041': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0042': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0043': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0044': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0045': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0046': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0047': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0048': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0049': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0050': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0051': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0052': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0053': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0054': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0055': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0056': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0057': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0058': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0059': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0060': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0061': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0062': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0063': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0064': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0065': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0066': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0067': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0068': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0069': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0070': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0071': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0072': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0073': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0074': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0075': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0076': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0077': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0078': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0079': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0080': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0081': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0082': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0083': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0084': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0085': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0086': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0087': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0088': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0089': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0090': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0091': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0092': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0093': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0094': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0095': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0096': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0097': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0098': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0099': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0100': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]}},
                'replanTime': '2022-06-02 00:00:00',
                'pendingProcessMachine': {'M01': [],
                                     'M02': [],
                                     'M03': [],
                                     'M04': [],
                                     'M05': [],
                                     'M06': ['P0011p001',
                                             'P0012p001',
                                             'P0013p001',
                                             'P0014p001',
                                             'P0015p001',
                                             'P0016p001',
                                             'P0017p001',
                                             'P0041p001',
                                             'P0042p001',
                                             'P0043p001',
                                             'P0044p001',
                                             'P0045p001',
                                             'P0046p001',
                                             'P0047p001',
                                             'P0048p001'],
                                     'M07': ['P0051p006',
                                             'P0051p007',
                                             'P0052p006',
                                             'P0052p007',
                                             'P0061p006',
                                             'P0061p007',
                                             'P0071p001',
                                             'P0072p001',
                                             'P0073p001',
                                             'P0074p001',
                                             'P0075p001',
                                             'P0076p001',
                                             'P0081p004',
                                             'P0081p010',
                                             'P0082p004',
                                             'P0082p010',
                                             'P0083p004',
                                             'P0084p004',
                                             'P0085p004',
                                             'P0086p004',
                                             'P0087p004',
                                             'P0088p004',
                                             'P0089p004',
                                             'P0090p004'],
                                     'M08': [],
                                     'M09': [],
                                     'M10': [],
                                     'M11': [],
                                     'M12': ['P0011p004',
                                             'P0011p007',
                                             'P0012p004',
                                             'P0013p004',
                                             'P0041p002',
                                             'P0041p003',
                                             'P0042p002',
                                             'P0042p003',
                                             'P0043p002',
                                             'P0043p003',
                                             'P0044p002',
                                             'P0044p003',
                                             'P0045p002',
                                             'P0045p003',
                                             'P0046p002',
                                             'P0046p003',
                                             'P0047p002',
                                             'P0047p003'],
                                     'M13': ['P0001p001',
                                             'P0002p001',
                                             'P0003p001',
                                             'P0004p001',
                                             'P0005p001',
                                             'P0006p001',
                                             'P0007p001',
                                             'P0008p001',
                                             'P0009p001',
                                             'P0010p001',
                                             'P0081p002',
                                             'P0081p006',
                                             'P0082p002',
                                             'P0082p006',
                                             'P0083p002',
                                             'P0083p006',
                                             'P0084p002',
                                             'P0084p006',
                                             'P0085p002',
                                             'P0085p006',
                                             'P0086p002',
                                             'P0086p006',
                                             'P0087p002',
                                             'P0087p006',
                                             'P0088p002',
                                             'P0088p006',
                                             'P0089p002',
                                             'P0089p006',
                                             'P0090p002'],
                                     'M14': ['P0011p006',
                                             'P0041p004',
                                             'P0042p004',
                                             'P0043p004',
                                             'P0044p004',
                                             'P0045p004',
                                             'P0051p005',
                                             'P0052p005',
                                             'P0053p005',
                                             'P0061p005',
                                             'P0081p003',
                                             'P0081p007',
                                             'P0082p003',
                                             'P0082p007',
                                             'P0083p003',
                                             'P0083p007',
                                             'P0084p003',
                                             'P0084p007',
                                             'P0085p003',
                                             'P0085p007',
                                             'P0086p003',
                                             'P0086p007',
                                             'P0087p003',
                                             'P0088p003',
                                             'P0089p003',
                                             'P0090p003'],
                                     'M15': ['P0011p005',
                                             'P0012p005',
                                             'P0041p005',
                                             'P0042p005',
                                             'P0043p005',
                                             'P0044p005',
                                             'P0045p005',
                                             'P0051p004',
                                             'P0052p004',
                                             'P0053p004',
                                             'P0054p004',
                                             'P0061p004',
                                             'P0062p004',
                                             'P0063p004',
                                             'P0064p004',
                                             'P0081p008',
                                             'P0081p009',
                                             'P0082p008',
                                             'P0082p009',
                                             'P0083p008',
                                             'P0083p009',
                                             'P0084p008',
                                             'P0084p009',
                                             'P0085p008',
                                             'P0085p009',
                                             'P0086p008'],
                                     'M16': ['P0011p002',
                                             'P0011p003',
                                             'P0012p002',
                                             'P0012p003',
                                             'P0013p002',
                                             'P0013p003',
                                             'P0014p002',
                                             'P0014p003',
                                             'P0015p002',
                                             'P0015p003',
                                             'P0016p002',
                                             'P0016p003',
                                             'P0041p006',
                                             'P0041p007',
                                             'P0042p006',
                                             'P0043p006',
                                             'P0051p001',
                                             'P0051p002',
                                             'P0051p003',
                                             'P0052p001',
                                             'P0052p002',
                                             'P0052p003',
                                             'P0053p001',
                                             'P0053p002',
                                             'P0053p003',
                                             'P0054p001',
                                             'P0054p002',
                                             'P0054p003',
                                             'P0055p001',
                                             'P0055p002',
                                             'P0055p003',
                                             'P0056p001',
                                             'P0056p002',
                                             'P0057p001',
                                             'P0061p001',
                                             'P0061p002',
                                             'P0061p003',
                                             'P0062p001',
                                             'P0062p002',
                                             'P0062p003',
                                             'P0063p001',
                                             'P0063p002',
                                             'P0063p003',
                                             'P0064p001',
                                             'P0064p002',
                                             'P0064p003',
                                             'P0065p001',
                                             'P0065p002',
                                             'P0065p003',
                                             'P0066p001',
                                             'P0066p002',
                                             'P0067p001',
                                             'P0068p001',
                                             'P0069p001',
                                             'P0070p001',
                                             'P0071p002',
                                             'P0071p003',
                                             'P0072p002',
                                             'P0072p003',
                                             'P0073p002',
                                             'P0074p002'],
                                     'M17': ['P0031p001',
                                             'P0038p001',
                                             'P0081p001',
                                             'P0081p005',
                                             'P0082p001',
                                             'P0082p005',
                                             'P0083p001',
                                             'P0083p005',
                                             'P0084p001',
                                             'P0084p005',
                                             'P0085p001',
                                             'P0085p005',
                                             'P0086p001',
                                             'P0086p005',
                                             'P0087p001',
                                             'P0087p005',
                                             'P0088p001',
                                             'P0088p005',
                                             'P0089p001',
                                             'P0089p005',
                                             'P0090p001',
                                             'P0090p005'],
                                     'M18': [],
                                     'M19': [],
                                     'M20': [],
                                     'M21': [],
                                     'M22': [],
                                     'M23': [],
                                     'M24': [],
                                     'M25': [],
                                     'M26': ['P0021p001', 'P0021p002', 'P0022p001', 'P0023p001'],
                                     'M27': [],
                                     'M28': [],
                                     'M29': [],
                                     'M30': [],
                                     'M31': [],
                                     'M32': [],
                                     'M33': [],
                                     'M34': [],
                                     'M35': [],
                                     'M36': [],
                                     'M37': [],
                                     'M38': [],
                                     'M39': [],
                                     'M40': [],
                                     'M41': [],
                                     'M42': [],
                                     'M43': [],
                                     'M44': [],
                                     'M45': [],
                                     'M46': ['P0032p001',
                                             'P0033p001',
                                             'P0034p001',
                                             'P0035p001',
                                             'P0036p001',
                                             'P0037p001',
                                             'P0039p001',
                                             'P0040p001'],
                                     'M47': []},
                'pendingProcessOriginalPlan': {'M01': [], 'M02': [], 'M03': [], 'M04': [], 'M05': [],
                                     'M06': [['2022-06-01 00:00:00', '2022-06-01 02:22:00'],
                                             ['2022-06-01 07:58:00', '2022-06-01 10:20:00'],
                                             ['2022-06-01 10:20:00', '2022-06-01 12:42:00'],
                                             ['2022-06-01 18:17:00', '2022-06-01 20:39:00'],
                                             ['2022-06-01 23:27:00', '2022-06-02 01:49:00'],
                                             ['2022-06-02 04:37:00', '2022-06-02 06:59:00'],
                                             ['2022-06-02 09:47:00', '2022-06-02 12:09:00'],
                                             ['2022-06-01 02:22:00', '2022-06-01 05:10:00'],
                                             ['2022-06-01 05:10:00', '2022-06-01 07:58:00'],
                                             ['2022-06-01 12:41:00', '2022-06-01 15:29:00'],
                                             ['2022-06-01 15:29:00', '2022-06-01 18:17:00'],
                                             ['2022-06-01 20:39:00', '2022-06-01 23:27:00'],
                                             ['2022-06-02 01:49:00', '2022-06-02 04:37:00'],
                                             ['2022-06-02 06:59:00', '2022-06-02 09:47:00'],
                                             ['2022-06-02 12:09:00', '2022-06-02 14:57:00']],
                                     'M07': [['2022-06-03 19:56:00', '2022-06-03 22:25:00'],
                                             ['2022-06-04 01:39:00', '2022-06-04 04:08:00'],
                                             ['2022-06-04 14:19:00', '2022-06-04 16:48:00'],
                                             ['2022-06-05 00:08:00', '2022-06-05 02:37:00'],
                                             ['2022-06-04 08:15:00', '2022-06-04 10:44:00'],
                                             ['2022-06-04 18:25:00', '2022-06-04 20:54:00'],
                                             ['2022-06-01 00:00:00', '2022-06-01 03:35:00'],
                                             ['2022-06-01 06:08:00', '2022-06-01 09:43:00'],
                                             ['2022-06-01 11:20:00', '2022-06-01 14:55:00'],
                                             ['2022-06-01 16:32:00', '2022-06-01 20:07:00'],
                                             ['2022-06-01 21:44:00', '2022-06-02 01:19:00'],
                                             ['2022-06-04 10:44:00', '2022-06-04 14:19:00'],
                                             ['2022-06-01 04:31:00', '2022-06-01 06:08:00'],
                                             ['2022-06-04 05:01:00', '2022-06-04 06:38:00'],
                                             ['2022-06-01 09:43:00', '2022-06-01 11:20:00'],
                                             ['2022-06-04 20:54:00', '2022-06-04 22:31:00'],
                                             ['2022-06-01 14:55:00', '2022-06-01 16:32:00'],
                                             ['2022-06-01 20:07:00', '2022-06-01 21:44:00'],
                                             ['2022-06-03 22:25:00', '2022-06-04 00:02:00'],
                                             ['2022-06-04 00:02:00', '2022-06-04 01:39:00'],
                                             ['2022-06-04 06:38:00', '2022-06-04 08:15:00'],
                                             ['2022-06-04 16:48:00', '2022-06-04 18:25:00'],
                                             ['2022-06-04 22:31:00', '2022-06-05 00:08:00'],
                                             ['2022-06-05 02:37:00', '2022-06-05 04:14:00']], 'M08': [], 'M09': [],
                                     'M10': [], 'M11': [], 'M12': [['2022-06-03 07:23:00', '2022-06-03 09:47:00'],
                                                                   ['2022-06-06 04:15:00', '2022-06-06 06:39:00'],
                                                                   ['2022-06-04 08:07:00', '2022-06-04 10:31:00'],
                                                                   ['2022-06-06 08:25:00', '2022-06-06 10:49:00'],
                                                                   ['2022-06-01 05:10:00', '2022-06-01 06:56:00'],
                                                                   ['2022-06-01 06:56:00', '2022-06-01 08:42:00'],
                                                                   ['2022-06-01 08:41:00', '2022-06-01 10:27:00'],
                                                                   ['2022-06-01 10:27:00', '2022-06-01 12:13:00'],
                                                                   ['2022-06-01 15:29:00', '2022-06-01 17:15:00'],
                                                                   ['2022-06-01 17:15:00', '2022-06-01 19:01:00'],
                                                                   ['2022-06-01 19:01:00', '2022-06-01 20:47:00'],
                                                                   ['2022-06-03 09:47:00', '2022-06-03 11:33:00'],
                                                                   ['2022-06-03 11:33:00', '2022-06-03 13:19:00'],
                                                                   ['2022-06-04 10:31:00', '2022-06-04 12:17:00'],
                                                                   ['2022-06-03 13:19:00', '2022-06-03 15:05:00'],
                                                                   ['2022-06-04 12:17:00', '2022-06-04 14:03:00'],
                                                                   ['2022-06-04 14:03:00', '2022-06-04 15:49:00'],
                                                                   ['2022-06-06 06:39:00', '2022-06-06 08:25:00']],
                                     'M13': [['2022-06-01 00:00:00', '2022-06-01 00:18:00'],
                                             ['2022-06-01 00:18:00', '2022-06-01 00:36:00'],
                                             ['2022-06-01 00:36:00', '2022-06-01 00:54:00'],
                                             ['2022-06-01 00:54:00', '2022-06-01 01:12:00'],
                                             ['2022-06-01 01:12:00', '2022-06-01 01:30:00'],
                                             ['2022-06-01 01:30:00', '2022-06-01 01:48:00'],
                                             ['2022-06-01 03:17:00', '2022-06-01 03:35:00'],
                                             ['2022-06-01 03:35:00', '2022-06-01 03:53:00'],
                                             ['2022-06-03 11:24:00', '2022-06-03 11:42:00'],
                                             ['2022-06-03 11:42:00', '2022-06-03 12:00:00'],
                                             ['2022-06-01 01:48:00', '2022-06-01 03:17:00'],
                                             ['2022-06-03 12:00:00', '2022-06-03 13:29:00'],
                                             ['2022-06-01 03:53:00', '2022-06-01 05:22:00'],
                                             ['2022-06-03 19:25:00', '2022-06-03 20:54:00'],
                                             ['2022-06-01 05:59:00', '2022-06-01 07:28:00'],
                                             ['2022-06-03 22:23:00', '2022-06-03 23:52:00'],
                                             ['2022-06-01 07:28:00', '2022-06-01 08:57:00'],
                                             ['2022-06-03 23:52:00', '2022-06-04 01:21:00'],
                                             ['2022-06-03 13:29:00', '2022-06-03 14:58:00'],
                                             ['2022-06-04 01:21:00', '2022-06-04 02:50:00'],
                                             ['2022-06-03 14:58:00', '2022-06-03 16:27:00'],
                                             ['2022-06-04 21:00:00', '2022-06-04 22:29:00'],
                                             ['2022-06-03 16:27:00', '2022-06-03 17:56:00'],
                                             ['2022-06-04 22:29:00', '2022-06-04 23:58:00'],
                                             ['2022-06-03 17:56:00', '2022-06-03 19:25:00'],
                                             ['2022-06-04 23:58:00', '2022-06-05 01:27:00'],
                                             ['2022-06-03 20:54:00', '2022-06-03 22:23:00'],
                                             ['2022-06-05 01:27:00', '2022-06-05 02:56:00'],
                                             ['2022-06-04 02:50:00', '2022-06-04 04:19:00']],
                                     'M14': [['2022-06-04 12:05:00', '2022-06-04 14:20:00'],
                                             ['2022-06-01 08:42:00', '2022-06-01 11:18:00'],
                                             ['2022-06-03 16:22:00', '2022-06-03 18:58:00'],
                                             ['2022-06-03 20:12:00', '2022-06-03 22:48:00'],
                                             ['2022-06-04 06:38:00', '2022-06-04 09:14:00'],
                                             ['2022-06-04 16:48:00', '2022-06-04 19:24:00'],
                                             ['2022-06-03 18:58:00', '2022-06-03 19:56:00'],
                                             ['2022-06-04 02:30:00', '2022-06-04 03:28:00'],
                                             ['2022-06-04 04:26:00', '2022-06-04 05:24:00'],
                                             ['2022-06-04 03:28:00', '2022-06-04 04:26:00'],
                                             ['2022-06-01 03:17:00', '2022-06-01 04:31:00'],
                                             ['2022-06-03 13:29:00', '2022-06-03 14:43:00'],
                                             ['2022-06-01 05:22:00', '2022-06-01 06:36:00'],
                                             ['2022-06-04 00:02:00', '2022-06-04 01:16:00'],
                                             ['2022-06-01 07:28:00', '2022-06-01 08:42:00'],
                                             ['2022-06-04 01:16:00', '2022-06-04 02:30:00'],
                                             ['2022-06-01 11:18:00', '2022-06-01 12:32:00'],
                                             ['2022-06-04 14:20:00', '2022-06-04 15:34:00'],
                                             ['2022-06-03 14:58:00', '2022-06-03 16:12:00'],
                                             ['2022-06-04 19:24:00', '2022-06-04 20:38:00'],
                                             ['2022-06-03 22:48:00', '2022-06-04 00:02:00'],
                                             ['2022-06-04 22:29:00', '2022-06-04 23:43:00'],
                                             ['2022-06-04 05:24:00', '2022-06-04 06:38:00'],
                                             ['2022-06-04 15:34:00', '2022-06-04 16:48:00'],
                                             ['2022-06-04 20:38:00', '2022-06-04 21:52:00'],
                                             ['2022-06-04 23:43:00', '2022-06-05 00:57:00']],
                                     'M15': [['2022-06-03 09:47:00', '2022-06-03 12:47:00'],
                                             ['2022-06-04 10:31:00', '2022-06-04 13:31:00'],
                                             ['2022-06-02 18:29:00', '2022-06-02 19:55:00'],
                                             ['2022-06-04 03:28:00', '2022-06-04 04:54:00'],
                                             ['2022-06-04 13:31:00', '2022-06-04 14:57:00'],
                                             ['2022-06-04 15:48:00', '2022-06-04 17:14:00'],
                                             ['2022-06-06 08:03:00', '2022-06-06 09:29:00'],
                                             ['2022-06-01 13:41:00', '2022-06-01 16:17:00'],
                                             ['2022-06-02 19:55:00', '2022-06-02 22:31:00'],
                                             ['2022-06-02 22:31:00', '2022-06-03 01:07:00'],
                                             ['2022-06-03 17:56:00', '2022-06-03 20:32:00'],
                                             ['2022-06-04 00:52:00', '2022-06-04 03:28:00'],
                                             ['2022-06-04 04:54:00', '2022-06-04 07:30:00'],
                                             ['2022-06-05 00:34:00', '2022-06-05 03:10:00'],
                                             ['2022-06-06 05:27:00', '2022-06-06 08:03:00'],
                                             ['2022-06-03 23:10:00', '2022-06-04 00:01:00'],
                                             ['2022-06-04 00:01:00', '2022-06-04 00:52:00'],
                                             ['2022-06-04 09:40:00', '2022-06-04 10:31:00'],
                                             ['2022-06-04 17:14:00', '2022-06-04 18:05:00'],
                                             ['2022-06-04 14:57:00', '2022-06-04 15:48:00'],
                                             ['2022-06-04 18:56:00', '2022-06-04 19:47:00'],
                                             ['2022-06-04 18:05:00', '2022-06-04 18:56:00'],
                                             ['2022-06-04 19:47:00', '2022-06-04 20:38:00'],
                                             ['2022-06-04 20:38:00', '2022-06-04 21:29:00'],
                                             ['2022-06-04 21:29:00', '2022-06-04 22:20:00'],
                                             ['2022-06-04 23:43:00', '2022-06-05 00:34:00']],
                                     'M16': [['2022-06-01 07:56:00', '2022-06-01 11:05:00'],
                                             ['2022-06-03 04:25:00', '2022-06-03 07:23:00'],
                                             ['2022-06-02 22:31:00', '2022-06-03 01:40:00'],
                                             ['2022-06-04 03:37:00', '2022-06-04 06:35:00'],
                                             ['2022-06-05 00:51:00', '2022-06-05 04:00:00'],
                                             ['2022-06-06 05:27:00', '2022-06-06 08:25:00'],
                                             ['2022-06-05 04:00:00', '2022-06-05 07:09:00'],
                                             ['2022-06-06 14:10:00', '2022-06-06 17:08:00'],
                                             ['2022-06-06 08:25:00', '2022-06-06 11:34:00'],
                                             ['2022-06-07 07:37:00', '2022-06-07 10:35:00'],
                                             ['2022-06-07 12:10:00', '2022-06-07 15:19:00'],
                                             ['2022-06-07 22:24:00', '2022-06-08 01:22:00'],
                                             ['2022-06-03 17:56:00', '2022-06-03 19:31:00'],
                                             ['2022-06-05 12:30:00', '2022-06-05 14:05:00'],
                                             ['2022-06-07 10:35:00', '2022-06-07 12:10:00'],
                                             ['2022-06-07 18:04:00', '2022-06-07 19:39:00'],
                                             ['2022-06-01 00:00:00', '2022-06-01 02:36:00'],
                                             ['2022-06-01 05:20:00', '2022-06-01 07:56:00'],
                                             ['2022-06-01 11:05:00', '2022-06-01 13:41:00'],
                                             ['2022-06-01 16:26:00', '2022-06-01 19:02:00'],
                                             ['2022-06-01 21:38:00', '2022-06-02 00:14:00'],
                                             ['2022-06-02 17:19:00', '2022-06-02 19:55:00'],
                                             ['2022-06-01 19:02:00', '2022-06-01 21:38:00'],
                                             ['2022-06-02 11:05:00', '2022-06-02 13:41:00'],
                                             ['2022-06-02 19:55:00', '2022-06-02 22:31:00'],
                                             ['2022-06-02 08:29:00', '2022-06-02 11:05:00'],
                                             ['2022-06-03 07:23:00', '2022-06-03 09:59:00'],
                                             ['2022-06-03 15:20:00', '2022-06-03 17:56:00'],
                                             ['2022-06-03 09:59:00', '2022-06-03 12:35:00'],
                                             ['2022-06-05 07:09:00', '2022-06-05 09:45:00'],
                                             ['2022-06-05 14:05:00', '2022-06-05 16:41:00'],
                                             ['2022-06-03 19:31:00', '2022-06-03 22:07:00'],
                                             ['2022-06-06 11:34:00', '2022-06-06 14:10:00'],
                                             ['2022-06-06 19:53:00', '2022-06-06 22:29:00'],
                                             ['2022-06-01 02:35:00', '2022-06-01 05:20:00'],
                                             ['2022-06-02 00:14:00', '2022-06-02 02:59:00'],
                                             ['2022-06-03 22:07:00', '2022-06-04 00:52:00'],
                                             ['2022-06-01 13:41:00', '2022-06-01 16:26:00'],
                                             ['2022-06-02 05:44:00', '2022-06-02 08:29:00'],
                                             ['2022-06-04 00:52:00', '2022-06-04 03:37:00'],
                                             ['2022-06-02 02:59:00', '2022-06-02 05:44:00'],
                                             ['2022-06-03 01:40:00', '2022-06-03 04:25:00'],
                                             ['2022-06-04 09:20:00', '2022-06-04 12:05:00'],
                                             ['2022-06-03 12:35:00', '2022-06-03 15:20:00'],
                                             ['2022-06-05 09:45:00', '2022-06-05 12:30:00'],
                                             ['2022-06-06 02:42:00', '2022-06-06 05:27:00'],
                                             ['2022-06-04 06:35:00', '2022-06-04 09:20:00'],
                                             ['2022-06-05 23:57:00', '2022-06-06 02:42:00'],
                                             ['2022-06-06 17:08:00', '2022-06-06 19:53:00'],
                                             ['2022-06-04 15:43:00', '2022-06-04 18:28:00'],
                                             ['2022-06-07 15:19:00', '2022-06-07 18:04:00'],
                                             ['2022-06-04 18:28:00', '2022-06-04 21:13:00'],
                                             ['2022-06-07 02:07:00', '2022-06-07 04:52:00'],
                                             ['2022-06-07 04:52:00', '2022-06-07 07:37:00'],
                                             ['2022-06-07 19:39:00', '2022-06-07 22:24:00'],
                                             ['2022-06-02 13:41:00', '2022-06-02 17:19:00'],
                                             ['2022-06-04 21:13:00', '2022-06-05 00:51:00'],
                                             ['2022-06-04 12:05:00', '2022-06-04 15:43:00'],
                                             ['2022-06-06 22:29:00', '2022-06-07 02:07:00'],
                                             ['2022-06-05 16:41:00', '2022-06-05 20:19:00'],
                                             ['2022-06-05 20:19:00', '2022-06-05 23:57:00']],
                                     'M17': [['2022-06-01 01:19:00', '2022-06-01 02:00:00'],
                                             ['2022-06-01 05:57:00', '2022-06-01 06:38:00'],
                                             ['2022-06-01 00:00:00', '2022-06-01 01:19:00'],
                                             ['2022-06-01 07:57:00', '2022-06-01 09:16:00'],
                                             ['2022-06-01 02:00:00', '2022-06-01 03:19:00'],
                                             ['2022-06-03 18:06:00', '2022-06-03 19:25:00'],
                                             ['2022-06-01 03:19:00', '2022-06-01 04:38:00'],
                                             ['2022-06-03 20:44:00', '2022-06-03 22:03:00'],
                                             ['2022-06-01 04:38:00', '2022-06-01 05:57:00'],
                                             ['2022-06-03 22:03:00', '2022-06-03 23:22:00'],
                                             ['2022-06-01 06:38:00', '2022-06-01 07:57:00'],
                                             ['2022-06-04 00:02:00', '2022-06-04 01:21:00'],
                                             ['2022-06-01 09:16:00', '2022-06-01 10:35:00'],
                                             ['2022-06-04 01:39:00', '2022-06-04 02:58:00'],
                                             ['2022-06-01 10:35:00', '2022-06-01 11:54:00'],
                                             ['2022-06-04 08:15:00', '2022-06-04 09:34:00'],
                                             ['2022-06-01 11:54:00', '2022-06-01 13:13:00'],
                                             ['2022-06-04 22:39:00', '2022-06-04 23:58:00'],
                                             ['2022-06-01 13:13:00', '2022-06-01 14:32:00'],
                                             ['2022-06-05 00:08:00', '2022-06-05 01:27:00'],
                                             ['2022-06-03 19:25:00', '2022-06-03 20:44:00'],
                                             ['2022-06-05 04:14:00', '2022-06-05 05:33:00']], 'M18': [], 'M19': [],
                                     'M20': [], 'M21': [], 'M22': [], 'M23': [], 'M24': [], 'M25': [],
                                     'M26': [['2022-06-01 00:00:00', '2022-06-01 01:09:00'],
                                             ['2022-06-01 02:17:00', '2022-06-01 03:26:00'],
                                             ['2022-06-01 01:09:00', '2022-06-01 02:18:00'],
                                             ['2022-06-01 03:26:00', '2022-06-01 04:35:00']], 'M27': [], 'M28': [],
                                     'M29': [], 'M30': [], 'M31': [], 'M32': [], 'M33': [], 'M34': [], 'M35': [],
                                     'M36': [],
                                     'M37': [], 'M38': [], 'M39': [], 'M40': [], 'M41': [], 'M42': [], 'M43': [],
                                     'M44': [],
                                     'M45': [], 'M46': [['2022-06-01 00:00:00', '2022-06-01 00:31:00'],
                                                        ['2022-06-01 00:31:00', '2022-06-01 01:02:00'],
                                                        ['2022-06-01 01:02:00', '2022-06-01 01:33:00'],
                                                        ['2022-06-01 01:33:00', '2022-06-01 02:04:00'],
                                                        ['2022-06-01 02:04:00', '2022-06-01 02:35:00'],
                                                        ['2022-06-01 02:35:00', '2022-06-01 03:06:00'],
                                                        ['2022-06-01 03:06:00', '2022-06-01 03:37:00'],
                                                        ['2022-06-01 03:37:00', '2022-06-01 04:08:00']], 'M47': []},
                'processProductBelong': {'P0001': ['P0001p001'], 'P0002': ['P0002p001'], 'P0003': ['P0003p001'],
                                         'P0004': ['P0004p001'], 'P0005': ['P0005p001'], 'P0006': ['P0006p001'],
                                         'P0007': ['P0007p001'], 'P0008': ['P0008p001'], 'P0009': ['P0009p001'],
                                         'P0010': ['P0010p001'],
                                         'P0011': ['P0011p001', 'P0011p002', 'P0011p003', 'P0011p004', 'P0011p005',
                                                   'P0011p006',
                                                   'P0011p007', 'P0011p008'],
                                         'P0012': ['P0012p001', 'P0012p002', 'P0012p003', 'P0012p004', 'P0012p005',
                                                   'P0012p006',
                                                   'P0012p007', 'P0012p008'],
                                         'P0013': ['P0013p001', 'P0013p002', 'P0013p003', 'P0013p004', 'P0013p005',
                                                   'P0013p006',
                                                   'P0013p007', 'P0013p008'],
                                         'P0014': ['P0014p001', 'P0014p002', 'P0014p003', 'P0014p004', 'P0014p005',
                                                   'P0014p006',
                                                   'P0014p007', 'P0014p008'],
                                         'P0015': ['P0015p001', 'P0015p002', 'P0015p003', 'P0015p004', 'P0015p005',
                                                   'P0015p006',
                                                   'P0015p007', 'P0015p008'],
                                         'P0016': ['P0016p001', 'P0016p002', 'P0016p003', 'P0016p004', 'P0016p005',
                                                   'P0016p006',
                                                   'P0016p007', 'P0016p008'],
                                         'P0017': ['P0017p001', 'P0017p002', 'P0017p003', 'P0017p004', 'P0017p005',
                                                   'P0017p006',
                                                   'P0017p007', 'P0017p008'],
                                         'P0018': ['P0018p001', 'P0018p002', 'P0018p003', 'P0018p004', 'P0018p005',
                                                   'P0018p006',
                                                   'P0018p007', 'P0018p008'],
                                         'P0019': ['P0019p001', 'P0019p002', 'P0019p003', 'P0019p004', 'P0019p005',
                                                   'P0019p006',
                                                   'P0019p007', 'P0019p008'],
                                         'P0020': ['P0020p001', 'P0020p002', 'P0020p003', 'P0020p004', 'P0020p005',
                                                   'P0020p006',
                                                   'P0020p007', 'P0020p008'],
                                         'P0021': ['P0021p001', 'P0021p002', 'P0021p003', 'P0021p004', 'P0021p005',
                                                   'P0021p006'],
                                         'P0022': ['P0022p001', 'P0022p002', 'P0022p003', 'P0022p004', 'P0022p005',
                                                   'P0022p006'],
                                         'P0023': ['P0023p001', 'P0023p002', 'P0023p003', 'P0023p004', 'P0023p005',
                                                   'P0023p006'],
                                         'P0024': ['P0024p001', 'P0024p002', 'P0024p003', 'P0024p004', 'P0024p005',
                                                   'P0024p006'],
                                         'P0025': ['P0025p001', 'P0025p002', 'P0025p003', 'P0025p004', 'P0025p005',
                                                   'P0025p006'],
                                         'P0026': ['P0026p001', 'P0026p002', 'P0026p003', 'P0026p004', 'P0026p005',
                                                   'P0026p006'],
                                         'P0027': ['P0027p001', 'P0027p002', 'P0027p003', 'P0027p004', 'P0027p005',
                                                   'P0027p006'],
                                         'P0028': ['P0028p001', 'P0028p002', 'P0028p003', 'P0028p004', 'P0028p005',
                                                   'P0028p006'],
                                         'P0029': ['P0029p001', 'P0029p002', 'P0029p003', 'P0029p004', 'P0029p005',
                                                   'P0029p006'],
                                         'P0030': ['P0030p001', 'P0030p002', 'P0030p003', 'P0030p004', 'P0030p005',
                                                   'P0030p006'], 'P0031': ['P0031p001'], 'P0032': ['P0032p001'],
                                         'P0033': ['P0033p001'], 'P0034': ['P0034p001'], 'P0035': ['P0035p001'],
                                         'P0036': ['P0036p001'], 'P0037': ['P0037p001'], 'P0038': ['P0038p001'],
                                         'P0039': ['P0039p001'], 'P0040': ['P0040p001'],
                                         'P0041': ['P0041p001', 'P0041p002', 'P0041p003', 'P0041p004', 'P0041p005',
                                                   'P0041p006',
                                                   'P0041p007'],
                                         'P0042': ['P0042p001', 'P0042p002', 'P0042p003', 'P0042p004', 'P0042p005',
                                                   'P0042p006',
                                                   'P0042p007'],
                                         'P0043': ['P0043p001', 'P0043p002', 'P0043p003', 'P0043p004', 'P0043p005',
                                                   'P0043p006',
                                                   'P0043p007'],
                                         'P0044': ['P0044p001', 'P0044p002', 'P0044p003', 'P0044p004', 'P0044p005',
                                                   'P0044p006',
                                                   'P0044p007'],
                                         'P0045': ['P0045p001', 'P0045p002', 'P0045p003', 'P0045p004', 'P0045p005',
                                                   'P0045p006',
                                                   'P0045p007'],
                                         'P0046': ['P0046p001', 'P0046p002', 'P0046p003', 'P0046p004', 'P0046p005',
                                                   'P0046p006',
                                                   'P0046p007'],
                                         'P0047': ['P0047p001', 'P0047p002', 'P0047p003', 'P0047p004', 'P0047p005',
                                                   'P0047p006',
                                                   'P0047p007'],
                                         'P0048': ['P0048p001', 'P0048p002', 'P0048p003', 'P0048p004', 'P0048p005',
                                                   'P0048p006',
                                                   'P0048p007'],
                                         'P0049': ['P0049p001', 'P0049p002', 'P0049p003', 'P0049p004', 'P0049p005',
                                                   'P0049p006',
                                                   'P0049p007'],
                                         'P0050': ['P0050p001', 'P0050p002', 'P0050p003', 'P0050p004', 'P0050p005',
                                                   'P0050p006',
                                                   'P0050p007'],
                                         'P0051': ['P0051p001', 'P0051p002', 'P0051p003', 'P0051p004', 'P0051p005',
                                                   'P0051p006',
                                                   'P0051p007'],
                                         'P0052': ['P0052p001', 'P0052p002', 'P0052p003', 'P0052p004', 'P0052p005',
                                                   'P0052p006',
                                                   'P0052p007'],
                                         'P0053': ['P0053p001', 'P0053p002', 'P0053p003', 'P0053p004', 'P0053p005',
                                                   'P0053p006',
                                                   'P0053p007'],
                                         'P0054': ['P0054p001', 'P0054p002', 'P0054p003', 'P0054p004', 'P0054p005',
                                                   'P0054p006',
                                                   'P0054p007'],
                                         'P0055': ['P0055p001', 'P0055p002', 'P0055p003', 'P0055p004', 'P0055p005',
                                                   'P0055p006',
                                                   'P0055p007'],
                                         'P0056': ['P0056p001', 'P0056p002', 'P0056p003', 'P0056p004', 'P0056p005',
                                                   'P0056p006',
                                                   'P0056p007'],
                                         'P0057': ['P0057p001', 'P0057p002', 'P0057p003', 'P0057p004', 'P0057p005',
                                                   'P0057p006',
                                                   'P0057p007'],
                                         'P0058': ['P0058p001', 'P0058p002', 'P0058p003', 'P0058p004', 'P0058p005',
                                                   'P0058p006',
                                                   'P0058p007'],
                                         'P0059': ['P0059p001', 'P0059p002', 'P0059p003', 'P0059p004', 'P0059p005',
                                                   'P0059p006',
                                                   'P0059p007'],
                                         'P0060': ['P0060p001', 'P0060p002', 'P0060p003', 'P0060p004', 'P0060p005',
                                                   'P0060p006',
                                                   'P0060p007'],
                                         'P0061': ['P0061p001', 'P0061p002', 'P0061p003', 'P0061p004', 'P0061p005',
                                                   'P0061p006',
                                                   'P0061p007'],
                                         'P0062': ['P0062p001', 'P0062p002', 'P0062p003', 'P0062p004', 'P0062p005',
                                                   'P0062p006',
                                                   'P0062p007'],
                                         'P0063': ['P0063p001', 'P0063p002', 'P0063p003', 'P0063p004', 'P0063p005',
                                                   'P0063p006',
                                                   'P0063p007'],
                                         'P0064': ['P0064p001', 'P0064p002', 'P0064p003', 'P0064p004', 'P0064p005',
                                                   'P0064p006',
                                                   'P0064p007'],
                                         'P0065': ['P0065p001', 'P0065p002', 'P0065p003', 'P0065p004', 'P0065p005',
                                                   'P0065p006',
                                                   'P0065p007'],
                                         'P0066': ['P0066p001', 'P0066p002', 'P0066p003', 'P0066p004', 'P0066p005',
                                                   'P0066p006',
                                                   'P0066p007'],
                                         'P0067': ['P0067p001', 'P0067p002', 'P0067p003', 'P0067p004', 'P0067p005',
                                                   'P0067p006',
                                                   'P0067p007'],
                                         'P0068': ['P0068p001', 'P0068p002', 'P0068p003', 'P0068p004', 'P0068p005',
                                                   'P0068p006',
                                                   'P0068p007'],
                                         'P0069': ['P0069p001', 'P0069p002', 'P0069p003', 'P0069p004', 'P0069p005',
                                                   'P0069p006',
                                                   'P0069p007'],
                                         'P0070': ['P0070p001', 'P0070p002', 'P0070p003', 'P0070p004', 'P0070p005',
                                                   'P0070p006',
                                                   'P0070p007'],
                                         'P0071': ['P0071p001', 'P0071p002', 'P0071p003', 'P0071p004', 'P0071p005',
                                                   'P0071p006',
                                                   'P0071p007'],
                                         'P0072': ['P0072p001', 'P0072p002', 'P0072p003', 'P0072p004', 'P0072p005',
                                                   'P0072p006',
                                                   'P0072p007'],
                                         'P0073': ['P0073p001', 'P0073p002', 'P0073p003', 'P0073p004', 'P0073p005',
                                                   'P0073p006',
                                                   'P0073p007'],
                                         'P0074': ['P0074p001', 'P0074p002', 'P0074p003', 'P0074p004', 'P0074p005',
                                                   'P0074p006',
                                                   'P0074p007'],
                                         'P0075': ['P0075p001', 'P0075p002', 'P0075p003', 'P0075p004', 'P0075p005',
                                                   'P0075p006',
                                                   'P0075p007'],
                                         'P0076': ['P0076p001', 'P0076p002', 'P0076p003', 'P0076p004', 'P0076p005',
                                                   'P0076p006',
                                                   'P0076p007'],
                                         'P0077': ['P0077p001', 'P0077p002', 'P0077p003', 'P0077p004', 'P0077p005',
                                                   'P0077p006',
                                                   'P0077p007'],
                                         'P0078': ['P0078p001', 'P0078p002', 'P0078p003', 'P0078p004', 'P0078p005',
                                                   'P0078p006',
                                                   'P0078p007'],
                                         'P0079': ['P0079p001', 'P0079p002', 'P0079p003', 'P0079p004', 'P0079p005',
                                                   'P0079p006',
                                                   'P0079p007'],
                                         'P0080': ['P0080p001', 'P0080p002', 'P0080p003', 'P0080p004', 'P0080p005',
                                                   'P0080p006',
                                                   'P0080p007'],
                                         'P0081': ['P0081p001', 'P0081p002', 'P0081p003', 'P0081p004', 'P0081p005',
                                                   'P0081p006',
                                                   'P0081p007', 'P0081p008', 'P0081p009', 'P0081p010'],
                                         'P0082': ['P0082p001', 'P0082p002', 'P0082p003', 'P0082p004', 'P0082p005',
                                                   'P0082p006',
                                                   'P0082p007', 'P0082p008', 'P0082p009', 'P0082p010'],
                                         'P0083': ['P0083p001', 'P0083p002', 'P0083p003', 'P0083p004', 'P0083p005',
                                                   'P0083p006',
                                                   'P0083p007', 'P0083p008', 'P0083p009', 'P0083p010'],
                                         'P0084': ['P0084p001', 'P0084p002', 'P0084p003', 'P0084p004', 'P0084p005',
                                                   'P0084p006',
                                                   'P0084p007', 'P0084p008', 'P0084p009', 'P0084p010'],
                                         'P0085': ['P0085p001', 'P0085p002', 'P0085p003', 'P0085p004', 'P0085p005',
                                                   'P0085p006',
                                                   'P0085p007', 'P0085p008', 'P0085p009', 'P0085p010'],
                                         'P0086': ['P0086p001', 'P0086p002', 'P0086p003', 'P0086p004', 'P0086p005',
                                                   'P0086p006',
                                                   'P0086p007', 'P0086p008', 'P0086p009', 'P0086p010'],
                                         'P0087': ['P0087p001', 'P0087p002', 'P0087p003', 'P0087p004', 'P0087p005',
                                                   'P0087p006',
                                                   'P0087p007', 'P0087p008', 'P0087p009', 'P0087p010'],
                                         'P0088': ['P0088p001', 'P0088p002', 'P0088p003', 'P0088p004', 'P0088p005',
                                                   'P0088p006',
                                                   'P0088p007', 'P0088p008', 'P0088p009', 'P0088p010'],
                                         'P0089': ['P0089p001', 'P0089p002', 'P0089p003', 'P0089p004', 'P0089p005',
                                                   'P0089p006',
                                                   'P0089p007', 'P0089p008', 'P0089p009', 'P0089p010'],
                                         'P0090': ['P0090p001', 'P0090p002', 'P0090p003', 'P0090p004', 'P0090p005',
                                                   'P0090p006',
                                                   'P0090p007', 'P0090p008', 'P0090p009', 'P0090p010'],
                                         'P0091': ['P0091p001', 'P0091p002', 'P0091p003', 'P0091p004', 'P0091p005',
                                                   'P0091p006',
                                                   'P0091p007', 'P0091p008', 'P0091p009', 'P0091p010'],
                                         'P0092': ['P0092p001', 'P0092p002', 'P0092p003', 'P0092p004', 'P0092p005',
                                                   'P0092p006',
                                                   'P0092p007', 'P0092p008', 'P0092p009', 'P0092p010'],
                                         'P0093': ['P0093p001', 'P0093p002', 'P0093p003', 'P0093p004', 'P0093p005',
                                                   'P0093p006',
                                                   'P0093p007', 'P0093p008', 'P0093p009', 'P0093p010'],
                                         'P0094': ['P0094p001', 'P0094p002', 'P0094p003', 'P0094p004', 'P0094p005',
                                                   'P0094p006',
                                                   'P0094p007', 'P0094p008', 'P0094p009', 'P0094p010'],
                                         'P0095': ['P0095p001', 'P0095p002', 'P0095p003', 'P0095p004', 'P0095p005',
                                                   'P0095p006',
                                                   'P0095p007', 'P0095p008', 'P0095p009', 'P0095p010'],
                                         'P0096': ['P0096p001', 'P0096p002', 'P0096p003', 'P0096p004', 'P0096p005',
                                                   'P0096p006',
                                                   'P0096p007', 'P0096p008', 'P0096p009', 'P0096p010'],
                                         'P0097': ['P0097p001', 'P0097p002', 'P0097p003', 'P0097p004', 'P0097p005',
                                                   'P0097p006',
                                                   'P0097p007', 'P0097p008', 'P0097p009', 'P0097p010'],
                                         'P0098': ['P0098p001', 'P0098p002', 'P0098p003', 'P0098p004', 'P0098p005',
                                                   'P0098p006',
                                                   'P0098p007', 'P0098p008', 'P0098p009', 'P0098p010'],
                                         'P0099': ['P0099p001', 'P0099p002', 'P0099p003', 'P0099p004', 'P0099p005',
                                                   'P0099p006',
                                                   'P0099p007', 'P0099p008', 'P0099p009', 'P0099p010'],
                                         'P0100': ['P0100p001', 'P0100p002', 'P0100p003', 'P0100p004', 'P0100p005',
                                                   'P0100p006',
                                                   'P0100p007', 'P0100p008', 'P0100p009', 'P0100p010']},
                'revokeProductDecision': {
                    'P0062': [{'specificProcessID': ['P0062p001', 'P0062p002', 'P0062p003', 'P0062p004'],
                               'machineID': [['M15'], ['M14'], ['M12'], ['M12']],
                               'machinePriority': [[5], [5], [5], [5]],
                               'processTime': [[180.0], [135.0], [144.0], [144.0]]}, 'Y'],
                    'P0063': ['null', 'N']},
                'productDateEnd': {'P0001': '2022-06-08 00:00:00',
                                   'P0002': '2022-06-08 00:00:00',
                                   'P0003': '2022-06-08 00:00:00',
                                   'P0004': '2022-06-08 00:00:00',
                                   'P0005': '2022-06-08 00:00:00',
                                   'P0006': '2022-06-08 00:00:00',
                                   'P0007': '2022-06-08 00:00:00',
                                   'P0008': '2022-06-08 00:00:00',
                                   'P0009': '2022-06-08 00:00:00',
                                   'P0010': '2022-06-08 00:00:00',
                                   'P0011': '2022-06-08 00:00:00',
                                   'P0012': '2022-06-08 00:00:00',
                                   'P0013': '2022-06-08 00:00:00',
                                   'P0014': '2022-06-08 00:00:00',
                                   'P0015': '2022-06-08 00:00:00',
                                   'P0016': '2022-06-08 00:00:00',
                                   'P0017': '2022-06-08 00:00:00',
                                   'P0018': '2022-06-08 00:00:00',
                                   'P0019': '2022-06-08 00:00:00',
                                   'P0020': '2022-06-08 00:00:00',
                                   'P0021': '2022-06-08 00:00:00',
                                   'P0022': '2022-06-08 00:00:00',
                                   'P0023': '2022-06-08 00:00:00',
                                   'P0024': '2022-06-08 00:00:00',
                                   'P0025': '2022-06-08 00:00:00',
                                   'P0026': '2022-06-08 00:00:00',
                                   'P0027': '2022-06-08 00:00:00',
                                   'P0028': '2022-06-08 00:00:00',
                                   'P0029': '2022-06-08 00:00:00',
                                   'P0030': '2022-06-08 00:00:00',
                                   'P0031': '2022-06-08 00:00:00',
                                   'P0032': '2022-06-08 00:00:00',
                                   'P0033': '2022-06-08 00:00:00',
                                   'P0034': '2022-06-08 00:00:00',
                                   'P0035': '2022-06-08 00:00:00',
                                   'P0036': '2022-06-08 00:00:00',
                                   'P0037': '2022-06-08 00:00:00',
                                   'P0038': '2022-06-08 00:00:00',
                                   'P0039': '2022-06-08 00:00:00',
                                   'P0040': '2022-06-08 00:00:00',
                                   'P0041': '2022-06-08 00:00:00',
                                   'P0042': '2022-06-08 00:00:00',
                                   'P0043': '2022-06-08 00:00:00',
                                   'P0044': '2022-06-08 00:00:00',
                                   'P0045': '2022-06-08 00:00:00',
                                   'P0046': '2022-06-08 00:00:00',
                                   'P0047': '2022-06-08 00:00:00',
                                   'P0048': '2022-06-08 00:00:00',
                                   'P0049': '2022-06-08 00:00:00',
                                   'P0050': '2022-06-08 00:00:00',
                                   'P0051': '2022-06-08 00:00:00',
                                   'P0052': '2022-06-08 00:00:00',
                                   'P0053': '2022-06-08 00:00:00',
                                   'P0054': '2022-06-08 00:00:00',
                                   'P0055': '2022-06-08 00:00:00',
                                   'P0056': '2022-06-08 00:00:00',
                                   'P0057': '2022-06-08 00:00:00',
                                   'P0058': '2022-06-08 00:00:00',
                                   'P0059': '2022-06-08 00:00:00',
                                   'P0060': '2022-06-08 00:00:00',
                                   'P0061': '2022-06-08 00:00:00',
                                   'P0062': '2022-06-08 00:00:00',
                                   'P0063': '2022-06-08 00:00:00',
                                   'P0064': '2022-06-08 00:00:00',
                                   'P0065': '2022-06-08 00:00:00',
                                   'P0066': '2022-06-08 00:00:00',
                                   'P0067': '2022-06-08 00:00:00',
                                   'P0068': '2022-06-08 00:00:00',
                                   'P0069': '2022-06-08 00:00:00',
                                   'P0070': '2022-06-08 00:00:00',
                                   'P0071': '2022-06-08 00:00:00',
                                   'P0072': '2022-06-08 00:00:00',
                                   'P0073': '2022-06-08 00:00:00',
                                   'P0074': '2022-06-08 00:00:00',
                                   'P0075': '2022-06-08 00:00:00',
                                   'P0076': '2022-06-08 00:00:00',
                                   'P0077': '2022-06-08 00:00:00',
                                   'P0078': '2022-06-08 00:00:00',
                                   'P0079': '2022-06-08 00:00:00',
                                   'P0080': '2022-06-08 00:00:00',
                                   'P0081': '2022-06-08 00:00:00',
                                   'P0082': '2022-06-08 00:00:00',
                                   'P0083': '2022-06-08 00:00:00',
                                   'P0084': '2022-06-08 00:00:00',
                                   'P0085': '2022-06-08 00:00:00',
                                   'P0086': '2022-06-08 00:00:00',
                                   'P0087': '2022-06-08 00:00:00',
                                   'P0088': '2022-06-08 00:00:00',
                                   'P0089': '2022-06-08 00:00:00',
                                   'P0090': '2022-06-08 00:00:00',
                                   'P0091': '2022-06-08 00:00:00',
                                   'P0092': '2022-06-08 00:00:00',
                                   'P0093': '2022-06-08 00:00:00',
                                   'P0094': '2022-06-08 00:00:00',
                                   'P0095': '2022-06-08 00:00:00',
                                   'P0096': '2022-06-08 00:00:00',
                                   'P0097': '2022-06-08 00:00:00',
                                   'P0098': '2022-06-08 00:00:00',
                                   'P0099': '2022-06-08 00:00:00',
                                   'P0100': '2022-06-08 00:00:00'}
                }

    # 输入排产周期起始时刻、时长、每日休息时段
    def information(data):
        planstart = data['planStart']
        planstart = dt.datetime.strptime(planstart, "%Y-%m-%d %H:%M:%S")
        span = data['periodLength']
        planspan = dt.timedelta(days=span)
        planend = planstart + planspan
        reststart = '22:00'  # 暂定为22:00
        [a, b] = reststart.split(':')
        a = int(a)
        b = int(b)
        reststart = dt.timedelta(minutes=a * 60 + b)
        restend = '24:00'  # 暂定为24:00
        [a, b] = restend.split(':')
        a = int(a)
        b = int(b)
        restend = dt.timedelta(minutes=a * 60 + b)
        # 输出每日工作时间段
        restduration = []
        for i in range(0, span):
            a = planstart + dt.timedelta(days=i) + reststart
            b = planstart + dt.timedelta(days=i) + restend
            restduration.append([a, b])
        T = data['replanTime']
        T = dt.datetime.strptime(T, "%Y-%m-%d %H:%M:%S")
        for i in range(0, span):  # 若插单时刻发生在休息时段，将其移至该休息时段末尾
            if (restduration[i][0] <= T) & (restduration[i][1] >= T):
                T = restduration[i][1]
        return planstart, planend, planspan, restduration, T


    # 按时间顺序对Q和QQ重排
    def adjust(Q, QQ):
        for key in Q.keys():
            if len(Q[key]) <= 1:
                continue
            for k in range(0, len(Q[key]) - 1):
                for j in range(k + 1, len(Q[key])):
                    a = dt.datetime.strptime(Q[key][k][0], "%Y-%m-%d %H:%M:%S")
                    b = dt.datetime.strptime(Q[key][j][0], "%Y-%m-%d %H:%M:%S")
                    if a > b:
                        temp1 = Q[key][k]
                        Q[key][k] = Q[key][j]
                        Q[key][j] = temp1
                        temp2 = QQ[key][k]
                        QQ[key][k] = QQ[key][j]
                        QQ[key][j] = temp2
        return Q, QQ


    # 分离计划中specificprocessID包含的productID和processID
    def seperate(a, b):
        if b.startswith(a):
            return b.replace(a, '', 1)


    # 判断specificprocessID（b）包含的productID是否为productID（a）
    def containjudge(a, b):
        if b.startswith(a):
            return True


    # 计算设备从紧急插单点到排产周期末的占用时间
    def caculation(T, Q, planend):  # T为发生紧急插单的时刻点，Q为原排产计划的加工时间集合
        t = {}
        for key in Q.keys():
            if Q[key] == []:  # case1
                t[key] = dt.timedelta(minutes=0)
            else:
                between = 0
                for j in range(0, len(Q[key])):
                    start = dt.datetime.strptime(Q[key][j][0], "%Y-%m-%d %H:%M:%S")
                    end = dt.datetime.strptime(Q[key][j][1], "%Y-%m-%d %H:%M:%S")
                    if (start <= T) & (end >= T):  # case2订单插入时设备在加工
                        t1 = end - T
                        # print(t1)#正在加工工序还需占用的设备时间
                        t2 = dt.timedelta(minutes=0)
                        for k in range(j + 1, len(Q[key])):
                            start1 = dt.datetime.strptime(Q[key][k][0], "%Y-%m-%d %H:%M:%S")
                            end1 = dt.datetime.strptime(Q[key][k][1], "%Y-%m-%d %H:%M:%S")
                            if end1 <= planend:
                                t2 = t2 + end1 - start1  # 重拍后超出24小时排产周期的不应当计入设备占用时间
                            elif (start1 <= planend) & (end1 > planend):  # 对于生产时间跨越排产周期结点的情况
                                t2 = t2 + planend - start1
                            else:
                                break
                        t0 = t1 + t2
                        t[key] = t0
                        between = 1
                        break
                if between == 0:
                    # 设置一个临时变量busy，对紧急插单时刻原排产计划未安排工序的设为0
                    busy = 0
                    for j in range(0, len(Q[key])):
                        start = dt.datetime.strptime(Q[key][j][0], "%Y-%m-%d %H:%M:%S")
                        if start > T:
                            t2 = dt.timedelta(minutes=0)
                            for k in range(j, len(Q[key])):
                                start1 = dt.datetime.strptime(Q[key][k][0], "%Y-%m-%d %H:%M:%S")
                                end1 = dt.datetime.strptime(Q[key][k][1], "%Y-%m-%d %H:%M:%S")
                                if end1 <= planend:
                                    t2 = t2 + end1 - start1  # 重排后超出24小时排产周期的不应当计入设备占用时间
                                else:
                                    t2 = t2 + planend - start1
                                    break
                            t[key] = t2
                            # print(t2)
                            busy = 1
                            break
                    if busy == 0:
                        t[key] = dt.timedelta(minutes=0)
        return t


    # 更新插入工序后的设备占用时间
    def rearrange(T, Q, a, CQ, N,
                  QQ):  # TT当前时刻，Q当前加工时间集合，a选中设备编号，记录插入的工序在集合N中的位置索引，N待加工工序相关信息（这里主要用到工序耗时），QQ当前加工工序名称顺序集合
        # print("当前插单工序（粗排）：", N[CQ[0]][CQ[1]][0])
        CQT = []
        between = 0
        if Q[a] == []:  # 对原排产计划上没有安排工序的情况
            CQT.append(T.strftime("%Y-%m-%d %H:%M:%S"))
            CQT.append((T + dt.timedelta(seconds=round(N[CQ[0]][CQ[1]][2][N[CQ[0]][CQ[1]][1].index(a)]))).strftime(
                "%Y-%m-%d %H:%M:%S"))  # 添加round保证不会出现微秒的情况
            Q[a].append(CQT)
            QQ[a].append(N[CQ[0]][CQ[1]][0])
        else:
            for i in range(0, len(Q[a])):
                start = dt.datetime.strptime(Q[a][i][0], "%Y-%m-%d %H:%M:%S")
                end = dt.datetime.strptime(Q[a][i][1], "%Y-%m-%d %H:%M:%S")
                if (start <= T) & (end >= T):  # 订单插入时设备在加工
                    CQT.append(end.strftime("%Y-%m-%d %H:%M:%S"))  # 正常来说这里应该要加上准备时间
                    CQT.append(
                        (end + dt.timedelta(seconds=round(N[CQ[0]][CQ[1]][2][N[CQ[0]][CQ[1]][1].index(a)]))).strftime(
                            "%Y-%m-%d %H:%M:%S"))
                    # 在这之后的工序加工时间顺延
                    for k in range(i + 1, len(Q[a])):
                        start1 = dt.datetime.strptime(Q[a][k][0], "%Y-%m-%d %H:%M:%S")
                        end1 = dt.datetime.strptime(Q[a][k][1], "%Y-%m-%d %H:%M:%S")
                        restart = start1 + dt.timedelta(
                            seconds=round(N[CQ[0]][CQ[1]][2][N[CQ[0]][CQ[1]][1].index(a)]))  # 正常来说这里应该加上插入工序后的设备整理时间
                        reend = end1 + dt.timedelta(seconds=round(N[CQ[0]][CQ[1]][2][N[CQ[0]][CQ[1]][1].index(a)]))
                        Q[a][k][0] = restart.strftime("%Y-%m-%d %H:%M:%S")
                        Q[a][k][1] = reend.strftime("%Y-%m-%d %H:%M:%S")
                    Q[a].append(CQT)
                    QQ[a].append(N[CQ[0]][CQ[1]][0])
                    between = 1
                    break
            if between == 0:
                busy = 0  # 临时变量，若当前紧急插单时刻在所有已安排工序的开始时间和结束时间之后（此时很有可能紧急插单时刻已经超出排产周期），则busy=0
                for j in range(0, len(Q[a])):
                    start = dt.datetime.strptime(Q[a][j][0], "%Y-%m-%d %H:%M:%S")
                    if start > T:
                        CQT.append(T.strftime("%Y-%m-%d %H:%M:%S"))  # 正常来说这里应该要加上准备时间
                        CQT.append(
                            (T + dt.timedelta(seconds=round(N[CQ[0]][CQ[1]][2][N[CQ[0]][CQ[1]][1].index(a)]))).strftime(
                                "%Y-%m-%d %H:%M:%S"))
                        for k in range(j, len(Q[a])):
                            start2 = dt.datetime.strptime(Q[a][k][0], "%Y-%m-%d %H:%M:%S")
                            end2 = dt.datetime.strptime(Q[a][k][1], "%Y-%m-%d %H:%M:%S")
                            restart = start2 + dt.timedelta(
                                seconds=round(N[CQ[0]][CQ[1]][2][N[CQ[0]][CQ[1]][1].index(a)]))  # 正常来说这里应该加上插入工序后的设备整理时间
                            reend = end2 + dt.timedelta(seconds=round(N[CQ[0]][CQ[1]][2][N[CQ[0]][CQ[1]][1].index(a)]))
                            Q[a][k][0] = restart.strftime("%Y-%m-%d %H:%M:%S")
                            Q[a][k][1] = reend.strftime("%Y-%m-%d %H:%M:%S")
                        Q[a].append(CQT)
                        QQ[a].append(N[CQ[0]][CQ[1]][0])
                        busy = 1
                        break
                if busy == 0:
                    CQT.append(T.strftime("%Y-%m-%d %H:%M:%S"))
                    CQT.append((T + dt.timedelta(seconds=round(N[CQ[0]][CQ[1]][2][N[CQ[0]][CQ[1]][1].index(a)]))).strftime(
                        "%Y-%m-%d %H:%M:%S"))
                    Q[a].append(CQT)
                    QQ[a].append(N[CQ[0]][CQ[1]][0])
        return Q, QQ, CQT


    # 对重拍后引起的后一工序的开始时间超过前一工序结束时间的情况进行调整
    def renew(Q, QQ, J, N, restduration, T, planstart, planend,productDateEnd):  # Q为粗排的加工时间集合，QQ为粗排的加工工序集合
        QW = []  # 按设备分类的待加工工序集合
        # QK=[]#不区分设备的待加工工序集合
        QQW = []  # 按设备分类的待加工工序名称集合
        QQ2 = []  # QQW备份，检查环节使用
        Q1 = []  # 存储插单时刻前的工序时间
        QQ1 = []  # 存储插单时刻前的工序名称
        NW = []  # 按(具体)产品编号分类的待加工工序集合
        for key in Q.keys():
            qw = []
            qqw = []
            qq2 = []
            if Q[key] == []:  # 若某一设备未安排工序，则插单时刻后的加工工序、时间集合为空
                q1 = []
                qw = []
                qqw = []
                qq2 = []
                qq1 = []
                Q1.append(q1)
                QW.append(qw)
                QQW.append(qqw)
                QQ2.append(qq2)
                QQ1.append(qq1)
            else:
                q1 = []
                qq1 = []
                for j in range(0, len(Q[key])):
                    start = dt.datetime.strptime(Q[key][j][0], "%Y-%m-%d %H:%M:%S")
                    if start >= T:
                        qw.append(Q[key][j])
                        qqw.append(QQ[key][j])
                        qq2.append(QQ[key][j])
                    else:
                        q1.append(Q[key][j])
                        qq1.append(QQ[key][j])
                Q1.append(q1)
                QQ1.append(qq1)
                QQ2.append(qq2)
                QW.append(qw)
                QQW.append(qqw)
        '''
        print("插单时刻前生产计划部分：", Q1)
        print("插单时刻前生产计划部分（名称）", QQ1)
        '''
        for key in J.keys():
            nw = []
            for j in range(0, len(J[key])):
                for k in range(0, len(QQW)):
                    if J[key][j] in QQW[k]:
                        nw.append(J[key][j])
            if nw != []:
                NW.append(nw)
        # QK.extend(qqw)
        for i in range(0, len(N)):
            nw = []
            for j in range(0, len(N[i])):
                for k in range(0, len(QQW)):
                    if N[i][j][0] in QQW[k]:
                        nw.append(N[i][j][0])
            NW.append(nw)
        print("插单时刻后的产品待加工工序名称集合（按产品分类）：", NW)
        print("插单时刻后的产品待加工工序名称集合（按设备分类）：", QQW)
        print("插单时刻后的产品待加工工序时间集合（按设备分类）：", QW)
        print("插单时刻后工序名称：", QQ2)
        # 初始化设备空闲时间（可以开始加工的最早时间）
        TW = []
        for i in range(0, len(QW)):
            if QW[i] == []:  # 若某一设备未安排工序，设置工序的最早加工时间这一排产周期的起点
                TW.append(planstart)
            else:
                TW.append(QW[i][0][0])
        # 初始化产品上一工序时间
        l = len(NW)
        TTW = [T.strftime("%Y-%m-%d %H:%M:%S") for _ in range(l)]  # 初始化时间为紧急插单的时刻
        # 正式更新排产计划
        z = len(QQW)
        finalQ = [[] for _ in range(z)]  # 用来存储最后各设备各工序的加工时间
        while (QQW != [[] for _ in range(z)]):
            for i in range(0, len(QQW)):
                j = 0
                while (j < len(NW)):  # 如果产品集合未遍历完
                    if QQW[i] != []:
                        if NW[j] != []:
                            if QQW[i][0] == NW[j][0]:
                                print("当前排产工序", QQW[i][0])
                                tcost = dt.datetime.strptime(QW[i][0][1],
                                                             "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(
                                    QW[i][0][0], "%Y-%m-%d %H:%M:%S")
                                a = dt.datetime.strptime(TW[i], "%Y-%m-%d %H:%M:%S")  # 设备最早可加工时间
                                b = dt.datetime.strptime(TTW[j], "%Y-%m-%d %H:%M:%S")  # 上一工序结束时间
                                if a <= b:
                                    temp1 = b
                                else:
                                    temp1 = a
                                # 判断当前安排的工序时间是否侵占了休息时间
                                x = temp1 + tcost
                                for p in range(0, len(restduration)):
                                    if restduration[p][0].day == temp1.day:
                                        if (temp1 > restduration[p][0]) & (
                                                temp1 < restduration[p][1]):  # 休息时段内才开始的，一律推迟到休息时段末
                                            temp1 = restduration[p][1]
                                            x = temp1 + tcost
                                        elif (temp1 <= restduration[p][0]) & (
                                                x > restduration[p][1]):  # 休息时段前开始，但耗完休息时段仍为加工完的，推迟至休息时段末
                                            temp1 = restduration[p][1]
                                            x = temp1 + tcost
                                        break
                                QW[i][0][0] = temp1.strftime("%Y-%m-%d %H:%M:%S")
                                QW[i][0][1] = x.strftime("%Y-%m-%d %H:%M:%S")
                                TW[i] = QW[i][0][1]  # 更新设备的最早可加工时间
                                TTW[j] = QW[i][0][1]  # 更新产品上一工序结束时间
                                finalQ[i].append(QW[i][0])  # 将安排好的工序的开始时间和结束时间放入最终顺序中
                                del QQW[i][0]
                                del QW[i][0]
                                del NW[j][0]
                                print(QQW)
                                print(QW)
                                print(NW)
                                print(finalQ)
                                j = 0  # 如果找到某一件产品的当前最前工序为当前设备的最前工序，那么当前设备的第二个工序成为最前工序，同样需要对所有产品种类进行遍历，重置产品序号为1
                            else:
                                j = j + 1  # 如果当前产品的最前工序与当前设备的最前工序不同，产品序号加1，判断下一产品的最前工序是否为当前设备的最前工序
                        else:
                            j = j + 1  # 如果当前产品的工序已经安排完，则去比对下一产品的最前工序
                    else:
                        break  # 如果某一设备的工序已被安排完，则安排下一个设备的工序
        print("插单时刻后工序名称：", QQ2)

        # 按产品检查，能否将某些工序移动至设备时间线的空闲处
        for i in range(0, len(finalQ)):
            if finalQ[i] != []:
                for j in range(1, len(finalQ[i])):  # 从设备安排的第二道工序开始，第一道工序已为最前，无法调整
                    for key_3 in J.keys():
                        if QQ2[i][j] in J[key_3]:
                            break
                    aaa = key_3
                    bbb = seperate(aaa,QQ2[i][j])
                    print("当前检查工序：", QQ2[i][j])
                    if J[aaa].index(QQ2[i][j]) != 0:
                        ab = J[aaa][J[aaa].index(QQ2[i][j])-1]
                        print("紧前工序名称", ab)
                        Qspan = dt.datetime.strptime(finalQ[i][j][1],
                                                     "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(
                            finalQ[i][j][0], "%Y-%m-%d %H:%M:%S")
                        print("工序耗时", Qspan)
                        lastend = '0'
                        find = '0'
                        for m in range(0, len(QQ2)):  #
                            for n in range(0, len(QQ2[m])):
                                print(m)
                                print(n)
                                if QQ2[m][n] == ab:
                                    lastend = dt.datetime.strptime(finalQ[m][n][1],
                                                                   "%Y-%m-%d %H:%M:%S")  # 记录紧前工序的结束加工时间，从这个时刻点开始向后检查空当
                                    print("紧前工序结束时间", lastend)
                                    find = '1'
                                    break
                            if (find == '1'):
                                break
                        if (lastend != '0'):  # 找到紧前工序
                            for jj in range(0, j):
                                c = dt.datetime.strptime(finalQ[i][jj][1], "%Y-%m-%d %H:%M:%S")
                                print("空当检查工序", QQ2[i][jj])
                                if c >= lastend:  # 只对该设备该工序前、紧前工序结束时刻后的工序进行检查
                                    cspan = dt.datetime.strptime(finalQ[i][jj + 1][0],
                                                                 "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(
                                        finalQ[i][jj][1], "%Y-%m-%d %H:%M:%S")
                                    print("空当起始工序", QQ2[i][jj])
                                    print("空当时间", cspan)
                                    if cspan >= Qspan:
                                        print("有空位，可前移", QQ2[i][j])
                                        finalQ[i][j][0] = finalQ[i][jj][1]
                                        cc = dt.datetime.strptime(finalQ[i][j][0], "%Y-%m-%d %H:%M:%S") + Qspan
                                        finalQ[i][j][1] = cc.strftime("%Y-%m-%d %H:%M:%S")
                                        TEMP1 = finalQ[i][j][0]
                                        TEMP2 = finalQ[i][j][1]
                                        TEMP = QQ2[i][j]
                                        for kk in range(j - 1, jj, -1):  # 将该工序移动至空当
                                            a = finalQ[i][kk][0]
                                            b = finalQ[i][kk][1]
                                            finalQ[i][kk + 1][0] = a
                                            finalQ[i][kk + 1][1] = b
                                            QQ2[i][kk + 1] = QQ2[i][kk]
                                        finalQ[i][jj + 1][0] = TEMP1
                                        finalQ[i][jj + 1][1] = TEMP2
                                        QQ2[i][jj + 1] = TEMP
                                        print("移动后的工序顺序", QQ2[i])
                                        print("移动后的工序时间集合", finalQ[i])
                                        break
        # 输出
        for i in range(len(Q1)):
            Q1[i].extend(finalQ[i])
        print("重排完成后的加工时间集合", Q1)
        for i in range(len(QQ1)):
            QQ1[i].extend(QQ2[i])
        print("重排完成后的工序名称集合", QQ1)
        # 反馈无法在交付日期前完成的产品件号信息
        unableDelieverOnTime = {}  # 存储不能按时交付产品件号、原定交货时间、当前安排计划下的完工时间
        for key in productDateEnd.keys():
            finalprocessID = J[key][-1]  # 最后一道工序
            for i in range(len(QQ1)):
                if finalprocessID in QQ1[i]:
                    finalprocessfinishtime = dt.datetime.strptime(Q1[i][QQ1[i].index(finalprocessID)][-1],
                                                                  "%Y-%m-%d %H:%M:%S")  # 当前安排计划下的完工时间
                    dateend = dt.datetime.strptime(productDateEnd[key], "%Y-%m-%d %H:%M:%S")  # 原定交货时间
                    if finalprocessfinishtime > dateend:
                        unableDelieverOnTime[key] = {}
                        unableDelieverOnTime[key]['dateEnd'] = productDateEnd[key]
                        unableDelieverOnTime[key]['planedFinishTime'] = Q1[i][QQ1[i].index(finalprocessID)][-1]
                    break
        '''
            # 输出超出排产周期的工序名称及下一阶段设备的最早可加工时间
        TNEXT = [dt.timedelta(minutes=0) for i in range(z)]  # 存储下一阶段设备的最早可加工时间
        QNEXT = [[] for _ in range(z)]  # 存储推迟到下一排产周期的工序
        for i in range(0, len(Q1)):
            print("当前设备", i)
            removeQ = []
            removeQQ = []  # 储存需要移除的工序时间及名称
            if Q1[i] != []:
                t = dt.timedelta(minutes=0)
                for j in range(0, len(Q1[i])):
                    a = dt.datetime.strptime(Q1[i][j][0], "%Y-%m-%d %H:%M:%S")
                    b = dt.datetime.strptime(Q1[i][j][1], "%Y-%m-%d %H:%M:%S")
                    if a >= planend:
                        QNEXT[i].append(QQ1[i][j])
                        removeQ.append(Q1[i][j])
                        removeQQ.append(QQ1[i][j])
                    if (a < planend) & (b > planend):
                        t = t + b - planend
                TNEXT[i] = t
                for j in range(0, len(removeQ)):
                    print(j)
                    Q1[i].remove(removeQ[j])
                    QQ1[i].remove(removeQQ[j])
        '''

        return Q1, QQ1,unableDelieverOnTime


    def home():

        planstart, planend, planspan, restduration, T = information(data)
        QQ = data['pendingProcessMachine']
        Q = data['pendingProcessOriginalPlan']
        J = data['processProductBelong']
        productDateEnd = data['productDateEnd']
        # QQ,Q,product,J,data,PQ,PNAME=planinput(planstart)
        Q, QQ = adjust(Q, QQ)
        revokeproduct = data['revokeProductDecision']
        feedback = {}  # 反馈信息
        RPUF = []  # RPUF:revokeproductprocessunfinished撤单产品在撤单时刻还未开始加工的工序集合
        for key in revokeproduct.keys():
            rpuf = [key, [], []]  # 第一个元素为具体产品编号，第二个括号存储未开始加工的工序名称,第三个括号存储未开始加工的工序时间集合
            for key_1 in QQ.keys():
                for b in range(0, len(QQ[key_1])):
                    if containjudge(key, QQ[key_1][b]):  # 判断QQ[key_1][b]是否归属于key表示的产品
                        end = dt.datetime.strptime(Q[key_1][b][1], "%Y-%m-%d %H:%M:%S")
                        if end > T:
                            rpuf[1].append(QQ[key_1][b])
                            rpuf[2].append(Q[key_1][b])
                            rpuf[1] = sorted(rpuf[1])
                            rpuf[2] = sorted(rpuf[2])
            RPUF.append(rpuf)
        # print("未开始加工", RPUF)
        N = []  # [[[工序编号‘PXXXXpXXX’,[可选设备列表]，[加工时间列表]，[机器优先级列表]],...],...]
        for key in revokeproduct.keys():
            NN = []
            finish = 1  # 表示某撤单产品是否已全部加工完
            unstart = 0  # 记录撤单时刻还未上料的产品在RPUF的index
            for j in range(len(RPUF)):
                if RPUF[j][0] == key:
                    if RPUF[j][1] != []:  # 有未加工工序
                        finish = 0
                        if RPUF[j][1][0] == J[key][0]:  # 未加工的最前工序为产品的第一道工序(最先工序)，说明未上料，记录下其在RPUF矩阵的坐标
                            unstart = 1
                    break
            if finish == 1:
                feedback[key] = "撤单产品的全部工序已在撤单发起前全部完成，无移除工序！"
            else:
                if revokeproduct[key][0] == 'null':  # 不改制
                    if revokeproduct[key][1] == 'Y':  # 人工判断移除后续加工计划
                        # 移除全部加工工序
                        for z in range(0, len(RPUF[j][1])):
                            for key_2 in QQ.keys():
                                if RPUF[j][1][j] in QQ[key_2]:
                                    QQ[key_2].remove(RPUF[j][1][z])
                                    Q[key_2].remove(RPUF[j][2][z])
                    else:  # 不移除后续工序
                        # 撤单时刻应将开始加工的产品让它继续加工完
                        # 撤单时刻还未开始加工第一道工序的产品，移除该产品的所有工序
                        if unstart != 0:
                            for q in range(0, len(RPUF[j][1])):
                                for key_2 in QQ.keys():
                                    if RPUF[j][1][q] in QQ[key_2]:
                                        QQ[key_2].remove(RPUF[j][1][q])
                                        Q[key_2].remove(RPUF[j][2][q])
                else:  # 改制
                    # print("xxxxx", unstart)
                    # step1移除除此产品后续加工计划
                    # 撤单时刻应将开始加工的产品让它继续加工完
                    # 撤单时刻还未开始加工第一道工序的产品，移除该产品的所有工序
                    if unstart != 0:
                        for q in range(0, len(RPUF[j][1])):
                            for key_2 in QQ.keys():
                                if RPUF[j][1][q] in QQ[key_2]:
                                    QQ[key_2].remove(RPUF[j][1][q])
                                    Q[key_2].remove(RPUF[j][2][q])
                    # step2获得新的加工工序信息
                    NNN = revokeproduct[key][0]
                    J[key] = sorted(list(set.union(set(J[key]), set(NNN['specificProcessID']))))  # 将改制工序编号更新到J中
                    for n in range(len(NNN['specificProcessID'])):
                        nn = [NNN['specificProcessID'][n], NNN['machineID'][n], NNN['processTime'][n],
                              NNN['machinePriority'][n]]
                        NN.append(nn)
            if NN != []:
                N.append(NN)
        '''
        print("重新输入的待加工工序集合", N)
        print("统一插单前工序时间集合", Q)
        print("统一插单前工序名称集合", QQ)
        # 进行统一插单
        # 更新排产计划
        # 绘制甘特图
        '''
        for i in range(0, len(N)):
            TT = T
            for j in range(0, len(N[i])):
                t = caculation(TT, Q, planend)
                lasta = 'M01'  # 初始化临时变量，记录上一次选中的设备编号
                lastt = planend - T  # 初始化临时变量，记录上一次选中的设备的占用时间为全部被占用
                for k in N[i][j][1]:  # 选设备
                    if t[k] <= lastt:
                        lasta = k
                        lastt = t[k]
                CQ = [i, j]  # 记录插入的工序在集合N中的位置索引
                Q, QQ, CQT = rearrange(TT, Q, lasta, CQ, N,
                                       QQ)  # TT当前时刻，Q当前加工时间集合，a选中设备下标，CQ记录插入的工序在集合N中的位置索引，N待加工工序相关信息（这里主要用到工序耗时），QQ当前加工工序名称顺序集合
                print('工序%s的插入时刻%s' % (N[i][j][0], CQT))
                TT = dt.datetime.strptime(CQT[1], "%Y-%m-%d %H:%M:%S")  # 更新当前时间
                Q, QQ = adjust(Q, QQ)
        finalQ, finalQQ,unableDelieverOnTime = renew(Q, QQ, J, N, restduration, T, planstart, planend,productDateEnd)  # finalQ加工时间集合，finalQQ加工名称集合
        replanProcessName = {}
        replanProcessTime = {}
        j = -1
        for key in Q.keys():
            j = j + 1
            replanProcessTime[key] = finalQ[j]
            replanProcessName[key] = finalQQ[j]
        respond = {'replanProcessName': replanProcessName,
                   'replanProcessTime': replanProcessTime,
                   'replanProcessOthers': {},
                   'unableDelieverOnTime':unableDelieverOnTime,
                   'feedback': feedback} # replanProcessOthers暂时定为空集
        return json.dumps(respond,ensure_ascii=False)
    return home()

# ######################################################
# 紧急撤单结束
# ##############################################
# 设备故障开始
# #######################################
@app.route('/machinefailure', methods=['GET', 'POST'])
def failure():
    if request.method == 'POST':
        data = request.get_data()
        data = json.loads(data)
    else:
        data = {'hours': 22,
                'periodLength': 7,
                'planStart': '2022-06-01 00:00:00',
                'process': {'P0001': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0002': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0003': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0004': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0005': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0006': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0007': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0008': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0009': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0010': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0011': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0012': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0013': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0014': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0015': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0016': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0017': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0018': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0019': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0020': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0021': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0022': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0023': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0024': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0025': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0026': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0027': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0028': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0029': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0030': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0031': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0032': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0033': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0034': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0035': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0036': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0037': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0038': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0039': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0040': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0041': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0042': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0043': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0044': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0045': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0046': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0047': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0048': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0049': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0050': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0051': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0052': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0053': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0054': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0055': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0056': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0057': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0058': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0059': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0060': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0061': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0062': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0063': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0064': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0065': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0066': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0067': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0068': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0069': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0070': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0071': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0072': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0073': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0074': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0075': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0076': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0077': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0078': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0079': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0080': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0081': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0082': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0083': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0084': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0085': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0086': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0087': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0088': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0089': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0090': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0091': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0092': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0093': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0094': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0095': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0096': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0097': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0098': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0099': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0100': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]}},
                'processProductBelong': {'P0001': ['P0001p001'], 'P0002': ['P0002p001'], 'P0003': ['P0003p001'],
                                         'P0004': ['P0004p001'], 'P0005': ['P0005p001'], 'P0006': ['P0006p001'],
                                         'P0007': ['P0007p001'], 'P0008': ['P0008p001'], 'P0009': ['P0009p001'],
                                         'P0010': ['P0010p001'],
                                         'P0011': ['P0011p001', 'P0011p002', 'P0011p003', 'P0011p004', 'P0011p005',
                                                   'P0011p006',
                                                   'P0011p007', 'P0011p008'],
                                         'P0012': ['P0012p001', 'P0012p002', 'P0012p003', 'P0012p004', 'P0012p005',
                                                   'P0012p006',
                                                   'P0012p007', 'P0012p008'],
                                         'P0013': ['P0013p001', 'P0013p002', 'P0013p003', 'P0013p004', 'P0013p005',
                                                   'P0013p006',
                                                   'P0013p007', 'P0013p008'],
                                         'P0014': ['P0014p001', 'P0014p002', 'P0014p003', 'P0014p004', 'P0014p005',
                                                   'P0014p006',
                                                   'P0014p007', 'P0014p008'],
                                         'P0015': ['P0015p001', 'P0015p002', 'P0015p003', 'P0015p004', 'P0015p005',
                                                   'P0015p006',
                                                   'P0015p007', 'P0015p008'],
                                         'P0016': ['P0016p001', 'P0016p002', 'P0016p003', 'P0016p004', 'P0016p005',
                                                   'P0016p006',
                                                   'P0016p007', 'P0016p008'],
                                         'P0017': ['P0017p001', 'P0017p002', 'P0017p003', 'P0017p004', 'P0017p005',
                                                   'P0017p006',
                                                   'P0017p007', 'P0017p008'],
                                         'P0018': ['P0018p001', 'P0018p002', 'P0018p003', 'P0018p004', 'P0018p005',
                                                   'P0018p006',
                                                   'P0018p007', 'P0018p008'],
                                         'P0019': ['P0019p001', 'P0019p002', 'P0019p003', 'P0019p004', 'P0019p005',
                                                   'P0019p006',
                                                   'P0019p007', 'P0019p008'],
                                         'P0020': ['P0020p001', 'P0020p002', 'P0020p003', 'P0020p004', 'P0020p005',
                                                   'P0020p006',
                                                   'P0020p007', 'P0020p008'],
                                         'P0021': ['P0021p001', 'P0021p002', 'P0021p003', 'P0021p004', 'P0021p005',
                                                   'P0021p006'],
                                         'P0022': ['P0022p001', 'P0022p002', 'P0022p003', 'P0022p004', 'P0022p005',
                                                   'P0022p006'],
                                         'P0023': ['P0023p001', 'P0023p002', 'P0023p003', 'P0023p004', 'P0023p005',
                                                   'P0023p006'],
                                         'P0024': ['P0024p001', 'P0024p002', 'P0024p003', 'P0024p004', 'P0024p005',
                                                   'P0024p006'],
                                         'P0025': ['P0025p001', 'P0025p002', 'P0025p003', 'P0025p004', 'P0025p005',
                                                   'P0025p006'],
                                         'P0026': ['P0026p001', 'P0026p002', 'P0026p003', 'P0026p004', 'P0026p005',
                                                   'P0026p006'],
                                         'P0027': ['P0027p001', 'P0027p002', 'P0027p003', 'P0027p004', 'P0027p005',
                                                   'P0027p006'],
                                         'P0028': ['P0028p001', 'P0028p002', 'P0028p003', 'P0028p004', 'P0028p005',
                                                   'P0028p006'],
                                         'P0029': ['P0029p001', 'P0029p002', 'P0029p003', 'P0029p004', 'P0029p005',
                                                   'P0029p006'],
                                         'P0030': ['P0030p001', 'P0030p002', 'P0030p003', 'P0030p004', 'P0030p005',
                                                   'P0030p006'], 'P0031': ['P0031p001'], 'P0032': ['P0032p001'],
                                         'P0033': ['P0033p001'], 'P0034': ['P0034p001'], 'P0035': ['P0035p001'],
                                         'P0036': ['P0036p001'], 'P0037': ['P0037p001'], 'P0038': ['P0038p001'],
                                         'P0039': ['P0039p001'], 'P0040': ['P0040p001'],
                                         'P0041': ['P0041p001', 'P0041p002', 'P0041p003', 'P0041p004', 'P0041p005',
                                                   'P0041p006',
                                                   'P0041p007'],
                                         'P0042': ['P0042p001', 'P0042p002', 'P0042p003', 'P0042p004', 'P0042p005',
                                                   'P0042p006',
                                                   'P0042p007'],
                                         'P0043': ['P0043p001', 'P0043p002', 'P0043p003', 'P0043p004', 'P0043p005',
                                                   'P0043p006',
                                                   'P0043p007'],
                                         'P0044': ['P0044p001', 'P0044p002', 'P0044p003', 'P0044p004', 'P0044p005',
                                                   'P0044p006',
                                                   'P0044p007'],
                                         'P0045': ['P0045p001', 'P0045p002', 'P0045p003', 'P0045p004', 'P0045p005',
                                                   'P0045p006',
                                                   'P0045p007'],
                                         'P0046': ['P0046p001', 'P0046p002', 'P0046p003', 'P0046p004', 'P0046p005',
                                                   'P0046p006',
                                                   'P0046p007'],
                                         'P0047': ['P0047p001', 'P0047p002', 'P0047p003', 'P0047p004', 'P0047p005',
                                                   'P0047p006',
                                                   'P0047p007'],
                                         'P0048': ['P0048p001', 'P0048p002', 'P0048p003', 'P0048p004', 'P0048p005',
                                                   'P0048p006',
                                                   'P0048p007'],
                                         'P0049': ['P0049p001', 'P0049p002', 'P0049p003', 'P0049p004', 'P0049p005',
                                                   'P0049p006',
                                                   'P0049p007'],
                                         'P0050': ['P0050p001', 'P0050p002', 'P0050p003', 'P0050p004', 'P0050p005',
                                                   'P0050p006',
                                                   'P0050p007'],
                                         'P0051': ['P0051p001', 'P0051p002', 'P0051p003', 'P0051p004', 'P0051p005',
                                                   'P0051p006',
                                                   'P0051p007'],
                                         'P0052': ['P0052p001', 'P0052p002', 'P0052p003', 'P0052p004', 'P0052p005',
                                                   'P0052p006',
                                                   'P0052p007'],
                                         'P0053': ['P0053p001', 'P0053p002', 'P0053p003', 'P0053p004', 'P0053p005',
                                                   'P0053p006',
                                                   'P0053p007'],
                                         'P0054': ['P0054p001', 'P0054p002', 'P0054p003', 'P0054p004', 'P0054p005',
                                                   'P0054p006',
                                                   'P0054p007'],
                                         'P0055': ['P0055p001', 'P0055p002', 'P0055p003', 'P0055p004', 'P0055p005',
                                                   'P0055p006',
                                                   'P0055p007'],
                                         'P0056': ['P0056p001', 'P0056p002', 'P0056p003', 'P0056p004', 'P0056p005',
                                                   'P0056p006',
                                                   'P0056p007'],
                                         'P0057': ['P0057p001', 'P0057p002', 'P0057p003', 'P0057p004', 'P0057p005',
                                                   'P0057p006',
                                                   'P0057p007'],
                                         'P0058': ['P0058p001', 'P0058p002', 'P0058p003', 'P0058p004', 'P0058p005',
                                                   'P0058p006',
                                                   'P0058p007'],
                                         'P0059': ['P0059p001', 'P0059p002', 'P0059p003', 'P0059p004', 'P0059p005',
                                                   'P0059p006',
                                                   'P0059p007'],
                                         'P0060': ['P0060p001', 'P0060p002', 'P0060p003', 'P0060p004', 'P0060p005',
                                                   'P0060p006',
                                                   'P0060p007'],
                                         'P0061': ['P0061p001', 'P0061p002', 'P0061p003', 'P0061p004', 'P0061p005',
                                                   'P0061p006',
                                                   'P0061p007'],
                                         'P0062': ['P0062p001', 'P0062p002', 'P0062p003', 'P0062p004', 'P0062p005',
                                                   'P0062p006',
                                                   'P0062p007'],
                                         'P0063': ['P0063p001', 'P0063p002', 'P0063p003', 'P0063p004', 'P0063p005',
                                                   'P0063p006',
                                                   'P0063p007'],
                                         'P0064': ['P0064p001', 'P0064p002', 'P0064p003', 'P0064p004', 'P0064p005',
                                                   'P0064p006',
                                                   'P0064p007'],
                                         'P0065': ['P0065p001', 'P0065p002', 'P0065p003', 'P0065p004', 'P0065p005',
                                                   'P0065p006',
                                                   'P0065p007'],
                                         'P0066': ['P0066p001', 'P0066p002', 'P0066p003', 'P0066p004', 'P0066p005',
                                                   'P0066p006',
                                                   'P0066p007'],
                                         'P0067': ['P0067p001', 'P0067p002', 'P0067p003', 'P0067p004', 'P0067p005',
                                                   'P0067p006',
                                                   'P0067p007'],
                                         'P0068': ['P0068p001', 'P0068p002', 'P0068p003', 'P0068p004', 'P0068p005',
                                                   'P0068p006',
                                                   'P0068p007'],
                                         'P0069': ['P0069p001', 'P0069p002', 'P0069p003', 'P0069p004', 'P0069p005',
                                                   'P0069p006',
                                                   'P0069p007'],
                                         'P0070': ['P0070p001', 'P0070p002', 'P0070p003', 'P0070p004', 'P0070p005',
                                                   'P0070p006',
                                                   'P0070p007'],
                                         'P0071': ['P0071p001', 'P0071p002', 'P0071p003', 'P0071p004', 'P0071p005',
                                                   'P0071p006',
                                                   'P0071p007'],
                                         'P0072': ['P0072p001', 'P0072p002', 'P0072p003', 'P0072p004', 'P0072p005',
                                                   'P0072p006',
                                                   'P0072p007'],
                                         'P0073': ['P0073p001', 'P0073p002', 'P0073p003', 'P0073p004', 'P0073p005',
                                                   'P0073p006',
                                                   'P0073p007'],
                                         'P0074': ['P0074p001', 'P0074p002', 'P0074p003', 'P0074p004', 'P0074p005',
                                                   'P0074p006',
                                                   'P0074p007'],
                                         'P0075': ['P0075p001', 'P0075p002', 'P0075p003', 'P0075p004', 'P0075p005',
                                                   'P0075p006',
                                                   'P0075p007'],
                                         'P0076': ['P0076p001', 'P0076p002', 'P0076p003', 'P0076p004', 'P0076p005',
                                                   'P0076p006',
                                                   'P0076p007'],
                                         'P0077': ['P0077p001', 'P0077p002', 'P0077p003', 'P0077p004', 'P0077p005',
                                                   'P0077p006',
                                                   'P0077p007'],
                                         'P0078': ['P0078p001', 'P0078p002', 'P0078p003', 'P0078p004', 'P0078p005',
                                                   'P0078p006',
                                                   'P0078p007'],
                                         'P0079': ['P0079p001', 'P0079p002', 'P0079p003', 'P0079p004', 'P0079p005',
                                                   'P0079p006',
                                                   'P0079p007'],
                                         'P0080': ['P0080p001', 'P0080p002', 'P0080p003', 'P0080p004', 'P0080p005',
                                                   'P0080p006',
                                                   'P0080p007'],
                                         'P0081': ['P0081p001', 'P0081p002', 'P0081p003', 'P0081p004', 'P0081p005',
                                                   'P0081p006',
                                                   'P0081p007', 'P0081p008', 'P0081p009', 'P0081p010'],
                                         'P0082': ['P0082p001', 'P0082p002', 'P0082p003', 'P0082p004', 'P0082p005',
                                                   'P0082p006',
                                                   'P0082p007', 'P0082p008', 'P0082p009', 'P0082p010'],
                                         'P0083': ['P0083p001', 'P0083p002', 'P0083p003', 'P0083p004', 'P0083p005',
                                                   'P0083p006',
                                                   'P0083p007', 'P0083p008', 'P0083p009', 'P0083p010'],
                                         'P0084': ['P0084p001', 'P0084p002', 'P0084p003', 'P0084p004', 'P0084p005',
                                                   'P0084p006',
                                                   'P0084p007', 'P0084p008', 'P0084p009', 'P0084p010'],
                                         'P0085': ['P0085p001', 'P0085p002', 'P0085p003', 'P0085p004', 'P0085p005',
                                                   'P0085p006',
                                                   'P0085p007', 'P0085p008', 'P0085p009', 'P0085p010'],
                                         'P0086': ['P0086p001', 'P0086p002', 'P0086p003', 'P0086p004', 'P0086p005',
                                                   'P0086p006',
                                                   'P0086p007', 'P0086p008', 'P0086p009', 'P0086p010'],
                                         'P0087': ['P0087p001', 'P0087p002', 'P0087p003', 'P0087p004', 'P0087p005',
                                                   'P0087p006',
                                                   'P0087p007', 'P0087p008', 'P0087p009', 'P0087p010'],
                                         'P0088': ['P0088p001', 'P0088p002', 'P0088p003', 'P0088p004', 'P0088p005',
                                                   'P0088p006',
                                                   'P0088p007', 'P0088p008', 'P0088p009', 'P0088p010'],
                                         'P0089': ['P0089p001', 'P0089p002', 'P0089p003', 'P0089p004', 'P0089p005',
                                                   'P0089p006',
                                                   'P0089p007', 'P0089p008', 'P0089p009', 'P0089p010'],
                                         'P0090': ['P0090p001', 'P0090p002', 'P0090p003', 'P0090p004', 'P0090p005',
                                                   'P0090p006',
                                                   'P0090p007', 'P0090p008', 'P0090p009', 'P0090p010'],
                                         'P0091': ['P0091p001', 'P0091p002', 'P0091p003', 'P0091p004', 'P0091p005',
                                                   'P0091p006',
                                                   'P0091p007', 'P0091p008', 'P0091p009', 'P0091p010'],
                                         'P0092': ['P0092p001', 'P0092p002', 'P0092p003', 'P0092p004', 'P0092p005',
                                                   'P0092p006',
                                                   'P0092p007', 'P0092p008', 'P0092p009', 'P0092p010'],
                                         'P0093': ['P0093p001', 'P0093p002', 'P0093p003', 'P0093p004', 'P0093p005',
                                                   'P0093p006',
                                                   'P0093p007', 'P0093p008', 'P0093p009', 'P0093p010'],
                                         'P0094': ['P0094p001', 'P0094p002', 'P0094p003', 'P0094p004', 'P0094p005',
                                                   'P0094p006',
                                                   'P0094p007', 'P0094p008', 'P0094p009', 'P0094p010'],
                                         'P0095': ['P0095p001', 'P0095p002', 'P0095p003', 'P0095p004', 'P0095p005',
                                                   'P0095p006',
                                                   'P0095p007', 'P0095p008', 'P0095p009', 'P0095p010'],
                                         'P0096': ['P0096p001', 'P0096p002', 'P0096p003', 'P0096p004', 'P0096p005',
                                                   'P0096p006',
                                                   'P0096p007', 'P0096p008', 'P0096p009', 'P0096p010'],
                                         'P0097': ['P0097p001', 'P0097p002', 'P0097p003', 'P0097p004', 'P0097p005',
                                                   'P0097p006',
                                                   'P0097p007', 'P0097p008', 'P0097p009', 'P0097p010'],
                                         'P0098': ['P0098p001', 'P0098p002', 'P0098p003', 'P0098p004', 'P0098p005',
                                                   'P0098p006',
                                                   'P0098p007', 'P0098p008', 'P0098p009', 'P0098p010'],
                                         'P0099': ['P0099p001', 'P0099p002', 'P0099p003', 'P0099p004', 'P0099p005',
                                                   'P0099p006',
                                                   'P0099p007', 'P0099p008', 'P0099p009', 'P0099p010'],
                                         'P0100': ['P0100p001', 'P0100p002', 'P0100p003', 'P0100p004', 'P0100p005',
                                                   'P0100p006',
                                                   'P0100p007', 'P0100p008', 'P0100p009', 'P0100p010']},
                'replanTime': '2022-06-02 00:00:00',
                'breakMachine': {'M07': False, 'M15': True},
                'pendingProcessProduct': {'P0042': ['P0042p005', 'P0042p006'],
                                          'P0088': ['P0088p003', 'P0088p004', 'P0088p005', 'P0088p006'],
                                          'P0053': ['P0053p005'], 'P0067': ['P0067p001'],
                                          'P0012': ['P0012p003', 'P0012p004', 'P0012p005'],
                                          'P0063': ['P0063p003', 'P0063p004'],
                                          'P0083': ['P0083p007', 'P0083p008', 'P0083p009'],
                                          'P0066': ['P0066p001', 'P0066p002'], 'P0041': ['P0041p007'],
                                          'P0074': ['P0074p002'], 'P0068': ['P0068p001'], 'P0076': ['P0076p001'],
                                          'P0044': ['P0044p004', 'P0044p005'], 'P0055': ['P0055p002', 'P0055p003'],
                                          'P0052': ['P0052p005', 'P0052p006'],
                                          'P0072': ['P0072p002', 'P0072p003'],
                                          'P0087': ['P0087p003', 'P0087p004', 'P0087p005', 'P0087p006'],
                                          'P0085': ['P0085p004', 'P0085p005', 'P0085p006', 'P0085p007', 'P0085p008',
                                                    'P0085p009'], 'P0046': ['P0046p003'],
                                          'P0065': ['P0065p001', 'P0065p002', 'P0065p003'],
                                          'P0013': ['P0013p002', 'P0013p003', 'P0013p004'], 'P0070': ['P0070p001'],
                                          'P0016': ['P0016p002', 'P0016p003'],
                                          'P0090': ['P0090p002', 'P0090p003', 'P0090p004', 'P0090p005'],
                                          'P0047': ['P0047p002', 'P0047p003'], 'P0051': ['P0051p006', 'P0051p007'],
                                          'P0062': ['P0062p003', 'P0062p004'],
                                          'P0089': ['P0089p003', 'P0089p004', 'P0089p005', 'P0089p006'],
                                          'P0014': ['P0014p002', 'P0014p003'],
                                          'P0064': ['P0064p002', 'P0064p003', 'P0064p004'], 'P0057': ['P0057p001'],
                                          'P0082': ['P0082p007', 'P0082p008', 'P0082p009', 'P0082p010'],
                                          'P0084': ['P0084p007', 'P0084p008', 'P0084p009'], 'P0069': ['P0069p001'],
                                          'P0086': ['P0086p004', 'P0086p005', 'P0086p006', 'P0086p007', 'P0086p008'],
                                          'P0081': ['P0081p009', 'P0081p010'], 'P0043': ['P0043p005', 'P0043p006'],
                                          'P0045': ['P0045p003', 'P0045p004', 'P0045p005'], 'P0073': ['P0073p002'],
                                          'P0011': ['P0011p006', 'P0011p007'], 'P0015': ['P0015p002', 'P0015p003'],
                                          'P0056': ['P0056p002'], 'P0071': ['P0071p003'],
                                          'P0061': ['P0061p004', 'P0061p005', 'P0061p006', 'P0061p007']},
                'pendingProcessMachine': {'M01': [], 'M02': [], 'M03': [], 'M04': [], 'M05': [], 'M06': [],
                                          'M07': ['P0051p006', 'P0085p004', 'P0086p004', 'P0051p007', 'P0081p010',
                                                  'P0087p004',
                                                  'P0061p006', 'P0076p001',
                                                  'P0052p006', 'P0088p004', 'P0061p007',
                                                  'P0082p010', 'P0089p004', 'P0052p007', 'P0090p004'],
                                          'M08': [], 'M09': [], 'M10': [], 'M11': [],
                                          'M12': ['P0012p004', 'P0045p003', 'P0046p003', 'P0047p002', 'P0011p007',
                                                  'P0047p003',
                                                  'P0013p004'],
                                          'M13': ['P0085p006', 'P0090p002', 'P0086p006', 'P0087p006', 'P0088p006',
                                                  'P0089p006'],
                                          'M14': ['P0082p007', 'P0083p007', 'P0052p005', 'P0061p005', 'P0053p005',
                                                  'P0087p003',
                                                  'P0044p004', 'P0011p006',
                                                  'P0084p007', 'P0088p003', 'P0045p004',
                                                  'P0085p007', 'P0089p003', 'P0086p007', 'P0090p003'],
                                          'M15': ['P0081p009', 'P0061p004', 'P0042p005', 'P0062p004', 'P0082p008',
                                                  'P0012p005',
                                                  'P0043p005', 'P0083p008',
                                                  'P0044p005', 'P0082p009', 'P0084p008',
                                                  'P0083p009', 'P0084p009', 'P0085p008', 'P0085p009', 'P0086p008',
                                                  'P0063p004',
                                                  'P0064p004', 'P0045p005'],
                                          'M16': ['P0062p003', 'P0012p003', 'P0065p001', 'P0063p003', 'P0072p002',
                                                  'P0066p001',
                                                  'P0067p001', 'P0071p003',
                                                  'P0013p002', 'P0014p002', 'P0055p002',
                                                  'P0064p002', 'P0041p007', 'P0055p003', 'P0073p002', 'P0074p002',
                                                  'P0065p002',
                                                  'P0064p003', 'P0013p003',
                                                  'P0015p002', 'P0056p002', 'P0014p003',
                                                  'P0065p003', 'P0057p001', 'P0072p003', 'P0068p001', 'P0069p001',
                                                  'P0015p003',
                                                  'P0042p006', 'P0016p002',
                                                  'P0066p002', 'P0043p006', 'P0070p001',
                                                  'P0016p003'],
                                          'M17': ['P0085p005', 'P0086p005', 'P0087p005', 'P0088p005', 'P0089p005',
                                                  'P0090p005'],
                                          'M18': [], 'M19': [],
                                          'M20': [], 'M21': [], 'M22': [],
                                          'M23': [], 'M24': [], 'M25': [], 'M26': [], 'M27': [], 'M28': [], 'M29': [],
                                          'M30': [], 'M31': [], 'M32': [], 'M33': [], 'M34': [], 'M35': [], 'M36': [],
                                          'M37': [], 'M38': [], 'M39': [], 'M40': [], 'M41': [], 'M42': [], 'M43': [],
                                          'M44': [], 'M45': [], 'M46': [], 'M47': []},
                'pendingProcessOriginalPlan': {'M01': [],
                                               'M02': [],
                                               'M03': [],
                                               'M04': [],
                                               'M05': [],
                                               'M06': [],
                                               'M07': [['2021-09-04 00:00:00', '2021-09-04 02:29:00'],
                                                       ['2021-09-04 00:00:00', '2021-09-04 01:37:00'],
                                                       ['2021-09-04 00:02:00', '2021-09-04 01:39:00'],
                                                       ['2021-09-04 01:39:00', '2021-09-04 04:08:00'],
                                                       ['2021-09-04 05:01:00', '2021-09-04 06:38:00'],
                                                       ['2021-09-04 06:38:00', '2021-09-04 08:15:00'],
                                                       ['2021-09-04 08:15:00', '2021-09-04 10:44:00'],
                                                       ['2021-09-04 10:44:00', '2021-09-04 14:19:00'],
                                                       ['2021-09-04 14:19:00', '2021-09-04 16:48:00'],
                                                       ['2021-09-04 16:48:00', '2021-09-04 18:25:00'],
                                                       ['2021-09-04 18:25:00', '2021-09-04 20:54:00'],
                                                       ['2021-09-04 20:54:00', '2021-09-04 22:31:00'],
                                                       ['2021-09-04 22:31:00', '2021-09-05 00:08:00'],
                                                       ['2021-09-05 00:08:00', '2021-09-05 02:37:00'],
                                                       ['2021-09-05 02:37:00', '2021-09-05 04:14:00']],
                                               'M08': [],
                                               'M09': [],
                                               'M10': [],
                                               'M11': [],
                                               'M12': [['2021-09-04 08:07:00', '2021-09-04 10:31:00'],
                                                       ['2021-09-04 10:31:00', '2021-09-04 12:17:00'],
                                                       ['2021-09-04 12:17:00', '2021-09-04 14:03:00'],
                                                       ['2021-09-04 14:03:00', '2021-09-04 15:49:00'],
                                                       ['2021-09-06 04:15:00', '2021-09-06 06:39:00'],
                                                       ['2021-09-06 06:39:00', '2021-09-06 08:25:00'],
                                                       ['2021-09-06 08:25:00', '2021-09-06 10:49:00']],
                                               'M13': [['2021-09-04 01:21:00', '2021-09-04 02:50:00'],
                                                       ['2021-09-04 02:50:00', '2021-09-04 04:19:00'],
                                                       ['2021-09-04 21:00:00', '2021-09-04 22:29:00'],
                                                       ['2021-09-04 22:29:00', '2021-09-04 23:58:00'],
                                                       ['2021-09-04 23:58:00', '2021-09-05 01:27:00'],
                                                       ['2021-09-05 01:27:00', '2021-09-05 02:56:00']],
                                               'M14': [['2021-09-04 00:02:00', '2021-09-04 01:16:00'],
                                                       ['2021-09-04 01:16:00', '2021-09-04 02:30:00'],
                                                       ['2021-09-04 02:30:00', '2021-09-04 03:28:00'],
                                                       ['2021-09-04 03:28:00', '2021-09-04 04:26:00'],
                                                       ['2021-09-04 04:26:00', '2021-09-04 05:24:00'],
                                                       ['2021-09-04 05:24:00', '2021-09-04 06:38:00'],
                                                       ['2021-09-04 06:38:00', '2021-09-04 09:14:00'],
                                                       ['2021-09-04 12:05:00', '2021-09-04 14:20:00'],
                                                       ['2021-09-04 14:20:00', '2021-09-04 15:34:00'],
                                                       ['2021-09-04 15:34:00', '2021-09-04 16:48:00'],
                                                       ['2021-09-04 16:48:00', '2021-09-04 19:24:00'],
                                                       ['2021-09-04 19:24:00', '2021-09-04 20:38:00'],
                                                       ['2021-09-04 20:38:00', '2021-09-04 21:52:00'],
                                                       ['2021-09-04 22:29:00', '2021-09-04 23:43:00'],
                                                       ['2021-09-04 23:43:00', '2021-09-05 00:57:00']],
                                               'M15': [['2021-09-04 00:01:00', '2021-09-04 00:52:00'],
                                                       ['2021-09-04 00:52:00', '2021-09-04 03:28:00'],
                                                       ['2021-09-04 03:28:00', '2021-09-04 04:54:00'],
                                                       ['2021-09-04 04:54:00', '2021-09-04 07:30:00'],
                                                       ['2021-09-04 09:40:00', '2021-09-04 10:31:00'],
                                                       ['2021-09-04 10:31:00', '2021-09-04 13:31:00'],
                                                       ['2021-09-04 13:31:00', '2021-09-04 14:57:00'],
                                                       ['2021-09-04 14:57:00', '2021-09-04 15:48:00'],
                                                       ['2021-09-04 15:48:00', '2021-09-04 17:14:00'],
                                                       ['2021-09-04 17:14:00', '2021-09-04 18:05:00'],
                                                       ['2021-09-04 18:05:00', '2021-09-04 18:56:00'],
                                                       ['2021-09-04 18:56:00', '2021-09-04 19:47:00'],
                                                       ['2021-09-04 19:47:00', '2021-09-04 20:38:00'],
                                                       ['2021-09-04 20:38:00', '2021-09-04 21:29:00'],
                                                       ['2021-09-04 21:29:00', '2021-09-04 22:20:00'],
                                                       ['2021-09-04 23:43:00', '2021-09-05 00:34:00'],
                                                       ['2021-09-05 00:34:00', '2021-09-05 03:10:00'],
                                                       ['2021-09-06 05:27:00', '2021-09-06 08:03:00'],
                                                       ['2021-09-06 08:03:00', '2021-09-06 09:29:00']],
                                               'M16': [['2021-09-04 00:52:00', '2021-09-04 03:37:00'],
                                                       ['2021-09-04 03:37:00', '2021-09-04 06:35:00'],
                                                       ['2021-09-04 06:35:00', '2021-09-04 09:20:00'],
                                                       ['2021-09-04 09:20:00', '2021-09-04 12:05:00'],
                                                       ['2021-09-04 12:05:00', '2021-09-04 15:43:00'],
                                                       ['2021-09-04 15:43:00', '2021-09-04 18:28:00'],
                                                       ['2021-09-04 18:28:00', '2021-09-04 21:13:00'],
                                                       ['2021-09-04 21:13:00', '2021-09-05 00:51:00'],
                                                       ['2021-09-05 00:51:00', '2021-09-05 04:00:00'],
                                                       ['2021-09-05 04:00:00', '2021-09-05 07:09:00'],
                                                       ['2021-09-05 07:09:00', '2021-09-05 09:45:00'],
                                                       ['2021-09-05 09:45:00', '2021-09-05 12:30:00'],
                                                       ['2021-09-05 12:30:00', '2021-09-05 14:05:00'],
                                                       ['2021-09-05 14:05:00', '2021-09-05 16:41:00'],
                                                       ['2021-09-05 16:41:00', '2021-09-05 20:19:00'],
                                                       ['2021-09-05 20:19:00', '2021-09-05 23:57:00'],
                                                       ['2021-09-05 23:57:00', '2021-09-06 02:42:00'],
                                                       ['2021-09-06 02:42:00', '2021-09-06 05:27:00'],
                                                       ['2021-09-06 05:27:00', '2021-09-06 08:25:00'],
                                                       ['2021-09-06 08:25:00', '2021-09-06 11:34:00'],
                                                       ['2021-09-06 11:34:00', '2021-09-06 14:10:00'],
                                                       ['2021-09-06 14:10:00', '2021-09-06 17:08:00'],
                                                       ['2021-09-06 17:08:00', '2021-09-06 19:53:00'],
                                                       ['2021-09-06 19:53:00', '2021-09-06 22:29:00'],
                                                       ['2021-09-06 22:29:00', '2021-09-07 02:07:00'],
                                                       ['2021-09-07 02:07:00', '2021-09-07 04:52:00'],
                                                       ['2021-09-07 04:52:00', '2021-09-07 07:37:00'],
                                                       ['2021-09-07 07:37:00', '2021-09-07 10:35:00'],
                                                       ['2021-09-07 10:35:00', '2021-09-07 12:10:00'],
                                                       ['2021-09-07 12:10:00', '2021-09-07 15:19:00'],
                                                       ['2021-09-07 15:19:00', '2021-09-07 18:04:00'],
                                                       ['2021-09-07 18:04:00', '2021-09-07 19:39:00'],
                                                       ['2021-09-07 19:39:00', '2021-09-07 22:24:00'],
                                                       ['2021-09-07 22:24:00', '2021-09-08 01:22:00']],
                                               'M17': [['2021-09-04 00:02:00', '2021-09-04 01:21:00'],
                                                       ['2021-09-04 01:39:00', '2021-09-04 02:58:00'],
                                                       ['2021-09-04 08:15:00', '2021-09-04 09:34:00'],
                                                       ['2021-09-04 22:39:00', '2021-09-04 23:58:00'],
                                                       ['2021-09-05 00:08:00', '2021-09-05 01:27:00'],
                                                       ['2021-09-05 04:14:00', '2021-09-05 05:33:00']],
                                               'M18': [],
                                               'M19': [],
                                               'M20': [],
                                               'M21': [],
                                               'M22': [],
                                               'M23': [],
                                               'M24': [],
                                               'M25': [],
                                               'M26': [],
                                               'M27': [],
                                               'M28': [],
                                               'M29': [],
                                               'M30': [],
                                               'M31': [],
                                               'M32': [],
                                               'M33': [],
                                               'M34': [],
                                               'M35': [],
                                               'M36': [],
                                               'M37': [],
                                               'M38': [],
                                               'M39': [],
                                               'M40': [],
                                               'M41': [],
                                               'M42': [],
                                               'M43': [],
                                               'M44': [],
                                               'M45': [],
                                               'M46': [],
                                               'M47': []},
                'machineAval': {'M01': '2022-06-02 00:00:00',
                                'M02': '2022-06-02 00:00:00',
                                'M03': '2022-06-02 00:00:00',
                                'M04': '2022-06-02 00:00:00',
                                'M05': '2022-06-02 00:00:00',
                                'M06': '2022-06-02 00:00:00',
                                'M07': '2022-06-02 00:02:00',
                                'M08': '2022-06-02 00:00:00',
                                'M09': '2022-06-02 00:00:00',
                                'M10': '2022-06-02 00:00:00',
                                'M11': '2022-06-02 00:00:00',
                                'M12': '2022-06-02 00:00:00',
                                'M13': '2022-06-02 01:21:00',
                                'M14': '2022-06-02 00:02:00',
                                'M15': '2022-06-02 00:01:00',
                                'M16': '2022-06-02 00:52:00',
                                'M17': '2022-06-02 00:00:00',
                                'M18': '2022-06-02 00:00:00',
                                'M19': '2022-06-02 00:00:00',
                                'M20': '2022-06-02 00:00:00',
                                'M21': '2022-06-02 00:00:00',
                                'M22': '2022-06-02 00:00:00',
                                'M23': '2022-06-02 00:00:00',
                                'M24': '2022-06-02 00:00:00',
                                'M25': '2022-06-02 00:00:00',
                                'M26': '2022-06-02 00:00:00',
                                'M27': '2022-06-02 00:00:00',
                                'M28': '2022-06-02 00:00:00',
                                'M29': '2022-06-02 00:00:00',
                                'M30': '2022-06-02 00:00:00',
                                'M31': '2022-06-02 00:00:00',
                                'M32': '2022-06-02 00:00:00',
                                'M33': '2022-06-02 00:00:00',
                                'M34': '2022-06-02 00:00:00',
                                'M35': '2022-06-02 00:00:00',
                                'M36': '2022-06-02 00:00:00',
                                'M37': '2022-06-02 00:00:00',
                                'M38': '2022-06-02 00:00:00',
                                'M39': '2022-06-02 00:00:00',
                                'M40': '2022-06-02 00:00:00',
                                'M41': '2022-06-02 00:00:00',
                                'M42': '2022-06-02 00:00:00',
                                'M43': '2022-06-02 00:00:00',
                                'M44': '2022-06-02 00:00:00',
                                'M45': '2022-06-02 00:00:00',
                                'M46': '2022-06-02 00:00:00',
                                'M47': '2022-06-02 00:00:00'}
                }

    def information(data):
        planstart = data['planStart']
        planstart = dt.datetime.strptime(planstart, "%Y-%m-%d %H:%M:%S")
        span = data['periodLength']
        planspan = dt.timedelta(days=span)
        planend = planstart + planspan
        reststart = '22:00'  # 暂定为22:00
        [a, b] = reststart.split(':')
        a = int(a)
        b = int(b)
        reststart = dt.timedelta(minutes=a * 60 + b)
        restend = '24:00'  # 暂定为24:00
        [a, b] = restend.split(':')
        a = int(a)
        b = int(b)
        restend = dt.timedelta(minutes=a * 60 + b)
        # 输出每日工作时间段
        restduration = []
        for i in range(0, span):
            a = planstart + dt.timedelta(days=i) + reststart
            b = planstart + dt.timedelta(days=i) + restend
            restduration.append([a, b])
        T = data['replanTime']
        T = dt.datetime.strptime(T, "%Y-%m-%d %H:%M:%S")
        for i in range(0, span):  # 若插单时刻发生在休息时段，将其移至该休息时段末尾
            if (restduration[i][0] <= T) & (restduration[i][1] >= T):
                T = restduration[i][1]
        return planstart, planend, planspan, restduration, T

    # 按时间顺序对Q和QQ重排
    def adjust(Q, QQ):
        for key in Q.keys():
            if len(Q[key]) <= 1:
                continue
            for k in range(0, len(Q[key]) - 1):
                for j in range(k + 1, len(Q[key])):
                    a = dt.datetime.strptime(Q[key][k][0], "%Y-%m-%d %H:%M:%S")
                    b = dt.datetime.strptime(Q[key][j][0], "%Y-%m-%d %H:%M:%S")
                    if a > b:
                        temp1 = Q[key][k]
                        Q[key][k] = Q[key][j]
                        Q[key][j] = temp1
                        temp2 = QQ[key][k]
                        QQ[key][k] = QQ[key][j]
                        QQ[key][j] = temp2
        return Q, QQ

    # 分离计划中specificprocessID包含的productID和processID
    def seperate(a, b):
        if b.startswith(a):
            return b.replace(a, '', 1)

    # 判断specificprocessID（b）包含的productID是否为productID（a）
    def containjudge(a, b):
        if b.startswith(a):
            return True

    # 计算设备从紧急插单点到排产周期末的占用时间
    def caculation(T, Q, planend):  # T为发生紧急插单的时刻点，Q为原排产计划的加工时间集合
        t = {}
        for key in Q.keys():
            if Q[key] == []:  # case1
                t[key] = dt.timedelta(minutes=0)
            else:
                between = 0
                for j in range(0, len(Q[key])):
                    start = dt.datetime.strptime(Q[key][j][0], "%Y-%m-%d %H:%M:%S")
                    end = dt.datetime.strptime(Q[key][j][1], "%Y-%m-%d %H:%M:%S")
                    if (start <= T) & (end >= T):  # case2订单插入时设备在加工
                        t1 = end - T
                        # print(t1)#正在加工工序还需占用的设备时间
                        t2 = dt.timedelta(minutes=0)
                        for k in range(j + 1, len(Q[key])):
                            start1 = dt.datetime.strptime(Q[key][k][0], "%Y-%m-%d %H:%M:%S")
                            end1 = dt.datetime.strptime(Q[key][k][1], "%Y-%m-%d %H:%M:%S")
                            if end1 <= planend:
                                t2 = t2 + end1 - start1  # 重拍后超出24小时排产周期的不应当计入设备占用时间
                            elif (start1 <= planend) & (end1 > planend):  # 对于生产时间跨越排产周期结点的情况
                                t2 = t2 + planend - start1
                            else:
                                break
                        t0 = t1 + t2
                        t[key] = t0
                        between = 1
                        break
                if between == 0:
                    # 设置一个临时变量busy，对紧急插单时刻原排产计划未安排工序的设为0
                    busy = 0
                    for j in range(0, len(Q[key])):
                        start = dt.datetime.strptime(Q[key][j][0], "%Y-%m-%d %H:%M:%S")
                        if start > T:
                            t2 = dt.timedelta(minutes=0)
                            for k in range(j, len(Q[key])):
                                start1 = dt.datetime.strptime(Q[key][k][0], "%Y-%m-%d %H:%M:%S")
                                end1 = dt.datetime.strptime(Q[key][k][1], "%Y-%m-%d %H:%M:%S")
                                if end1 <= planend:
                                    t2 = t2 + end1 - start1  # 重排后超出24小时排产周期的不应当计入设备占用时间
                                else:
                                    t2 = t2 + planend - start1
                                    break
                            t[key] = t2
                            # print(t2)
                            busy = 1
                            break
                    if busy == 0:
                        t[key] = dt.timedelta(minutes=0)
        return t

    def new(MNEXTQ, MNEXTQNAME, TNEXT, PNEXT, replantime,
            restduration, J):
        QW = []  # 按设备分类的待加工工序集合
        QQW = []  # 按设备分类的待加工工序名称集合
        NW = []  # 按产品分类的待加工工序集合
        for key in MNEXTQ.keys():
            QW.append(MNEXTQ[key])
            QQW.append(MNEXTQNAME[key])
        for key in PNEXT.keys():
            nw = []
            for j in range(0, len(PNEXT[key])):
                '''
                for k in range(0, len(QQW)):
                    if PNEXT[key][j] in QQW[k]:
                '''
                nw.append(PNEXT[key][j])
            NW.append(nw)
        QQ2 = copy.deepcopy(QQW)  # 检查环节用
        # 设备空闲时间（可以开始加工的最早时间）
        TW = [TNEXT[i] for i in TNEXT.keys()]
        # 初始化产品上一工序时间
        TTW = [replantime.strftime("%Y-%m-%d %H:%M:%S") for _ in range(len(NW))]  # 初始化时间为重排时刻
        # 正式更新排产计划
        z = len(QQW)
        finalQ = [[] for _ in range(z)]  # 用来存储最后各设备各工序的加工时间
        while (QQW != [[] for _ in range(z)]):
            for i in range(0, len(QQW)):
                j = 0
                while (j < len(NW)):  # 如果产品集合未遍历完
                    if QQW[i] != []:
                        if NW[j] != []:
                            if QQW[i][0] == NW[j][0]:
                                print("当前排产工序", QQW[i][0])
                                tcost = dt.datetime.strptime(QW[i][0][1], "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(
                                    QW[i][0][0], "%Y-%m-%d %H:%M:%S")
                                a = dt.datetime.strptime(TW[i], "%Y-%m-%d %H:%M:%S")  # 设备最早可加工时间
                                b = dt.datetime.strptime(TTW[j], "%Y-%m-%d %H:%M:%S")  # 上一工序结束时间
                                if a <= b:
                                    temp1 = b
                                else:
                                    temp1 = a
                                # 判断当前安排的工序时间是否侵占了休息时间
                                x = temp1 + tcost
                                for p in range(0, len(restduration)):
                                    if restduration[p][0].day == temp1.day:
                                        if (temp1 > restduration[p][0]) & (
                                                temp1 < restduration[p][1]):  # 休息时段内才开始的，一律推迟到休息时段末
                                            temp1 = restduration[p][1]
                                            x = temp1 + tcost
                                        elif (temp1 <= restduration[p][0]) & (
                                                x > restduration[p][1]):  # 休息时段前开始，但耗完休息时段仍为加工完的，推迟至休息时段末
                                            temp1 = restduration[p][1]
                                            x = temp1 + tcost
                                        break
                                QW[i][0][0] = temp1.strftime("%Y-%m-%d %H:%M:%S")
                                QW[i][0][1] = x.strftime("%Y-%m-%d %H:%M:%S")
                                TW[i] = QW[i][0][1]  # 更新设备的最早可加工时间
                                TTW[j] = QW[i][0][1]  # 更新产品上一工序结束时间
                                finalQ[i].append(QW[i][0])  # 将安排好的工序的开始时间和结束时间放入最终顺序中
                                del QQW[i][0]
                                del QW[i][0]
                                del NW[j][0]
                                print(QQW)
                                print(QW)
                                print(NW)
                                print(finalQ)
                                j = 0  # 如果找到某一件产品的当前最前工序为当前设备的最前工序，那么当前设备的第二个工序成为最前工序，同样需要对所有产品种类进行遍历，重置产品序号为1
                            else:
                                j = j + 1  # 如果当前产品的最前工序与当前设备的最前工序不同，产品序号加1，判断下一产品的最前工序是否为当前设备的最前工序
                        else:
                            j = j + 1  # 如果当前产品的工序已经安排完，则去比对下一产品的最前工序
                    else:
                        break  # 如果某一设备的工序已被安排完，则安排下一个设备的工序
        print("重排时刻后工序名称：", QQ2)

        # 按产品检查，能否将某些工序移动至设备时间线的空闲处
        for i in range(0, len(finalQ)):
            if finalQ[i] != []:
                for j in range(1, len(finalQ[i])):  # 从设备安排的第二道工序开始，第一道工序已为最前，无法调整
                    for key_3 in J.keys():
                        if QQ2[i][j] in J[key_3]:
                            break
                    aaa = key_3
                    bbb = seperate(aaa, QQ2[i][j])
                    print("当前检查工序：", QQ2[i][j])
                    if J[aaa].index(QQ2[i][j]) != 0:
                        ab = J[aaa][J[aaa].index(QQ2[i][j]) - 1]
                        print("紧前工序名称", ab)
                        Qspan = dt.datetime.strptime(finalQ[i][j][1],
                                                     "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(
                            finalQ[i][j][0], "%Y-%m-%d %H:%M:%S")
                        print("工序耗时", Qspan)
                        lastend = '0'
                        find = '0'
                        for m in range(0, len(QQ2)):  #
                            for n in range(0, len(QQ2[m])):
                                print(m)
                                print(n)
                                if QQ2[m][n] == ab:
                                    lastend = dt.datetime.strptime(finalQ[m][n][1],
                                                                   "%Y-%m-%d %H:%M:%S")  # 记录紧前工序的结束加工时间，从这个时刻点开始向后检查空当
                                    print("紧前工序结束时间", lastend)
                                    find = '1'
                                    break
                            if (find == '1'):
                                break
                        if (lastend != '0'):  # 找到紧前工序
                            for jj in range(0, j):
                                c = dt.datetime.strptime(finalQ[i][jj][1], "%Y-%m-%d %H:%M:%S")
                                print("空当检查工序", QQ2[i][jj])
                                if c >= lastend:  # 只对该设备该工序前、紧前工序结束时刻后的工序进行检查
                                    cspan = dt.datetime.strptime(finalQ[i][jj + 1][0],
                                                                 "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(
                                        finalQ[i][jj][1], "%Y-%m-%d %H:%M:%S")
                                    print("空当起始工序", QQ2[i][jj])
                                    print("空当时间", cspan)
                                    if cspan >= Qspan:
                                        print("有空位，可前移", QQ2[i][j])
                                        finalQ[i][j][0] = finalQ[i][jj][1]
                                        cc = dt.datetime.strptime(finalQ[i][j][0], "%Y-%m-%d %H:%M:%S") + Qspan
                                        finalQ[i][j][1] = cc.strftime("%Y-%m-%d %H:%M:%S")
                                        TEMP1 = finalQ[i][j][0]
                                        TEMP2 = finalQ[i][j][1]
                                        TEMP = QQ2[i][j]
                                        for kk in range(j - 1, jj, -1):  # 将该工序移动至空当
                                            a = finalQ[i][kk][0]
                                            b = finalQ[i][kk][1]
                                            finalQ[i][kk + 1][0] = a
                                            finalQ[i][kk + 1][1] = b
                                            QQ2[i][kk + 1] = QQ2[i][kk]
                                        finalQ[i][jj + 1][0] = TEMP1
                                        finalQ[i][jj + 1][1] = TEMP2
                                        QQ2[i][jj + 1] = TEMP
                                        print("移动后的工序顺序", QQ2[i])
                                        print("移动后的工序时间集合", finalQ[i])
                                        break
        # 输出
        print("重排完成后的加工时间集合", finalQ)
        print("重排完成后的工序名称集合", QQ2)
        return finalQ, QQ2

    def home():

        planstart, planend, planspan, restduration, replantime = information(data)
        J = data['processProductBelong']

        BMOK = data['breakMachine']
        PNEXT = data['pendingProcessProduct']
        MNEXTQ = data['pendingProcessOriginalPlan']
        MNEXTNAME = data['pendingProcessMachine']
        TNEXT = data['machineAval']
        process = data['process']
        Wait = []  # 存放因无可替代设备需要暂时移除的工序（包括后续工序）
        BM = []  # 记录未修复的设备编号
        for key in BMOK.keys():
            '''
                if BMOK[key] == True:  # 设备已修复，按可选设备和待加工工序进行重排
                reQ, RETNEXT, RENEXT, reQQ = new(MNEXTQ, MNEXTNAME, TNEXT, PNEXT, replantime, restduration)
            '''
            if BMOK[key] == False:  # 将故障设备工序调整至可替代设备上，若无可替代设备则将该工序暂时移除，待故障设备修复好后插回
                BM.append(key)
                O1 = []  # 故障设备待加工工序信息[工序名称，原计划加工设备，可替代设备集合，加工耗时集合]
                O2 = []  # 故障设备待加工工序的后续工序[工序名称，原计划加工设备，可替代设备集合，加工耗时集合]
                for i in range(len(MNEXTQ[key]) - 1, -1, -1):
                    aaa = MNEXTNAME[key][i]
                    O3 = []
                    for key_1 in J.keys():
                        if containjudge(key_1, aaa):
                            break
                    for kk in range(len(J[key_1])):
                        if J[key_1][kk] == aaa:
                            break
                    O1.append([aaa, key, process[key_1]['machineID'][kk], process[key_1]['processTime'][kk]])
                    for j in range(kk + 1, len(J[key_1])):
                        bbb = J[key_1][j]
                        for key_2 in process[key_1]['machineID'][j]:
                            if bbb in MNEXTNAME[key_2]:
                                boriginalmachine = key_2
                                O3.append([bbb, boriginalmachine, process[key_1]['machineID'][j],
                                           process[key_1]['processTime'][j]])  # 只考虑在待加工工序集合内的后续加工工序
                    O2.append(O3)
        for i in range(len(O1)):
            if len(list(set(O1[i][2]).difference(set(BM)))) == 0:  # 除未修复的设备外无可替代设备
                skip = 0
                for j in range(0, len(Wait)):
                    if O1[i] in Wait[j]:
                        skip = 1
                        O1[i] = []  # 该工序在之前工序对应的后续工序中，所以将该工序对应的O1位置置为空，无需重复考虑
                if skip != 1:
                    if O2 != []:
                        O1[i] = [O1[i]] + O2[i]
                    else:
                        O1[i] = [O1[i]]
                    Wait.append(O1[i])  # 将无可替代设备的工序信息移入# 将故障设备上无可替代设备的工序的后续工序信息也移入Wait中
                    # 同时从生产计划中移除因无可替代设备放入Wait集合中的工序及其后续工序，更新MNEXTNAME、MNEXTQ、PNEXT
                    for key_2 in PNEXT.keys():
                        if containjudge(key_2, O1[i][0][0]):
                            for z in range(0, len(PNEXT[key_2])):
                                if PNEXT[key_2][z] == O1[i][0][0]:
                                    PNEXT[key_2] = PNEXT[key_2][:z]
                                    break
                            break
                    for r in range(0, len(O1[i])):
                        mindex = O1[i][r][1]
                        for j in range(len(MNEXTNAME[mindex]) - 1, -1, -1):
                            if MNEXTNAME[mindex][j] == O1[i][r][0]:
                                MNEXTNAME[mindex].remove(MNEXTNAME[mindex][j])
                                MNEXTQ[mindex].remove(MNEXTQ[mindex][j])
                                break
                O1[i] = []
            else:
                A = copy.deepcopy(O1[i][2])
                B = copy.deepcopy(O1[i][3])
                for w in BM:
                    if w in A:
                        index = A.index[w]
                        del A[index]
                        del B[index]
                O1[i][2] = copy.deepcopy(A)
                O1[i][3] = copy.deepcopy(B)  # 从可替代设备及耗时集合中去掉未修复设备的信息，以免在重新安排故障设备工序时又选到故障设备上
        for i in range(len(O1) - 1, -1, -1):
            if O1[i] == []:
                del O1[i]
        # emptycheck = 0  # 看重拍工序是否全为空，全为空时为0
        for i in range(0, len(O1)):
            if O1[i] != []:
                # emptycheck = 1
                if len(O1[i][2]) == 1:  # 如果仅存在一个可替代设备
                    # 将该工序移至该设备的最前面，其他工序后移
                    duration = dt.timedelta(minutes=O1[i][3][0])
                    newend = replantime + duration
                    In = [replantime.strftime("%Y-%m-%d %H:%M:%S"), newend.strftime("%Y-%m-%d %H:%M:%S")]
                    ii = MNEXTNAME[O1[i][1]].index(O1[i][0])
                    del MNEXTNAME[O1[i][1]][ii]
                    del MNEXTQ[O1[i][1]][ii]  # 将工序从故障设备上移除
                    IMindex = O1[i][2][0]  # 插入的目标设备
                    MNEXTQ[IMindex].insert(0, In)
                    MNEXTNAME[IMindex].insert(0, O1[i][0])  # 插入到最前面
                    for k in range(1, len(MNEXTQ[IMindex])):  # 其余工序后移
                        start = dt.datetime.strptime(MNEXTQ[IMindex][k][0], "%Y-%m-%d %H:%M:%S")
                        end = dt.datetime.strptime(MNEXTQ[IMindex][k][1], "%Y-%m-%d %H:%M:%S")
                        restart = dt.datetime.strptime(MNEXTQ[IMindex][k - 1][1], "%Y-%m-%d %H:%M:%S")
                        reend = restart + end - start
                        MNEXTQ[IMindex][k][0] = restart.strftime("%Y-%m-%d %H:%M:%S")
                        MNEXTQ[IMindex][k][1] = reend.strftime("%Y-%m-%d %H:%M:%S")
                    MNEXTQ, MNEXTNAME = adjust(MNEXTQ, MNEXTNAME)
                else:  # 若存在多个可替代设备，选择设备占用时间少的作为插入的设备
                    # 下面caculation函数中的第一个参数最后做入系统时，是读取系统运行时日期的零时刻点，在调试阶段暂时使用自行输入的重排时刻replantime
                    t = caculation(replantime, MNEXTQ, planend)
                    t0 = dt.timedelta(minutes=10080)
                    imindex = 0  # 初始化选中设备在可选设备列表中的下标
                    for j in range(0, len(O1[i][2])):
                        jj = O1[i][2][j]
                        tt = t[jj]
                        if tt < t0:
                            t0 = tt
                            imindex = jj
                    # print(IMindex)
                    # 将该工序移至该设备的最前面，其他工序后移
                    duration = dt.timedelta(minutes=O1[i][3][imindex])
                    newend = replantime + duration
                    In = [replantime.strftime("%Y-%m-%d %H:%M:%S"), newend.strftime("%Y-%m-%d %H:%M:%S")]
                    ii = MNEXTNAME[O1[i][1]].index(O1[i][0])
                    del MNEXTNAME[O1[i][1]][ii]
                    del MNEXTQ[O1[i][1]][ii]  # 将工序从故障设备上移除
                    IMindex = O1[i][2][imindex]  # 插入的目标设备
                    MNEXTQ[IMindex].insert(0, In)
                    MNEXTNAME[IMindex].insert(0, O1[i][0])  # 插入到最前面
                    for k in range(1, len(MNEXTQ[IMindex])):  # 其余工序后移
                        start = dt.datetime.strptime(MNEXTQ[IMindex][k][0], "%Y-%m-%d %H:%M:%S")
                        end = dt.datetime.strptime(MNEXTQ[IMindex][k][1], "%Y-%m-%d %H:%M:%S")
                        restart = dt.datetime.strptime(MNEXTQ[IMindex][k - 1][1], "%Y-%m-%d %H:%M:%S")
                        reend = restart + end - start
                        MNEXTQ[IMindex][k][0] = restart.strftime("%Y-%m-%d %H:%M:%S")
                        MNEXTQ[IMindex][k][1] = reend.strftime("%Y-%m-%d %H:%M:%S")
                    MNEXTQ, MNEXTNAME = adjust(MNEXTQ, MNEXTNAME)
        # if emptycheck == 1:
        reQ, reQQ = new(MNEXTQ, MNEXTNAME, TNEXT, PNEXT, replantime, restduration, J)

        replanProcessName = {}
        replanProcessTime = {}
        j = -1
        for key in MNEXTQ.keys():
            j = j + 1
            replanProcessTime[key] = reQ[j]
            replanProcessName[key] = reQQ[j]
        respond = {'replanProcessName': replanProcessName,
                   'replanProcessTime': replanProcessTime,
                   'replanProcessOthers': {}}  # 暂时定为空集

        return json.dumps(respond)

    return home()


################################################
# 设备故障结束
# ######################################
# 质量不合格
# ###################################

@app.route('/disqualify', methods=['GET', 'POST'])
def disqualify():
    if request.method == 'POST':
        data = request.get_data()
        data = json.loads(data)
    else:
        data = {'hours': 22,
                'periodLength': 7,
                'planStart': '2022-06-01 00:00:00',
                'process': {'P0001': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0002': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0003': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0004': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0005': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0006': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0007': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0008': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0009': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0010': {'machineID': [['M12', 'M13']], 'machinePriority': [[5, 5]],
                                      'processTime': [[18.0, 18.0]]},
                            'P0011': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0012': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0013': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0014': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0015': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0016': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0017': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0018': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0019': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0020': {
                                'machineID': [['M06'], ['M16'], ['M16'], ['M12'], ['M15'], ['M14'], ['M12'], ['M12']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[142.0], [189.0], [178.0], [144.0], [180.0], [135.0], [144.0],
                                                [144.0]]},
                            'P0021': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0022': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0023': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0024': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0025': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0026': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0027': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0028': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0029': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0030': {'machineID': [['M26'], ['M26'], ['M26'], ['M26'], ['M26'], ['M26']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5]],
                                      'processTime': [[69.0], [69.0], [69.0], [69.0], [69.0], [18.0]]},
                            'P0031': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0032': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0033': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0034': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0035': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0036': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0037': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0038': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0039': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0040': {'machineID': [['M16', 'M17', 'M46']], 'machinePriority': [[5, 5, 5]],
                                      'processTime': [[41.25, 41.25, 31.53]]},
                            'P0041': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0042': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0043': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0044': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0045': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0046': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0047': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0048': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0049': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0050': {'machineID': [['M06'], ['M12'], ['M12'], ['M14'], ['M15'], ['M16'], ['M16']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[168.23], [106.8], [106.8], [156.93], [86.53], [95.13], [95.13]]},
                            'P0051': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0052': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0053': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0054': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0055': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0056': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0057': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0058': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0059': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0060': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[156.42], [156.42], [156.42], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0061': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0062': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0063': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0064': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0065': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0066': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0067': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0068': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0069': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0070': {'machineID': [['M16'], ['M16'], ['M16'], ['M15'], ['M14'], ['M07'], ['M07']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[165.43], [165.43], [165.43], [156.8], [58.21], [149.08],
                                                      [149.08]]},
                            'P0071': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0072': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0073': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0074': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0075': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0076': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0077': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0078': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0079': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0080': {'machineID': [['M07'], ['M16'], ['M16'], ['M15'], ['M14'], ['M12'], ['M12']],
                                      'machinePriority': [[5], [5], [5], [5], [5], [5], [5]],
                                      'processTime': [[215.3], [218.19], [218.19], [193.46], [117.2], [126.845],
                                                      [126.845]]},
                            'P0081': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0082': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0083': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0084': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0085': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0086': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0087': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0088': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0089': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0090': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[79.2], [89.195], [74.905], [97.28], [79.2], [89.195], [74.905],
                                                [51.815], [51.815], [97.28]]},
                            'P0091': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0092': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0093': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0094': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0095': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0096': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0097': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0098': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0099': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]},
                            'P0100': {
                                'machineID': [['M17'], ['M13'], ['M14'], ['M07'], ['M17'], ['M13'], ['M14'], ['M15'],
                                              ['M15'], ['M07']],
                                'machinePriority': [[5], [5], [5], [5], [5], [5], [5], [5], [5], [5]],
                                'processTime': [[115.265], [110.315], [133.71], [163.05], [115.265], [110.315],
                                                [133.71], [156.24], [156.24], [163.05]]}},
                'replanTime': '2022-06-02 00:00:00',
                'pendingProcessMachine': {'M01': [],
                                          'M02': [],
                                          'M03': [],
                                          'M04': [],
                                          'M05': [],
                                          'M06': ['P0011p001',
                                                  'P0012p001',
                                                  'P0013p001',
                                                  'P0014p001',
                                                  'P0015p001',
                                                  'P0016p001',
                                                  'P0017p001',
                                                  'P0041p001',
                                                  'P0042p001',
                                                  'P0043p001',
                                                  'P0044p001',
                                                  'P0045p001',
                                                  'P0046p001',
                                                  'P0047p001',
                                                  'P0048p001'],
                                          'M07': ['P0051p006',
                                                  'P0051p007',
                                                  'P0052p006',
                                                  'P0052p007',
                                                  'P0061p006',
                                                  'P0061p007',
                                                  'P0071p001',
                                                  'P0072p001',
                                                  'P0073p001',
                                                  'P0074p001',
                                                  'P0075p001',
                                                  'P0076p001',
                                                  'P0081p004',
                                                  'P0081p010',
                                                  'P0082p004',
                                                  'P0082p010',
                                                  'P0083p004',
                                                  'P0084p004',
                                                  'P0085p004',
                                                  'P0086p004',
                                                  'P0087p004',
                                                  'P0088p004',
                                                  'P0089p004',
                                                  'P0090p004'],
                                          'M08': [],
                                          'M09': [],
                                          'M10': [],
                                          'M11': [],
                                          'M12': ['P0011p004',
                                                  'P0011p007',
                                                  'P0012p004',
                                                  'P0013p004',
                                                  'P0041p002',
                                                  'P0041p003',
                                                  'P0042p002',
                                                  'P0042p003',
                                                  'P0043p002',
                                                  'P0043p003',
                                                  'P0044p002',
                                                  'P0044p003',
                                                  'P0045p002',
                                                  'P0045p003',
                                                  'P0046p002',
                                                  'P0046p003',
                                                  'P0047p002',
                                                  'P0047p003'],
                                          'M13': ['P0001p001',
                                                  'P0002p001',
                                                  'P0003p001',
                                                  'P0004p001',
                                                  'P0005p001',
                                                  'P0006p001',
                                                  'P0007p001',
                                                  'P0008p001',
                                                  'P0009p001',
                                                  'P0010p001',
                                                  'P0081p002',
                                                  'P0081p006',
                                                  'P0082p002',
                                                  'P0082p006',
                                                  'P0083p002',
                                                  'P0083p006',
                                                  'P0084p002',
                                                  'P0084p006',
                                                  'P0085p002',
                                                  'P0085p006',
                                                  'P0086p002',
                                                  'P0086p006',
                                                  'P0087p002',
                                                  'P0087p006',
                                                  'P0088p002',
                                                  'P0088p006',
                                                  'P0089p002',
                                                  'P0089p006',
                                                  'P0090p002'],
                                          'M14': ['P0011p006',
                                                  'P0041p004',
                                                  'P0042p004',
                                                  'P0043p004',
                                                  'P0044p004',
                                                  'P0045p004',
                                                  'P0051p005',
                                                  'P0052p005',
                                                  'P0053p005',
                                                  'P0061p005',
                                                  'P0081p003',
                                                  'P0081p007',
                                                  'P0082p003',
                                                  'P0082p007',
                                                  'P0083p003',
                                                  'P0083p007',
                                                  'P0084p003',
                                                  'P0084p007',
                                                  'P0085p003',
                                                  'P0085p007',
                                                  'P0086p003',
                                                  'P0086p007',
                                                  'P0087p003',
                                                  'P0088p003',
                                                  'P0089p003',
                                                  'P0090p003'],
                                          'M15': ['P0011p005',
                                                  'P0012p005',
                                                  'P0041p005',
                                                  'P0042p005',
                                                  'P0043p005',
                                                  'P0044p005',
                                                  'P0045p005',
                                                  'P0051p004',
                                                  'P0052p004',
                                                  'P0053p004',
                                                  'P0054p004',
                                                  'P0061p004',
                                                  'P0062p004',
                                                  'P0063p004',
                                                  'P0064p004',
                                                  'P0081p008',
                                                  'P0081p009',
                                                  'P0082p008',
                                                  'P0082p009',
                                                  'P0083p008',
                                                  'P0083p009',
                                                  'P0084p008',
                                                  'P0084p009',
                                                  'P0085p008',
                                                  'P0085p009',
                                                  'P0086p008'],
                                          'M16': ['P0011p002',
                                                  'P0011p003',
                                                  'P0012p002',
                                                  'P0012p003',
                                                  'P0013p002',
                                                  'P0013p003',
                                                  'P0014p002',
                                                  'P0014p003',
                                                  'P0015p002',
                                                  'P0015p003',
                                                  'P0016p002',
                                                  'P0016p003',
                                                  'P0041p006',
                                                  'P0041p007',
                                                  'P0042p006',
                                                  'P0043p006',
                                                  'P0051p001',
                                                  'P0051p002',
                                                  'P0051p003',
                                                  'P0052p001',
                                                  'P0052p002',
                                                  'P0052p003',
                                                  'P0053p001',
                                                  'P0053p002',
                                                  'P0053p003',
                                                  'P0054p001',
                                                  'P0054p002',
                                                  'P0054p003',
                                                  'P0055p001',
                                                  'P0055p002',
                                                  'P0055p003',
                                                  'P0056p001',
                                                  'P0056p002',
                                                  'P0057p001',
                                                  'P0061p001',
                                                  'P0061p002',
                                                  'P0061p003',
                                                  'P0062p001',
                                                  'P0062p002',
                                                  'P0062p003',
                                                  'P0063p001',
                                                  'P0063p002',
                                                  'P0063p003',
                                                  'P0064p001',
                                                  'P0064p002',
                                                  'P0064p003',
                                                  'P0065p001',
                                                  'P0065p002',
                                                  'P0065p003',
                                                  'P0066p001',
                                                  'P0066p002',
                                                  'P0067p001',
                                                  'P0068p001',
                                                  'P0069p001',
                                                  'P0070p001',
                                                  'P0071p002',
                                                  'P0071p003',
                                                  'P0072p002',
                                                  'P0072p003',
                                                  'P0073p002',
                                                  'P0074p002'],
                                          'M17': ['P0031p001',
                                                  'P0038p001',
                                                  'P0081p001',
                                                  'P0081p005',
                                                  'P0082p001',
                                                  'P0082p005',
                                                  'P0083p001',
                                                  'P0083p005',
                                                  'P0084p001',
                                                  'P0084p005',
                                                  'P0085p001',
                                                  'P0085p005',
                                                  'P0086p001',
                                                  'P0086p005',
                                                  'P0087p001',
                                                  'P0087p005',
                                                  'P0088p001',
                                                  'P0088p005',
                                                  'P0089p001',
                                                  'P0089p005',
                                                  'P0090p001',
                                                  'P0090p005'],
                                          'M18': [],
                                          'M19': [],
                                          'M20': [],
                                          'M21': [],
                                          'M22': [],
                                          'M23': [],
                                          'M24': [],
                                          'M25': [],
                                          'M26': ['P0021p001', 'P0021p002', 'P0022p001', 'P0023p001'],
                                          'M27': [],
                                          'M28': [],
                                          'M29': [],
                                          'M30': [],
                                          'M31': [],
                                          'M32': [],
                                          'M33': [],
                                          'M34': [],
                                          'M35': [],
                                          'M36': [],
                                          'M37': [],
                                          'M38': [],
                                          'M39': [],
                                          'M40': [],
                                          'M41': [],
                                          'M42': [],
                                          'M43': [],
                                          'M44': [],
                                          'M45': [],
                                          'M46': ['P0032p001',
                                                  'P0033p001',
                                                  'P0034p001',
                                                  'P0035p001',
                                                  'P0036p001',
                                                  'P0037p001',
                                                  'P0039p001',
                                                  'P0040p001'],
                                          'M47': []},
                'pendingProcessOriginalPlan': {'M01': [], 'M02': [], 'M03': [], 'M04': [], 'M05': [],
                                               'M06': [['2022-06-01 00:00:00', '2022-06-01 02:22:00'],
                                                       ['2022-06-01 07:58:00', '2022-06-01 10:20:00'],
                                                       ['2022-06-01 10:20:00', '2022-06-01 12:42:00'],
                                                       ['2022-06-01 18:17:00', '2022-06-01 20:39:00'],
                                                       ['2022-06-01 23:27:00', '2022-06-02 01:49:00'],
                                                       ['2022-06-02 04:37:00', '2022-06-02 06:59:00'],
                                                       ['2022-06-02 09:47:00', '2022-06-02 12:09:00'],
                                                       ['2022-06-01 02:22:00', '2022-06-01 05:10:00'],
                                                       ['2022-06-01 05:10:00', '2022-06-01 07:58:00'],
                                                       ['2022-06-01 12:41:00', '2022-06-01 15:29:00'],
                                                       ['2022-06-01 15:29:00', '2022-06-01 18:17:00'],
                                                       ['2022-06-01 20:39:00', '2022-06-01 23:27:00'],
                                                       ['2022-06-02 01:49:00', '2022-06-02 04:37:00'],
                                                       ['2022-06-02 06:59:00', '2022-06-02 09:47:00'],
                                                       ['2022-06-02 12:09:00', '2022-06-02 14:57:00']],
                                               'M07': [['2022-06-03 19:56:00', '2022-06-03 22:25:00'],
                                                       ['2022-06-04 01:39:00', '2022-06-04 04:08:00'],
                                                       ['2022-06-04 14:19:00', '2022-06-04 16:48:00'],
                                                       ['2022-06-05 00:08:00', '2022-06-05 02:37:00'],
                                                       ['2022-06-04 08:15:00', '2022-06-04 10:44:00'],
                                                       ['2022-06-04 18:25:00', '2022-06-04 20:54:00'],
                                                       ['2022-06-01 00:00:00', '2022-06-01 03:35:00'],
                                                       ['2022-06-01 06:08:00', '2022-06-01 09:43:00'],
                                                       ['2022-06-01 11:20:00', '2022-06-01 14:55:00'],
                                                       ['2022-06-01 16:32:00', '2022-06-01 20:07:00'],
                                                       ['2022-06-01 21:44:00', '2022-06-02 01:19:00'],
                                                       ['2022-06-04 10:44:00', '2022-06-04 14:19:00'],
                                                       ['2022-06-01 04:31:00', '2022-06-01 06:08:00'],
                                                       ['2022-06-04 05:01:00', '2022-06-04 06:38:00'],
                                                       ['2022-06-01 09:43:00', '2022-06-01 11:20:00'],
                                                       ['2022-06-04 20:54:00', '2022-06-04 22:31:00'],
                                                       ['2022-06-01 14:55:00', '2022-06-01 16:32:00'],
                                                       ['2022-06-01 20:07:00', '2022-06-01 21:44:00'],
                                                       ['2022-06-03 22:25:00', '2022-06-04 00:02:00'],
                                                       ['2022-06-04 00:02:00', '2022-06-04 01:39:00'],
                                                       ['2022-06-04 06:38:00', '2022-06-04 08:15:00'],
                                                       ['2022-06-04 16:48:00', '2022-06-04 18:25:00'],
                                                       ['2022-06-04 22:31:00', '2022-06-05 00:08:00'],
                                                       ['2022-06-05 02:37:00', '2022-06-05 04:14:00']], 'M08': [],
                                               'M09': [],
                                               'M10': [], 'M11': [],
                                               'M12': [['2022-06-03 07:23:00', '2022-06-03 09:47:00'],
                                                       ['2022-06-06 04:15:00', '2022-06-06 06:39:00'],
                                                       ['2022-06-04 08:07:00', '2022-06-04 10:31:00'],
                                                       ['2022-06-06 08:25:00', '2022-06-06 10:49:00'],
                                                       ['2022-06-01 05:10:00', '2022-06-01 06:56:00'],
                                                       ['2022-06-01 06:56:00', '2022-06-01 08:42:00'],
                                                       ['2022-06-01 08:41:00', '2022-06-01 10:27:00'],
                                                       ['2022-06-01 10:27:00', '2022-06-01 12:13:00'],
                                                       ['2022-06-01 15:29:00', '2022-06-01 17:15:00'],
                                                       ['2022-06-01 17:15:00', '2022-06-01 19:01:00'],
                                                       ['2022-06-01 19:01:00', '2022-06-01 20:47:00'],
                                                       ['2022-06-03 09:47:00', '2022-06-03 11:33:00'],
                                                       ['2022-06-03 11:33:00', '2022-06-03 13:19:00'],
                                                       ['2022-06-04 10:31:00', '2022-06-04 12:17:00'],
                                                       ['2022-06-03 13:19:00', '2022-06-03 15:05:00'],
                                                       ['2022-06-04 12:17:00', '2022-06-04 14:03:00'],
                                                       ['2022-06-04 14:03:00', '2022-06-04 15:49:00'],
                                                       ['2022-06-06 06:39:00', '2022-06-06 08:25:00']],
                                               'M13': [['2022-06-01 00:00:00', '2022-06-01 00:18:00'],
                                                       ['2022-06-01 00:18:00', '2022-06-01 00:36:00'],
                                                       ['2022-06-01 00:36:00', '2022-06-01 00:54:00'],
                                                       ['2022-06-01 00:54:00', '2022-06-01 01:12:00'],
                                                       ['2022-06-01 01:12:00', '2022-06-01 01:30:00'],
                                                       ['2022-06-01 01:30:00', '2022-06-01 01:48:00'],
                                                       ['2022-06-01 03:17:00', '2022-06-01 03:35:00'],
                                                       ['2022-06-01 03:35:00', '2022-06-01 03:53:00'],
                                                       ['2022-06-03 11:24:00', '2022-06-03 11:42:00'],
                                                       ['2022-06-03 11:42:00', '2022-06-03 12:00:00'],
                                                       ['2022-06-01 01:48:00', '2022-06-01 03:17:00'],
                                                       ['2022-06-03 12:00:00', '2022-06-03 13:29:00'],
                                                       ['2022-06-01 03:53:00', '2022-06-01 05:22:00'],
                                                       ['2022-06-03 19:25:00', '2022-06-03 20:54:00'],
                                                       ['2022-06-01 05:59:00', '2022-06-01 07:28:00'],
                                                       ['2022-06-03 22:23:00', '2022-06-03 23:52:00'],
                                                       ['2022-06-01 07:28:00', '2022-06-01 08:57:00'],
                                                       ['2022-06-03 23:52:00', '2022-06-04 01:21:00'],
                                                       ['2022-06-03 13:29:00', '2022-06-03 14:58:00'],
                                                       ['2022-06-04 01:21:00', '2022-06-04 02:50:00'],
                                                       ['2022-06-03 14:58:00', '2022-06-03 16:27:00'],
                                                       ['2022-06-04 21:00:00', '2022-06-04 22:29:00'],
                                                       ['2022-06-03 16:27:00', '2022-06-03 17:56:00'],
                                                       ['2022-06-04 22:29:00', '2022-06-04 23:58:00'],
                                                       ['2022-06-03 17:56:00', '2022-06-03 19:25:00'],
                                                       ['2022-06-04 23:58:00', '2022-06-05 01:27:00'],
                                                       ['2022-06-03 20:54:00', '2022-06-03 22:23:00'],
                                                       ['2022-06-05 01:27:00', '2022-06-05 02:56:00'],
                                                       ['2022-06-04 02:50:00', '2022-06-04 04:19:00']],
                                               'M14': [['2022-06-04 12:05:00', '2022-06-04 14:20:00'],
                                                       ['2022-06-01 08:42:00', '2022-06-01 11:18:00'],
                                                       ['2022-06-03 16:22:00', '2022-06-03 18:58:00'],
                                                       ['2022-06-03 20:12:00', '2022-06-03 22:48:00'],
                                                       ['2022-06-04 06:38:00', '2022-06-04 09:14:00'],
                                                       ['2022-06-04 16:48:00', '2022-06-04 19:24:00'],
                                                       ['2022-06-03 18:58:00', '2022-06-03 19:56:00'],
                                                       ['2022-06-04 02:30:00', '2022-06-04 03:28:00'],
                                                       ['2022-06-04 04:26:00', '2022-06-04 05:24:00'],
                                                       ['2022-06-04 03:28:00', '2022-06-04 04:26:00'],
                                                       ['2022-06-01 03:17:00', '2022-06-01 04:31:00'],
                                                       ['2022-06-03 13:29:00', '2022-06-03 14:43:00'],
                                                       ['2022-06-01 05:22:00', '2022-06-01 06:36:00'],
                                                       ['2022-06-04 00:02:00', '2022-06-04 01:16:00'],
                                                       ['2022-06-01 07:28:00', '2022-06-01 08:42:00'],
                                                       ['2022-06-04 01:16:00', '2022-06-04 02:30:00'],
                                                       ['2022-06-01 11:18:00', '2022-06-01 12:32:00'],
                                                       ['2022-06-04 14:20:00', '2022-06-04 15:34:00'],
                                                       ['2022-06-03 14:58:00', '2022-06-03 16:12:00'],
                                                       ['2022-06-04 19:24:00', '2022-06-04 20:38:00'],
                                                       ['2022-06-03 22:48:00', '2022-06-04 00:02:00'],
                                                       ['2022-06-04 22:29:00', '2022-06-04 23:43:00'],
                                                       ['2022-06-04 05:24:00', '2022-06-04 06:38:00'],
                                                       ['2022-06-04 15:34:00', '2022-06-04 16:48:00'],
                                                       ['2022-06-04 20:38:00', '2022-06-04 21:52:00'],
                                                       ['2022-06-04 23:43:00', '2022-06-05 00:57:00']],
                                               'M15': [['2022-06-03 09:47:00', '2022-06-03 12:47:00'],
                                                       ['2022-06-04 10:31:00', '2022-06-04 13:31:00'],
                                                       ['2022-06-02 18:29:00', '2022-06-02 19:55:00'],
                                                       ['2022-06-04 03:28:00', '2022-06-04 04:54:00'],
                                                       ['2022-06-04 13:31:00', '2022-06-04 14:57:00'],
                                                       ['2022-06-04 15:48:00', '2022-06-04 17:14:00'],
                                                       ['2022-06-06 08:03:00', '2022-06-06 09:29:00'],
                                                       ['2022-06-01 13:41:00', '2022-06-01 16:17:00'],
                                                       ['2022-06-02 19:55:00', '2022-06-02 22:31:00'],
                                                       ['2022-06-02 22:31:00', '2022-06-03 01:07:00'],
                                                       ['2022-06-03 17:56:00', '2022-06-03 20:32:00'],
                                                       ['2022-06-04 00:52:00', '2022-06-04 03:28:00'],
                                                       ['2022-06-04 04:54:00', '2022-06-04 07:30:00'],
                                                       ['2022-06-05 00:34:00', '2022-06-05 03:10:00'],
                                                       ['2022-06-06 05:27:00', '2022-06-06 08:03:00'],
                                                       ['2022-06-03 23:10:00', '2022-06-04 00:01:00'],
                                                       ['2022-06-04 00:01:00', '2022-06-04 00:52:00'],
                                                       ['2022-06-04 09:40:00', '2022-06-04 10:31:00'],
                                                       ['2022-06-04 17:14:00', '2022-06-04 18:05:00'],
                                                       ['2022-06-04 14:57:00', '2022-06-04 15:48:00'],
                                                       ['2022-06-04 18:56:00', '2022-06-04 19:47:00'],
                                                       ['2022-06-04 18:05:00', '2022-06-04 18:56:00'],
                                                       ['2022-06-04 19:47:00', '2022-06-04 20:38:00'],
                                                       ['2022-06-04 20:38:00', '2022-06-04 21:29:00'],
                                                       ['2022-06-04 21:29:00', '2022-06-04 22:20:00'],
                                                       ['2022-06-04 23:43:00', '2022-06-05 00:34:00']],
                                               'M16': [['2022-06-01 07:56:00', '2022-06-01 11:05:00'],
                                                       ['2022-06-03 04:25:00', '2022-06-03 07:23:00'],
                                                       ['2022-06-02 22:31:00', '2022-06-03 01:40:00'],
                                                       ['2022-06-04 03:37:00', '2022-06-04 06:35:00'],
                                                       ['2022-06-05 00:51:00', '2022-06-05 04:00:00'],
                                                       ['2022-06-06 05:27:00', '2022-06-06 08:25:00'],
                                                       ['2022-06-05 04:00:00', '2022-06-05 07:09:00'],
                                                       ['2022-06-06 14:10:00', '2022-06-06 17:08:00'],
                                                       ['2022-06-06 08:25:00', '2022-06-06 11:34:00'],
                                                       ['2022-06-07 07:37:00', '2022-06-07 10:35:00'],
                                                       ['2022-06-07 12:10:00', '2022-06-07 15:19:00'],
                                                       ['2022-06-07 22:24:00', '2022-06-08 01:22:00'],
                                                       ['2022-06-03 17:56:00', '2022-06-03 19:31:00'],
                                                       ['2022-06-05 12:30:00', '2022-06-05 14:05:00'],
                                                       ['2022-06-07 10:35:00', '2022-06-07 12:10:00'],
                                                       ['2022-06-07 18:04:00', '2022-06-07 19:39:00'],
                                                       ['2022-06-01 00:00:00', '2022-06-01 02:36:00'],
                                                       ['2022-06-01 05:20:00', '2022-06-01 07:56:00'],
                                                       ['2022-06-01 11:05:00', '2022-06-01 13:41:00'],
                                                       ['2022-06-01 16:26:00', '2022-06-01 19:02:00'],
                                                       ['2022-06-01 21:38:00', '2022-06-02 00:14:00'],
                                                       ['2022-06-02 17:19:00', '2022-06-02 19:55:00'],
                                                       ['2022-06-01 19:02:00', '2022-06-01 21:38:00'],
                                                       ['2022-06-02 11:05:00', '2022-06-02 13:41:00'],
                                                       ['2022-06-02 19:55:00', '2022-06-02 22:31:00'],
                                                       ['2022-06-02 08:29:00', '2022-06-02 11:05:00'],
                                                       ['2022-06-03 07:23:00', '2022-06-03 09:59:00'],
                                                       ['2022-06-03 15:20:00', '2022-06-03 17:56:00'],
                                                       ['2022-06-03 09:59:00', '2022-06-03 12:35:00'],
                                                       ['2022-06-05 07:09:00', '2022-06-05 09:45:00'],
                                                       ['2022-06-05 14:05:00', '2022-06-05 16:41:00'],
                                                       ['2022-06-03 19:31:00', '2022-06-03 22:07:00'],
                                                       ['2022-06-06 11:34:00', '2022-06-06 14:10:00'],
                                                       ['2022-06-06 19:53:00', '2022-06-06 22:29:00'],
                                                       ['2022-06-01 02:35:00', '2022-06-01 05:20:00'],
                                                       ['2022-06-02 00:14:00', '2022-06-02 02:59:00'],
                                                       ['2022-06-03 22:07:00', '2022-06-04 00:52:00'],
                                                       ['2022-06-01 13:41:00', '2022-06-01 16:26:00'],
                                                       ['2022-06-02 05:44:00', '2022-06-02 08:29:00'],
                                                       ['2022-06-04 00:52:00', '2022-06-04 03:37:00'],
                                                       ['2022-06-02 02:59:00', '2022-06-02 05:44:00'],
                                                       ['2022-06-03 01:40:00', '2022-06-03 04:25:00'],
                                                       ['2022-06-04 09:20:00', '2022-06-04 12:05:00'],
                                                       ['2022-06-03 12:35:00', '2022-06-03 15:20:00'],
                                                       ['2022-06-05 09:45:00', '2022-06-05 12:30:00'],
                                                       ['2022-06-06 02:42:00', '2022-06-06 05:27:00'],
                                                       ['2022-06-04 06:35:00', '2022-06-04 09:20:00'],
                                                       ['2022-06-05 23:57:00', '2022-06-06 02:42:00'],
                                                       ['2022-06-06 17:08:00', '2022-06-06 19:53:00'],
                                                       ['2022-06-04 15:43:00', '2022-06-04 18:28:00'],
                                                       ['2022-06-07 15:19:00', '2022-06-07 18:04:00'],
                                                       ['2022-06-04 18:28:00', '2022-06-04 21:13:00'],
                                                       ['2022-06-07 02:07:00', '2022-06-07 04:52:00'],
                                                       ['2022-06-07 04:52:00', '2022-06-07 07:37:00'],
                                                       ['2022-06-07 19:39:00', '2022-06-07 22:24:00'],
                                                       ['2022-06-02 13:41:00', '2022-06-02 17:19:00'],
                                                       ['2022-06-04 21:13:00', '2022-06-05 00:51:00'],
                                                       ['2022-06-04 12:05:00', '2022-06-04 15:43:00'],
                                                       ['2022-06-06 22:29:00', '2022-06-07 02:07:00'],
                                                       ['2022-06-05 16:41:00', '2022-06-05 20:19:00'],
                                                       ['2022-06-05 20:19:00', '2022-06-05 23:57:00']],
                                               'M17': [['2022-06-01 01:19:00', '2022-06-01 02:00:00'],
                                                       ['2022-06-01 05:57:00', '2022-06-01 06:38:00'],
                                                       ['2022-06-01 00:00:00', '2022-06-01 01:19:00'],
                                                       ['2022-06-01 07:57:00', '2022-06-01 09:16:00'],
                                                       ['2022-06-01 02:00:00', '2022-06-01 03:19:00'],
                                                       ['2022-06-03 18:06:00', '2022-06-03 19:25:00'],
                                                       ['2022-06-01 03:19:00', '2022-06-01 04:38:00'],
                                                       ['2022-06-03 20:44:00', '2022-06-03 22:03:00'],
                                                       ['2022-06-01 04:38:00', '2022-06-01 05:57:00'],
                                                       ['2022-06-03 22:03:00', '2022-06-03 23:22:00'],
                                                       ['2022-06-01 06:38:00', '2022-06-01 07:57:00'],
                                                       ['2022-06-04 00:02:00', '2022-06-04 01:21:00'],
                                                       ['2022-06-01 09:16:00', '2022-06-01 10:35:00'],
                                                       ['2022-06-04 01:39:00', '2022-06-04 02:58:00'],
                                                       ['2022-06-01 10:35:00', '2022-06-01 11:54:00'],
                                                       ['2022-06-04 08:15:00', '2022-06-04 09:34:00'],
                                                       ['2022-06-01 11:54:00', '2022-06-01 13:13:00'],
                                                       ['2022-06-04 22:39:00', '2022-06-04 23:58:00'],
                                                       ['2022-06-01 13:13:00', '2022-06-01 14:32:00'],
                                                       ['2022-06-05 00:08:00', '2022-06-05 01:27:00'],
                                                       ['2022-06-03 19:25:00', '2022-06-03 20:44:00'],
                                                       ['2022-06-05 04:14:00', '2022-06-05 05:33:00']], 'M18': [],
                                               'M19': [],
                                               'M20': [], 'M21': [], 'M22': [], 'M23': [], 'M24': [], 'M25': [],
                                               'M26': [['2022-06-01 00:00:00', '2022-06-01 01:09:00'],
                                                       ['2022-06-01 02:17:00', '2022-06-01 03:26:00'],
                                                       ['2022-06-01 01:09:00', '2022-06-01 02:18:00'],
                                                       ['2022-06-01 03:26:00', '2022-06-01 04:35:00']], 'M27': [],
                                               'M28': [],
                                               'M29': [], 'M30': [], 'M31': [], 'M32': [], 'M33': [], 'M34': [],
                                               'M35': [],
                                               'M36': [],
                                               'M37': [], 'M38': [], 'M39': [], 'M40': [], 'M41': [], 'M42': [],
                                               'M43': [],
                                               'M44': [],
                                               'M45': [], 'M46': [['2022-06-01 00:00:00', '2022-06-01 00:31:00'],
                                                                  ['2022-06-01 00:31:00', '2022-06-01 01:02:00'],
                                                                  ['2022-06-01 01:02:00', '2022-06-01 01:33:00'],
                                                                  ['2022-06-01 01:33:00', '2022-06-01 02:04:00'],
                                                                  ['2022-06-01 02:04:00', '2022-06-01 02:35:00'],
                                                                  ['2022-06-01 02:35:00', '2022-06-01 03:06:00'],
                                                                  ['2022-06-01 03:06:00', '2022-06-01 03:37:00'],
                                                                  ['2022-06-01 03:37:00', '2022-06-01 04:08:00']],
                                               'M47': []},
                'lowQualityDecision': {'P0052': True, 'P0082': False},
                'lowQualityProcess': {'P0052': 'null',
                                      'P0082': ['P0082p001', 'P0082p002', 'P0082p003', 'P0082p004', 'P0082p005']},
                'processProductBelong': {'P0001': ['P0001p001'], 'P0002': ['P0002p001'], 'P0003': ['P0003p001'],
                                         'P0004': ['P0004p001'], 'P0005': ['P0005p001'], 'P0006': ['P0006p001'],
                                         'P0007': ['P0007p001'], 'P0008': ['P0008p001'], 'P0009': ['P0009p001'],
                                         'P0010': ['P0010p001'],
                                         'P0011': ['P0011p001', 'P0011p002', 'P0011p003', 'P0011p004', 'P0011p005',
                                                   'P0011p006',
                                                   'P0011p007', 'P0011p008'],
                                         'P0012': ['P0012p001', 'P0012p002', 'P0012p003', 'P0012p004', 'P0012p005',
                                                   'P0012p006',
                                                   'P0012p007', 'P0012p008'],
                                         'P0013': ['P0013p001', 'P0013p002', 'P0013p003', 'P0013p004', 'P0013p005',
                                                   'P0013p006',
                                                   'P0013p007', 'P0013p008'],
                                         'P0014': ['P0014p001', 'P0014p002', 'P0014p003', 'P0014p004', 'P0014p005',
                                                   'P0014p006',
                                                   'P0014p007', 'P0014p008'],
                                         'P0015': ['P0015p001', 'P0015p002', 'P0015p003', 'P0015p004', 'P0015p005',
                                                   'P0015p006',
                                                   'P0015p007', 'P0015p008'],
                                         'P0016': ['P0016p001', 'P0016p002', 'P0016p003', 'P0016p004', 'P0016p005',
                                                   'P0016p006',
                                                   'P0016p007', 'P0016p008'],
                                         'P0017': ['P0017p001', 'P0017p002', 'P0017p003', 'P0017p004', 'P0017p005',
                                                   'P0017p006',
                                                   'P0017p007', 'P0017p008'],
                                         'P0018': ['P0018p001', 'P0018p002', 'P0018p003', 'P0018p004', 'P0018p005',
                                                   'P0018p006',
                                                   'P0018p007', 'P0018p008'],
                                         'P0019': ['P0019p001', 'P0019p002', 'P0019p003', 'P0019p004', 'P0019p005',
                                                   'P0019p006',
                                                   'P0019p007', 'P0019p008'],
                                         'P0020': ['P0020p001', 'P0020p002', 'P0020p003', 'P0020p004', 'P0020p005',
                                                   'P0020p006',
                                                   'P0020p007', 'P0020p008'],
                                         'P0021': ['P0021p001', 'P0021p002', 'P0021p003', 'P0021p004', 'P0021p005',
                                                   'P0021p006'],
                                         'P0022': ['P0022p001', 'P0022p002', 'P0022p003', 'P0022p004', 'P0022p005',
                                                   'P0022p006'],
                                         'P0023': ['P0023p001', 'P0023p002', 'P0023p003', 'P0023p004', 'P0023p005',
                                                   'P0023p006'],
                                         'P0024': ['P0024p001', 'P0024p002', 'P0024p003', 'P0024p004', 'P0024p005',
                                                   'P0024p006'],
                                         'P0025': ['P0025p001', 'P0025p002', 'P0025p003', 'P0025p004', 'P0025p005',
                                                   'P0025p006'],
                                         'P0026': ['P0026p001', 'P0026p002', 'P0026p003', 'P0026p004', 'P0026p005',
                                                   'P0026p006'],
                                         'P0027': ['P0027p001', 'P0027p002', 'P0027p003', 'P0027p004', 'P0027p005',
                                                   'P0027p006'],
                                         'P0028': ['P0028p001', 'P0028p002', 'P0028p003', 'P0028p004', 'P0028p005',
                                                   'P0028p006'],
                                         'P0029': ['P0029p001', 'P0029p002', 'P0029p003', 'P0029p004', 'P0029p005',
                                                   'P0029p006'],
                                         'P0030': ['P0030p001', 'P0030p002', 'P0030p003', 'P0030p004', 'P0030p005',
                                                   'P0030p006'], 'P0031': ['P0031p001'], 'P0032': ['P0032p001'],
                                         'P0033': ['P0033p001'], 'P0034': ['P0034p001'], 'P0035': ['P0035p001'],
                                         'P0036': ['P0036p001'], 'P0037': ['P0037p001'], 'P0038': ['P0038p001'],
                                         'P0039': ['P0039p001'], 'P0040': ['P0040p001'],
                                         'P0041': ['P0041p001', 'P0041p002', 'P0041p003', 'P0041p004', 'P0041p005',
                                                   'P0041p006',
                                                   'P0041p007'],
                                         'P0042': ['P0042p001', 'P0042p002', 'P0042p003', 'P0042p004', 'P0042p005',
                                                   'P0042p006',
                                                   'P0042p007'],
                                         'P0043': ['P0043p001', 'P0043p002', 'P0043p003', 'P0043p004', 'P0043p005',
                                                   'P0043p006',
                                                   'P0043p007'],
                                         'P0044': ['P0044p001', 'P0044p002', 'P0044p003', 'P0044p004', 'P0044p005',
                                                   'P0044p006',
                                                   'P0044p007'],
                                         'P0045': ['P0045p001', 'P0045p002', 'P0045p003', 'P0045p004', 'P0045p005',
                                                   'P0045p006',
                                                   'P0045p007'],
                                         'P0046': ['P0046p001', 'P0046p002', 'P0046p003', 'P0046p004', 'P0046p005',
                                                   'P0046p006',
                                                   'P0046p007'],
                                         'P0047': ['P0047p001', 'P0047p002', 'P0047p003', 'P0047p004', 'P0047p005',
                                                   'P0047p006',
                                                   'P0047p007'],
                                         'P0048': ['P0048p001', 'P0048p002', 'P0048p003', 'P0048p004', 'P0048p005',
                                                   'P0048p006',
                                                   'P0048p007'],
                                         'P0049': ['P0049p001', 'P0049p002', 'P0049p003', 'P0049p004', 'P0049p005',
                                                   'P0049p006',
                                                   'P0049p007'],
                                         'P0050': ['P0050p001', 'P0050p002', 'P0050p003', 'P0050p004', 'P0050p005',
                                                   'P0050p006',
                                                   'P0050p007'],
                                         'P0051': ['P0051p001', 'P0051p002', 'P0051p003', 'P0051p004', 'P0051p005',
                                                   'P0051p006',
                                                   'P0051p007'],
                                         'P0052': ['P0052p001', 'P0052p002', 'P0052p003', 'P0052p004', 'P0052p005',
                                                   'P0052p006',
                                                   'P0052p007'],
                                         'P0053': ['P0053p001', 'P0053p002', 'P0053p003', 'P0053p004', 'P0053p005',
                                                   'P0053p006',
                                                   'P0053p007'],
                                         'P0054': ['P0054p001', 'P0054p002', 'P0054p003', 'P0054p004', 'P0054p005',
                                                   'P0054p006',
                                                   'P0054p007'],
                                         'P0055': ['P0055p001', 'P0055p002', 'P0055p003', 'P0055p004', 'P0055p005',
                                                   'P0055p006',
                                                   'P0055p007'],
                                         'P0056': ['P0056p001', 'P0056p002', 'P0056p003', 'P0056p004', 'P0056p005',
                                                   'P0056p006',
                                                   'P0056p007'],
                                         'P0057': ['P0057p001', 'P0057p002', 'P0057p003', 'P0057p004', 'P0057p005',
                                                   'P0057p006',
                                                   'P0057p007'],
                                         'P0058': ['P0058p001', 'P0058p002', 'P0058p003', 'P0058p004', 'P0058p005',
                                                   'P0058p006',
                                                   'P0058p007'],
                                         'P0059': ['P0059p001', 'P0059p002', 'P0059p003', 'P0059p004', 'P0059p005',
                                                   'P0059p006',
                                                   'P0059p007'],
                                         'P0060': ['P0060p001', 'P0060p002', 'P0060p003', 'P0060p004', 'P0060p005',
                                                   'P0060p006',
                                                   'P0060p007'],
                                         'P0061': ['P0061p001', 'P0061p002', 'P0061p003', 'P0061p004', 'P0061p005',
                                                   'P0061p006',
                                                   'P0061p007'],
                                         'P0062': ['P0062p001', 'P0062p002', 'P0062p003', 'P0062p004', 'P0062p005',
                                                   'P0062p006',
                                                   'P0062p007'],
                                         'P0063': ['P0063p001', 'P0063p002', 'P0063p003', 'P0063p004', 'P0063p005',
                                                   'P0063p006',
                                                   'P0063p007'],
                                         'P0064': ['P0064p001', 'P0064p002', 'P0064p003', 'P0064p004', 'P0064p005',
                                                   'P0064p006',
                                                   'P0064p007'],
                                         'P0065': ['P0065p001', 'P0065p002', 'P0065p003', 'P0065p004', 'P0065p005',
                                                   'P0065p006',
                                                   'P0065p007'],
                                         'P0066': ['P0066p001', 'P0066p002', 'P0066p003', 'P0066p004', 'P0066p005',
                                                   'P0066p006',
                                                   'P0066p007'],
                                         'P0067': ['P0067p001', 'P0067p002', 'P0067p003', 'P0067p004', 'P0067p005',
                                                   'P0067p006',
                                                   'P0067p007'],
                                         'P0068': ['P0068p001', 'P0068p002', 'P0068p003', 'P0068p004', 'P0068p005',
                                                   'P0068p006',
                                                   'P0068p007'],
                                         'P0069': ['P0069p001', 'P0069p002', 'P0069p003', 'P0069p004', 'P0069p005',
                                                   'P0069p006',
                                                   'P0069p007'],
                                         'P0070': ['P0070p001', 'P0070p002', 'P0070p003', 'P0070p004', 'P0070p005',
                                                   'P0070p006',
                                                   'P0070p007'],
                                         'P0071': ['P0071p001', 'P0071p002', 'P0071p003', 'P0071p004', 'P0071p005',
                                                   'P0071p006',
                                                   'P0071p007'],
                                         'P0072': ['P0072p001', 'P0072p002', 'P0072p003', 'P0072p004', 'P0072p005',
                                                   'P0072p006',
                                                   'P0072p007'],
                                         'P0073': ['P0073p001', 'P0073p002', 'P0073p003', 'P0073p004', 'P0073p005',
                                                   'P0073p006',
                                                   'P0073p007'],
                                         'P0074': ['P0074p001', 'P0074p002', 'P0074p003', 'P0074p004', 'P0074p005',
                                                   'P0074p006',
                                                   'P0074p007'],
                                         'P0075': ['P0075p001', 'P0075p002', 'P0075p003', 'P0075p004', 'P0075p005',
                                                   'P0075p006',
                                                   'P0075p007'],
                                         'P0076': ['P0076p001', 'P0076p002', 'P0076p003', 'P0076p004', 'P0076p005',
                                                   'P0076p006',
                                                   'P0076p007'],
                                         'P0077': ['P0077p001', 'P0077p002', 'P0077p003', 'P0077p004', 'P0077p005',
                                                   'P0077p006',
                                                   'P0077p007'],
                                         'P0078': ['P0078p001', 'P0078p002', 'P0078p003', 'P0078p004', 'P0078p005',
                                                   'P0078p006',
                                                   'P0078p007'],
                                         'P0079': ['P0079p001', 'P0079p002', 'P0079p003', 'P0079p004', 'P0079p005',
                                                   'P0079p006',
                                                   'P0079p007'],
                                         'P0080': ['P0080p001', 'P0080p002', 'P0080p003', 'P0080p004', 'P0080p005',
                                                   'P0080p006',
                                                   'P0080p007'],
                                         'P0081': ['P0081p001', 'P0081p002', 'P0081p003', 'P0081p004', 'P0081p005',
                                                   'P0081p006',
                                                   'P0081p007', 'P0081p008', 'P0081p009', 'P0081p010'],
                                         'P0082': ['P0082p001', 'P0082p002', 'P0082p003', 'P0082p004', 'P0082p005',
                                                   'P0082p006',
                                                   'P0082p007', 'P0082p008', 'P0082p009', 'P0082p010'],
                                         'P0083': ['P0083p001', 'P0083p002', 'P0083p003', 'P0083p004', 'P0083p005',
                                                   'P0083p006',
                                                   'P0083p007', 'P0083p008', 'P0083p009', 'P0083p010'],
                                         'P0084': ['P0084p001', 'P0084p002', 'P0084p003', 'P0084p004', 'P0084p005',
                                                   'P0084p006',
                                                   'P0084p007', 'P0084p008', 'P0084p009', 'P0084p010'],
                                         'P0085': ['P0085p001', 'P0085p002', 'P0085p003', 'P0085p004', 'P0085p005',
                                                   'P0085p006',
                                                   'P0085p007', 'P0085p008', 'P0085p009', 'P0085p010'],
                                         'P0086': ['P0086p001', 'P0086p002', 'P0086p003', 'P0086p004', 'P0086p005',
                                                   'P0086p006',
                                                   'P0086p007', 'P0086p008', 'P0086p009', 'P0086p010'],
                                         'P0087': ['P0087p001', 'P0087p002', 'P0087p003', 'P0087p004', 'P0087p005',
                                                   'P0087p006',
                                                   'P0087p007', 'P0087p008', 'P0087p009', 'P0087p010'],
                                         'P0088': ['P0088p001', 'P0088p002', 'P0088p003', 'P0088p004', 'P0088p005',
                                                   'P0088p006',
                                                   'P0088p007', 'P0088p008', 'P0088p009', 'P0088p010'],
                                         'P0089': ['P0089p001', 'P0089p002', 'P0089p003', 'P0089p004', 'P0089p005',
                                                   'P0089p006',
                                                   'P0089p007', 'P0089p008', 'P0089p009', 'P0089p010'],
                                         'P0090': ['P0090p001', 'P0090p002', 'P0090p003', 'P0090p004', 'P0090p005',
                                                   'P0090p006',
                                                   'P0090p007', 'P0090p008', 'P0090p009', 'P0090p010'],
                                         'P0091': ['P0091p001', 'P0091p002', 'P0091p003', 'P0091p004', 'P0091p005',
                                                   'P0091p006',
                                                   'P0091p007', 'P0091p008', 'P0091p009', 'P0091p010'],
                                         'P0092': ['P0092p001', 'P0092p002', 'P0092p003', 'P0092p004', 'P0092p005',
                                                   'P0092p006',
                                                   'P0092p007', 'P0092p008', 'P0092p009', 'P0092p010'],
                                         'P0093': ['P0093p001', 'P0093p002', 'P0093p003', 'P0093p004', 'P0093p005',
                                                   'P0093p006',
                                                   'P0093p007', 'P0093p008', 'P0093p009', 'P0093p010'],
                                         'P0094': ['P0094p001', 'P0094p002', 'P0094p003', 'P0094p004', 'P0094p005',
                                                   'P0094p006',
                                                   'P0094p007', 'P0094p008', 'P0094p009', 'P0094p010'],
                                         'P0095': ['P0095p001', 'P0095p002', 'P0095p003', 'P0095p004', 'P0095p005',
                                                   'P0095p006',
                                                   'P0095p007', 'P0095p008', 'P0095p009', 'P0095p010'],
                                         'P0096': ['P0096p001', 'P0096p002', 'P0096p003', 'P0096p004', 'P0096p005',
                                                   'P0096p006',
                                                   'P0096p007', 'P0096p008', 'P0096p009', 'P0096p010'],
                                         'P0097': ['P0097p001', 'P0097p002', 'P0097p003', 'P0097p004', 'P0097p005',
                                                   'P0097p006',
                                                   'P0097p007', 'P0097p008', 'P0097p009', 'P0097p010'],
                                         'P0098': ['P0098p001', 'P0098p002', 'P0098p003', 'P0098p004', 'P0098p005',
                                                   'P0098p006',
                                                   'P0098p007', 'P0098p008', 'P0098p009', 'P0098p010'],
                                         'P0099': ['P0099p001', 'P0099p002', 'P0099p003', 'P0099p004', 'P0099p005',
                                                   'P0099p006',
                                                   'P0099p007', 'P0099p008', 'P0099p009', 'P0099p010'],
                                         'P0100': ['P0100p001', 'P0100p002', 'P0100p003', 'P0100p004', 'P0100p005',
                                                   'P0100p006',
                                                   'P0100p007', 'P0100p008', 'P0100p009', 'P0100p010']},
                'productDateEnd': {'P0001': '2022-06-08 00:00:00',
                                   'P0002': '2022-06-08 00:00:00',
                                   'P0003': '2022-06-08 00:00:00',
                                   'P0004': '2022-06-08 00:00:00',
                                   'P0005': '2022-06-08 00:00:00',
                                   'P0006': '2022-06-08 00:00:00',
                                   'P0007': '2022-06-08 00:00:00',
                                   'P0008': '2022-06-08 00:00:00',
                                   'P0009': '2022-06-08 00:00:00',
                                   'P0010': '2022-06-08 00:00:00',
                                   'P0011': '2022-06-08 00:00:00',
                                   'P0012': '2022-06-08 00:00:00',
                                   'P0013': '2022-06-08 00:00:00',
                                   'P0014': '2022-06-08 00:00:00',
                                   'P0015': '2022-06-08 00:00:00',
                                   'P0016': '2022-06-08 00:00:00',
                                   'P0017': '2022-06-08 00:00:00',
                                   'P0018': '2022-06-08 00:00:00',
                                   'P0019': '2022-06-08 00:00:00',
                                   'P0020': '2022-06-08 00:00:00',
                                   'P0021': '2022-06-08 00:00:00',
                                   'P0022': '2022-06-08 00:00:00',
                                   'P0023': '2022-06-08 00:00:00',
                                   'P0024': '2022-06-08 00:00:00',
                                   'P0025': '2022-06-08 00:00:00',
                                   'P0026': '2022-06-08 00:00:00',
                                   'P0027': '2022-06-08 00:00:00',
                                   'P0028': '2022-06-08 00:00:00',
                                   'P0029': '2022-06-08 00:00:00',
                                   'P0030': '2022-06-08 00:00:00',
                                   'P0031': '2022-06-08 00:00:00',
                                   'P0032': '2022-06-08 00:00:00',
                                   'P0033': '2022-06-08 00:00:00',
                                   'P0034': '2022-06-08 00:00:00',
                                   'P0035': '2022-06-08 00:00:00',
                                   'P0036': '2022-06-08 00:00:00',
                                   'P0037': '2022-06-08 00:00:00',
                                   'P0038': '2022-06-08 00:00:00',
                                   'P0039': '2022-06-08 00:00:00',
                                   'P0040': '2022-06-08 00:00:00',
                                   'P0041': '2022-06-08 00:00:00',
                                   'P0042': '2022-06-08 00:00:00',
                                   'P0043': '2022-06-08 00:00:00',
                                   'P0044': '2022-06-08 00:00:00',
                                   'P0045': '2022-06-08 00:00:00',
                                   'P0046': '2022-06-08 00:00:00',
                                   'P0047': '2022-06-08 00:00:00',
                                   'P0048': '2022-06-08 00:00:00',
                                   'P0049': '2022-06-08 00:00:00',
                                   'P0050': '2022-06-08 00:00:00',
                                   'P0051': '2022-06-08 00:00:00',
                                   'P0052': '2022-06-08 00:00:00',
                                   'P0053': '2022-06-08 00:00:00',
                                   'P0054': '2022-06-08 00:00:00',
                                   'P0055': '2022-06-08 00:00:00',
                                   'P0056': '2022-06-08 00:00:00',
                                   'P0057': '2022-06-08 00:00:00',
                                   'P0058': '2022-06-08 00:00:00',
                                   'P0059': '2022-06-08 00:00:00',
                                   'P0060': '2022-06-08 00:00:00',
                                   'P0061': '2022-06-08 00:00:00',
                                   'P0062': '2022-06-08 00:00:00',
                                   'P0063': '2022-06-08 00:00:00',
                                   'P0064': '2022-06-08 00:00:00',
                                   'P0065': '2022-06-08 00:00:00',
                                   'P0066': '2022-06-08 00:00:00',
                                   'P0067': '2022-06-08 00:00:00',
                                   'P0068': '2022-06-08 00:00:00',
                                   'P0069': '2022-06-08 00:00:00',
                                   'P0070': '2022-06-08 00:00:00',
                                   'P0071': '2022-06-08 00:00:00',
                                   'P0072': '2022-06-08 00:00:00',
                                   'P0073': '2022-06-08 00:00:00',
                                   'P0074': '2022-06-08 00:00:00',
                                   'P0075': '2022-06-08 00:00:00',
                                   'P0076': '2022-06-08 00:00:00',
                                   'P0077': '2022-06-08 00:00:00',
                                   'P0078': '2022-06-08 00:00:00',
                                   'P0079': '2022-06-08 00:00:00',
                                   'P0080': '2022-06-08 00:00:00',
                                   'P0081': '2022-06-08 00:00:00',
                                   'P0082': '2022-06-08 00:00:00',
                                   'P0083': '2022-06-08 00:00:00',
                                   'P0084': '2022-06-08 00:00:00',
                                   'P0085': '2022-06-08 00:00:00',
                                   'P0086': '2022-06-08 00:00:00',
                                   'P0087': '2022-06-08 00:00:00',
                                   'P0088': '2022-06-08 00:00:00',
                                   'P0089': '2022-06-08 00:00:00',
                                   'P0090': '2022-06-08 00:00:00',
                                   'P0091': '2022-06-08 00:00:00',
                                   'P0092': '2022-06-08 00:00:00',
                                   'P0093': '2022-06-08 00:00:00',
                                   'P0094': '2022-06-08 00:00:00',
                                   'P0095': '2022-06-08 00:00:00',
                                   'P0096': '2022-06-08 00:00:00',
                                   'P0097': '2022-06-08 00:00:00',
                                   'P0098': '2022-06-08 00:00:00',
                                   'P0099': '2022-06-08 00:00:00',
                                   'P0100': '2022-06-08 00:00:00'}
                }

    # 输入排产周期起始时刻、时长、每日休息时段
    def information(data):
        planstart = data['planStart']
        planstart = dt.datetime.strptime(planstart, "%Y-%m-%d %H:%M:%S")
        span = data['periodLength']
        planspan = dt.timedelta(days=span)
        planend = planstart + planspan
        reststart = '22:00'  # 暂定为22:00
        [a, b] = reststart.split(':')
        a = int(a)
        b = int(b)
        reststart = dt.timedelta(minutes=a * 60 + b)
        restend = '24:00'  # 暂定为24:00
        [a, b] = restend.split(':')
        a = int(a)
        b = int(b)
        restend = dt.timedelta(minutes=a * 60 + b)
        # 输出每日工作时间段
        restduration = []
        for i in range(0, span):
            a = planstart + dt.timedelta(days=i) + reststart
            b = planstart + dt.timedelta(days=i) + restend
            restduration.append([a, b])
        T = data['replanTime']
        T = dt.datetime.strptime(T, "%Y-%m-%d %H:%M:%S")
        for i in range(0, span):  # 若插单时刻发生在休息时段，将其移至该休息时段末尾
            if (restduration[i][0] <= T) & (restduration[i][1] >= T):
                T = restduration[i][1]
        return planstart, planend, planspan, restduration, T

    # 对重拍后引起的后一工序的开始时间超过前一工序结束时间的情况进行调整
    def renew(Q, QQ, J, restduration, T, planstart, planend, productDateEnd):  # Q为粗排的加工时间集合，QQ为粗排的加工工序集合
        QW = []  # 按设备分类的待加工工序集合
        # QK=[]#不区分设备的待加工工序集合
        QQW = []  # 按设备分类的待加工工序名称集合
        QQ2 = []  # QQW备份，检查环节使用
        Q1 = []  # 存储插单时刻前的工序时间
        QQ1 = []  # 存储插单时刻前的工序名称
        NW = []  # 按(具体)产品编号分类的待加工工序集合
        for key in Q.keys():
            qw = []
            qqw = []
            qq2 = []
            if Q[key] == []:  # 若某一设备未安排工序，则插单时刻后的加工工序、时间集合为空
                q1 = []
                qw = []
                qqw = []
                qq2 = []
                qq1 = []
                Q1.append(q1)
                QW.append(qw)
                QQW.append(qqw)
                QQ2.append(qq2)
                QQ1.append(qq1)
            else:
                q1 = []
                qq1 = []
                for j in range(0, len(Q[key])):
                    start = dt.datetime.strptime(Q[key][j][0], "%Y-%m-%d %H:%M:%S")
                    if start >= T:
                        qw.append(Q[key][j])
                        qqw.append(QQ[key][j])
                        qq2.append(QQ[key][j])
                    else:
                        q1.append(Q[key][j])
                        qq1.append(QQ[key][j])
                Q1.append(q1)
                QQ1.append(qq1)
                QQ2.append(qq2)
                QW.append(qw)
                QQW.append(qqw)
        '''
        print("插单时刻前生产计划部分：", Q1)
        print("插单时刻前生产计划部分（名称）", QQ1)
        '''
        for key in J.keys():
            nw = []
            for j in range(0, len(J[key])):
                for k in range(0, len(QQW)):
                    if J[key][j] in QQW[k]:
                        nw.append(J[key][j])
            if nw != []:
                NW.append(nw)
        '''
        print("插单时刻后的产品待加工工序名称集合（按产品分类）：", NW)
        print("插单时刻后的产品待加工工序名称集合（按设备分类）：", QQW)
        print("插单时刻后的产品待加工工序时间集合（按设备分类）：", QW)
        print("插单时刻后工序名称：", QQ2)
        '''

        # 初始化设备空闲时间（可以开始加工的最早时间）
        TW = []
        for i in range(0, len(QW)):
            if QW[i] == []:  # 若某一设备未安排工序，设置工序的最早加工时间这一排产周期的起点
                TW.append(planstart)
            else:
                TW.append(QW[i][0][0])
        # 初始化产品上一工序时间
        l = len(NW)
        TTW = [T.strftime("%Y-%m-%d %H:%M:%S") for _ in range(l)]  # 初始化时间为紧急插单的时刻
        # 正式更新排产计划
        z = len(QQW)
        finalQ = [[] for _ in range(z)]  # 用来存储最后各设备各工序的加工时间
        while (QQW != [[] for _ in range(z)]):
            for i in range(0, len(QQW)):
                j = 0
                while (j < len(NW)):  # 如果产品集合未遍历完
                    if QQW[i] != []:
                        if NW[j] != []:
                            if QQW[i][0] == NW[j][0]:
                                print("当前排产工序", QQW[i][0])
                                tcost = dt.datetime.strptime(QW[i][0][1],
                                                             "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(
                                    QW[i][0][0], "%Y-%m-%d %H:%M:%S")
                                a = dt.datetime.strptime(TW[i], "%Y-%m-%d %H:%M:%S")  # 设备最早可加工时间
                                b = dt.datetime.strptime(TTW[j], "%Y-%m-%d %H:%M:%S")  # 上一工序结束时间
                                if a <= b:
                                    temp1 = b
                                else:
                                    temp1 = a
                                # 判断当前安排的工序时间是否侵占了休息时间
                                x = temp1 + tcost
                                for p in range(0, len(restduration)):
                                    if restduration[p][0].day == temp1.day:
                                        if (temp1 > restduration[p][0]) & (
                                                temp1 < restduration[p][1]):  # 休息时段内才开始的，一律推迟到休息时段末
                                            temp1 = restduration[p][1]
                                            x = temp1 + tcost
                                        elif (temp1 <= restduration[p][0]) & (
                                                x > restduration[p][1]):  # 休息时段前开始，但耗完休息时段仍为加工完的，推迟至休息时段末
                                            temp1 = restduration[p][1]
                                            x = temp1 + tcost
                                        break
                                QW[i][0][0] = temp1.strftime("%Y-%m-%d %H:%M:%S")
                                QW[i][0][1] = x.strftime("%Y-%m-%d %H:%M:%S")
                                TW[i] = QW[i][0][1]  # 更新设备的最早可加工时间
                                TTW[j] = QW[i][0][1]  # 更新产品上一工序结束时间
                                finalQ[i].append(QW[i][0])  # 将安排好的工序的开始时间和结束时间放入最终顺序中
                                del QQW[i][0]
                                del QW[i][0]
                                del NW[j][0]
                                print(QQW)
                                print(QW)
                                print(NW)
                                print(finalQ)
                                j = 0  # 如果找到某一件产品的当前最前工序为当前设备的最前工序，那么当前设备的第二个工序成为最前工序，同样需要对所有产品种类进行遍历，重置产品序号为1
                            else:
                                j = j + 1  # 如果当前产品的最前工序与当前设备的最前工序不同，产品序号加1，判断下一产品的最前工序是否为当前设备的最前工序
                        else:
                            j = j + 1  # 如果当前产品的工序已经安排完，则去比对下一产品的最前工序
                    else:
                        break  # 如果某一设备的工序已被安排完，则安排下一个设备的工序
        print("插单时刻后工序名称：", QQ2)

        # 按产品检查，能否将某些工序移动至设备时间线的空闲处
        for i in range(0, len(finalQ)):
            if finalQ[i] != []:
                for j in range(1, len(finalQ[i])):  # 从设备安排的第二道工序开始，第一道工序已为最前，无法调整
                    for key_3 in J.keys():
                        if QQ2[i][j] in J[key_3]:
                            break
                    aaa = key_3
                    bbb = seperate(aaa, QQ2[i][j])
                    print("当前检查工序：", QQ2[i][j])
                    if J[aaa].index(QQ2[i][j]) != 0:
                        ab = J[aaa][J[aaa].index(QQ2[i][j]) - 1]
                        print("紧前工序名称", ab)
                        Qspan = dt.datetime.strptime(finalQ[i][j][1],
                                                     "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(
                            finalQ[i][j][0], "%Y-%m-%d %H:%M:%S")
                        print("工序耗时", Qspan)
                        lastend = '0'
                        find = '0'
                        for m in range(0, len(QQ2)):  #
                            for n in range(0, len(QQ2[m])):
                                print(m)
                                print(n)
                                if QQ2[m][n] == ab:
                                    lastend = dt.datetime.strptime(finalQ[m][n][1],
                                                                   "%Y-%m-%d %H:%M:%S")  # 记录紧前工序的结束加工时间，从这个时刻点开始向后检查空当
                                    print("紧前工序结束时间", lastend)
                                    find = '1'
                                    break
                            if (find == '1'):
                                break
                        if (lastend != '0'):  # 找到紧前工序
                            for jj in range(0, j):
                                c = dt.datetime.strptime(finalQ[i][jj][1], "%Y-%m-%d %H:%M:%S")
                                print("空当检查工序", QQ2[i][jj])
                                if c >= lastend:  # 只对该设备该工序前、紧前工序结束时刻后的工序进行检查
                                    cspan = dt.datetime.strptime(finalQ[i][jj + 1][0],
                                                                 "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(
                                        finalQ[i][jj][1], "%Y-%m-%d %H:%M:%S")
                                    print("空当起始工序", QQ2[i][jj])
                                    print("空当时间", cspan)
                                    if cspan >= Qspan:
                                        print("有空位，可前移", QQ2[i][j])
                                        finalQ[i][j][0] = finalQ[i][jj][1]
                                        cc = dt.datetime.strptime(finalQ[i][j][0], "%Y-%m-%d %H:%M:%S") + Qspan
                                        finalQ[i][j][1] = cc.strftime("%Y-%m-%d %H:%M:%S")
                                        TEMP1 = finalQ[i][j][0]
                                        TEMP2 = finalQ[i][j][1]
                                        TEMP = QQ2[i][j]
                                        for kk in range(j - 1, jj, -1):  # 将该工序移动至空当
                                            a = finalQ[i][kk][0]
                                            b = finalQ[i][kk][1]
                                            finalQ[i][kk + 1][0] = a
                                            finalQ[i][kk + 1][1] = b
                                            QQ2[i][kk + 1] = QQ2[i][kk]
                                        finalQ[i][jj + 1][0] = TEMP1
                                        finalQ[i][jj + 1][1] = TEMP2
                                        QQ2[i][jj + 1] = TEMP
                                        print("移动后的工序顺序", QQ2[i])
                                        print("移动后的工序时间集合", finalQ[i])
                                        break
        # 输出
        for i in range(len(Q1)):
            Q1[i].extend(finalQ[i])
        print("重排完成后的加工时间集合", Q1)
        for i in range(len(QQ1)):
            QQ1[i].extend(QQ2[i])
        print("重排完成后的工序名称集合", QQ1)

        # 反馈无法在交付日期前完成的产品件号信息
        unableDelieverOnTime = {}  # 存储不能按时交付产品件号、原定交货时间、当前安排计划下的完工时间
        for key in productDateEnd.keys():
            finalprocessID = J[key][-1]  # 最后一道工序
            for i in range(len(QQ1)):
                if finalprocessID in QQ1[i]:
                    finalprocessfinishtime = dt.datetime.strptime(Q1[i][QQ1[i].index(finalprocessID)][-1],
                                                                  "%Y-%m-%d %H:%M:%S")  # 当前安排计划下的完工时间
                    dateend = dt.datetime.strptime(productDateEnd[key], "%Y-%m-%d %H:%M:%S")  # 原定交货时间
                    if finalprocessfinishtime > dateend:
                        unableDelieverOnTime[key] = {}
                        unableDelieverOnTime[key]['dateEnd'] = productDateEnd[key]
                        unableDelieverOnTime[key]['planedFinishTime'] = Q1[i][QQ1[i].index(finalprocessID)][-1]
                    break
        '''
            # 输出超出排产周期的工序名称及下一阶段设备的最早可加工时间
        TNEXT = [dt.timedelta(minutes=0) for i in range(z)]  # 存储下一阶段设备的最早可加工时间
        QNEXT = [[] for _ in range(z)]  # 存储推迟到下一排产周期的工序
        for i in range(0, len(Q1)):
            print("当前设备", i)
            removeQ = []
            removeQQ = []  # 储存需要移除的工序时间及名称
            if Q1[i] != []:
                t = dt.timedelta(minutes=0)
                for j in range(0, len(Q1[i])):
                    a = dt.datetime.strptime(Q1[i][j][0], "%Y-%m-%d %H:%M:%S")
                    b = dt.datetime.strptime(Q1[i][j][1], "%Y-%m-%d %H:%M:%S")
                    if a >= planend:
                        QNEXT[i].append(QQ1[i][j])
                        removeQ.append(Q1[i][j])
                        removeQQ.append(QQ1[i][j])
                    if (a < planend) & (b > planend):
                        t = t + b - planend
                TNEXT[i] = t
                for j in range(0, len(removeQ)):
                    print(j)
                    Q1[i].remove(removeQ[j])
                    QQ1[i].remove(removeQQ[j])
        '''

        return Q1, QQ1, unableDelieverOnTime

    # 按时间顺序对Q和QQ重排
    def adjust(Q, QQ):
        for key in Q.keys():
            if len(Q[key]) <= 1:
                continue
            for k in range(0, len(Q[key]) - 1):
                for j in range(k + 1, len(Q[key])):
                    a = dt.datetime.strptime(Q[key][k][0], "%Y-%m-%d %H:%M:%S")
                    b = dt.datetime.strptime(Q[key][j][0], "%Y-%m-%d %H:%M:%S")
                    if a > b:
                        temp1 = Q[key][k]
                        Q[key][k] = Q[key][j]
                        Q[key][j] = temp1
                        temp2 = QQ[key][k]
                        QQ[key][k] = QQ[key][j]
                        QQ[key][j] = temp2
        return Q, QQ

    # 分离计划中specificprocessID包含的productID和processID
    def seperate(a, b):
        if b.startswith(a):
            return b.replace(a, '', 1)

    # 判断specificprocessID（b）包含的productID是否为productID（a）
    def containjudge(a, b):
        if b.startswith(a):
            return True

    def home():

        planstart, planend, planspan, restduration, replantime = information(data)
        QQ = data['pendingProcessMachine']
        Q = data['pendingProcessOriginalPlan']
        J = data['processProductBelong']
        productDateEnd = data['productDateEnd']
        Q, QQ = adjust(Q, QQ)
        '''
            # 定义绘图颜色
        colors = {}
        for i in range(0, len(QQ)):
            for j in range(0, len(QQ[i])):
                jobname = QQ[i][j][0:3]
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                rgb = 'rgb(' + str(r) + ',' + str(g) + ',' + str(b) + ')'
                colors[jobname] = rgb
        from Insert import GANTT
        GANTT(QQ, Q, colors)
        '''
        lowQualityDecision = data['lowQualityDecision']
        lowQualityProcess = data['lowQualityProcess']
        # lowqualityP = ['Q52', 'Q82']
        A = {}
        for key in lowQualityDecision.keys():
            A[key] = []
        QQ1 = {}  # before replantime
        Q1 = {}
        for key_1 in QQ.keys():
            QQ1[key_1] = []
            Q1[key_1] = []
            for j in range(len(QQ[key_1]) - 1, -1, -1):
                start = dt.datetime.strptime(Q[key_1][j][0], "%Y-%m-%d %H:%M:%S")
                end = dt.datetime.strptime(Q[key_1][j][-1], "%Y-%m-%d %H:%M:%S")
                d = end - start
                if start <= replantime:
                    QQ1[key_1].insert(0, QQ[key_1][j])
                    Q1[key_1].insert(0, Q[key_1][j])
                    for key_2 in lowQualityDecision.keys():
                        if QQ[key_1] != []:
                            if containjudge(key_2, QQ[key_1][j]):
                                A[key_2].insert(0, [QQ[key_1][j], key_1, d])  # 记录不合格产品的工序名、原设备编号、工序用时
                    QQ[key_1].remove(QQ[key_1][j])
                    Q[key_1].remove(Q[key_1][j])  # 仅保留replantime后的计划作为renew函数的输入
                else:
                    for key_2 in lowQualityDecision.keys():
                        if QQ[key_1] != []:
                            if containjudge(key_2, QQ[key_1][j]):
                                A[key_2].insert(0, [QQ[key_1][j], key_1, d])  # 记录不合格产品的工序名、原设备编号、工序用时
                                QQ[key_1].remove(QQ[key_1][j])
                                Q[key_1].remove(Q[key_1][j])  # 移除归属于不合格品的工序

        for key in lowQualityDecision.keys():
            scrap = lowQualityDecision[key]
            if scrap == True:
                for j in range(0, len(A[key])):
                    QQ[A[key][j][1]].append(A[key][j][0])
                    if Q[A[key][j][1]] == []:
                        start = planstart
                    else:
                        start = dt.datetime.strptime(Q[A[key][j][1]][-1][-1], "%Y-%m-%d %H:%M:%S")
                    end = start + A[key][j][-1]
                    start = start.strftime("%Y-%m-%d %H:%M:%S")
                    end = end.strftime("%Y-%m-%d %H:%M:%S")
                    Q[A[key][j][1]].append([start, end])
            else:
                redo = lowQualityProcess[key]
                J[key] = redo
                for j in range(0, len(redo)):
                    for k in range(0, len(A[key])):
                        if A[key][k][0] == redo[j]:
                            QQ[A[key][k][1]].append(A[key][k][0])
                            if Q[A[key][k][1]] == []:
                                start = planstart
                            else:
                                start = dt.datetime.strptime(Q[A[key][k][1]][-1][-1], "%Y-%m-%d %H:%M:%S")
                            end = start + A[key][k][-1]
                            start = start.strftime("%Y-%m-%d %H:%M:%S")
                            end = end.strftime("%Y-%m-%d %H:%M:%S")
                            Q[A[key][k][1]].append([start, end])
                            break
        finalQ, finalQQ, unableDelieverOnTime = renew(Q, QQ, J, restduration, replantime, planstart, planend,
                                                      productDateEnd)  # finalQ加工时间集合，finalQQ加工名称集合
        replanProcessName = {}
        replanProcessTime = {}
        j = -1
        for key in Q.keys():
            j = j + 1
            replanProcessTime[key] = finalQ[j]
            replanProcessName[key] = finalQQ[j]
            replanProcessTime[key].extend(Q1[key])
            replanProcessName[key].extend(QQ1[key])
        replanProcessTime, replanProcessName = adjust(replanProcessTime, replanProcessName)
        respond = {'replanProcessName': replanProcessName,
                   'replanProcessTime': replanProcessTime,
                   'replanProcessOthers': {},
                   'unableDelieverOnTime': unableDelieverOnTime}  # 暂时定为空集

        return json.dumps(respond)

    return home()


# 质量不合格结束
# ####################################################3
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
