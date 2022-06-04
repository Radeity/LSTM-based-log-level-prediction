import copy
import os.path
import pickle
import re

from block import Block

exclude_type = set(["Block", "SimpleName", "SimpleType"])
LOG_TYPE = "ExpressionStatement"
BLOCK_TYPE = set(["Block", "SwitchCase", "IfStatement"])
LOG_PREFIX = set(["log.", "logg", "logl"])


class Log:
    def __init__(self, log_level, constant, callsite, line):
        self.log_level = log_level
        self.callsite = callsite
        self.constant = constant
        self.line = line


def parse_logs(project_name):
    logs = []
    s = set()
    for line in open("logs-{project_name}.txt".format(project_name=project_name)):
        match_item = re.match(
            r'<logcall>(.*)</logcall> <parameter>(.*)</parameter><constant>(.*)</constant><level>(.*)</level><callsite>(.*)</callsite><line>(.*)</line><superclass>(.*)</superclass>',
            line, re.M | re.I)
        log_list = []
        for i in range(7):
            log_list.append(match_item.group(i + 1))
        log = Log(match_item.group(4), match_item.group(3), match_item.group(5), (int)(match_item.group(6)))
        # if log.callsite in s:
        #     print("conflict!!!" + log.callsite)
        # else:
        #     s.add(log.callsite)
        logs.append(log)

    logs_path = './Data/log/log-{project_name}.pkl'.format(project_name=project_name)
    with open(logs_path, 'wb') as pkl_file:
        pickle.dump(logs, pkl_file)
    return logs


def parse_AST(project_name, logs):
    log_idx = -1
    log_sum = len(logs)
    syntactic_feature = []
    end = 0
    log_match_flag = True
    blocks = []
    log_block = Block("", "", "")
    last_block_end = 0
    last_method = ""
    log_cnt = 1
    for line in open("AST-{project_name}.txt".format(project_name=project_name)):
        match_item = re.match(
            r'<method>(.*)</method><type>(.*)</type><name>(.*)</name><begin>(.*)</begin><end>(.*)</end>',
            line, re.M | re.I)
        single_ast = []
        for i in range(5):
            single_ast.append(match_item.group(i + 1))

        cur_begin_line = (int)(single_ast[3])
        cur_end_line = (int)(single_ast[4])
        cur_method = single_ast[0]

        # new method (exist methods with same name)
        if single_ast[1] in BLOCK_TYPE or cur_begin_line > last_block_end:
            last_block_end = (int)(single_ast[4])
            if log_match_flag:
                log_block.syntactic_feature = copy.deepcopy(syntactic_feature)
                # if log_idx == 10:
                #     print(1)
                blocks.append(log_block)
                log_idx += log_cnt
                if log_cnt > 1:
                    print("Missing {cnt} logs in method `{method}`".format(cnt=log_cnt - 1, method=last_method))
                if log_idx >= log_sum:
                    break
                log_block = Block(single_ast[0], logs[log_idx].log_level, logs[log_idx].constant)
                log_match_flag = False
                log_cnt = 0

        if last_method != cur_method or cur_end_line > end:
            last_method = cur_method
            end = cur_end_line
            syntactic_feature = []

        # if single_ast[0] == "org.apache.kafka.clients.consumer.internals.OffsetsForLeaderEpochClient.handleResponse":
        #     if single_ast[1] == LOG_TYPE:
        #         print(1)
        if single_ast[0] != logs[log_idx].callsite:
            continue

        # if cur_begin_line > log_line or cur_end_line < log_line:
        #     continue

        if single_ast[1] not in exclude_type:
            syntactic_feature.append(single_ast[1])

        if single_ast[1] == LOG_TYPE and single_ast[2][0:4].lower() in LOG_PREFIX:
            log_match_flag = True
            log_cnt += 1

    del blocks[0]
    ast_path = './Data/ast/ast-{project_name}.pkl'.format(project_name=project_name)
    with open(ast_path, 'wb') as pkl_file:
        pickle.dump(blocks, pkl_file)
    return blocks


def load_dumped_data(type, project_name):
    file_path = './Data/{type}/{type}-{project_name}.pkl'.format(type=type, project_name=project_name)
    pkl_file = open(file_path, 'rb')
    obj = pickle.load(pkl_file)
    return obj


if __name__ == '__main__':
    # parse logs-*.txt
    if os.path.exists("./Data/log/log-kafka.pkl"):
        logs = load_dumped_data("log", "kafka")
    else:
        logs = parse_logs("kafka")

    # parse AST-*.txt
    ast = parse_AST("kafka", logs)

    if os.path.exists("./Data/ast/ast-kafka.pkl"):
        ast = load_dumped_data("ast", "kafka")
    else:
        print(False)

    print("ending the first step of preprocessing")