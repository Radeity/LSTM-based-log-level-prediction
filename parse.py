import copy
import os.path
import sys
import pickle
import re
from collections import defaultdict
from block import Block

exclude_type = {"Block", "SimpleName", "SimpleType"}  # , "QualifiedName"}
SEEK_LOG_TYPE = "ExpressionStatement"
LOG_TYPE = "LogStatement"
BLOCK_TYPE = {"Block", "SwitchCase", "IfStatement"}
LOG_PREFIX = {".info(", ".trace(", ".debug(", ".warn(", ".error("}  # "log.", "logg", "logl", "log_",


class Log:
    def __init__(self, log_level, constant, callsite, line):
        self.log_level = log_level
        self.callsite = callsite
        self.constant = constant
        self.line = line


def parse_logs(project_name):
    logs = []
    for line in open("logs-{project_name}.txt".format(project_name=project_name)):
        match_item = re.match(
            r'<logcall>(.*)</logcall> <parameter>(.*)</parameter><constant>(.*)</constant><level>(.*)</level><callsite>(.*)</callsite><line>(.*)</line><superclass>(.*)</superclass>',
            line, re.M | re.I)
        log_list = []
        for i in range(7):
            log_list.append(match_item.group(i + 1))
        log = Log(match_item.group(4), match_item.group(3), match_item.group(5), (int)(match_item.group(6)))
        logs.append(log)

    logs_path = './Data/log/log-{project_name}.pkl'.format(project_name=project_name)
    with open(logs_path, 'wb') as pkl_file:
        pickle.dump(logs, pkl_file)
    return logs


def parse_AST(project_name, logs):
    log_idx = 0
    log_sum = len(logs)
    syntactic_feature = []
    end = 0
    log_match_flag = False
    blocks = []
    last_block_end = 0
    last_method = ""
    log_cnt = 0
    log_line = 0
    stack = []
    # make sure the same order in log and ast file
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

        # new block
        if single_ast[1] in BLOCK_TYPE or cur_begin_line > last_block_end:
            if log_match_flag:
                log_block = Block(last_method, logs[log_idx].log_level, logs[log_idx].constant)
                # exist nested block, push into stack until meeting the block end line
                stack.append((last_block_end, log_block, syntactic_feature))
                log_idx += log_cnt
                # like switch case, useless to log level prediction
                if log_cnt > 1:
                    print("Missing {cnt} logs in method `{method}`".format(cnt=log_cnt - 1, method=last_method))
                # all logs match
                if log_idx >= log_sum:
                    break
                log_match_flag = False
                log_line = 0
                log_cnt = 0

            # meet the block end line, pop
            while len(stack) > 0 and (cur_begin_line >= stack[-1][0] or last_method != cur_method):
                top = stack.pop()
                top_log_block = top[1]
                top_log_block.syntactic_feature = copy.deepcopy(syntactic_feature)
                top_log_block.gen_combine_feature()
                blocks.append(top_log_block)

            # help judge the new block
            last_block_end = (int)(single_ast[4])

        # new method, use `cur_end_line > end` because polymorphism
        if last_method != cur_method or cur_end_line > end:
            last_method = cur_method
            end = cur_end_line
            syntactic_feature = []

        # not match any logs
        if single_ast[0] != logs[log_idx].callsite:
            continue

        # add syntactic feature and exclude useless feature in advance
        if cur_end_line > log_line and single_ast[1] not in exclude_type:
            syntactic_feature.append(single_ast[1])

        # match log statement
        if single_ast[1] == SEEK_LOG_TYPE and any(i in single_ast[2].lower() for i in LOG_PREFIX):#single_ast[2][0:4].lower() in LOG_PREFIX:
            log_line = cur_end_line
            syntactic_feature[-1] = LOG_TYPE
            log_match_flag = True
            log_cnt += 1

    # pop all log blocks
    while len(stack) > 0:
        top = stack.pop()
        top_log_block = top[1]
        top_log_block.syntactic_feature = copy.deepcopy(syntactic_feature)
        top_log_block.gen_combine_feature()
        blocks.append(top_log_block)

    ast_path = './Data/ast/ast-{project_name}.pkl'.format(project_name=project_name)
    with open(ast_path, 'wb') as pkl_file:
        pickle.dump(blocks, pkl_file)
    return blocks


def load_dumped_data(type, project_name):
    file_path = './Data/{type}/{type}-{project_name}.pkl'.format(type=type, project_name=project_name)
    pkl_file = open(file_path, 'rb')
    obj = pickle.load(pkl_file)
    return obj


def count_feature(ast):
    sync_count_dict = defaultdict(int)
    message_count_dict = defaultdict(int)
    for block in ast:
        for feature in block.syntactic_feature:
            sync_count_dict[feature] += 1
        for feature in block.log_message_feature:
            message_count_dict[feature] += 1

    sorted_syn_feature = sorted(sync_count_dict.items(), reverse=True, key=lambda x: x[1])
    sorted_msg_feature = sorted(message_count_dict.items(), reverse=True, key=lambda x: x[1])

    return sorted_syn_feature, sorted_msg_feature


if __name__ == '__main__':
    project_name = sys.argv[1]
    # parse logs-*.txt
    if os.path.exists("./Data/log/log-{project}.pkl".format(project=project_name)):
        logs = load_dumped_data("log", project_name)
    else:
        logs = parse_logs(project_name)

    # parse AST-*.txt
    # ast = parse_AST(project_name, logs)

    if os.path.exists("./Data/ast/ast-{project}.pkl".format(project=project_name)):
        ast = load_dumped_data("ast", project_name)
    else:
        print(False)

    idx = 0
    for block in ast:
        if len(block.combine_feature) > 100:
            print('{idx} : {len}'.format(idx=idx, len=len(block.combine_feature)))
        idx += 1
    sorted_syn_feature, sorted_msg_feature = count_feature(ast)
    print(sorted_syn_feature)
    print(sorted_msg_feature)


    print("ending the first step of preprocessing")