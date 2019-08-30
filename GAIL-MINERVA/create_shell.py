input_dir = './'

if __name__ == '__main__':
    # 20 tasks
    relations = ['agentbelongstoorganization',
                 'athletehomestadium',
                 'athleteplaysforteam',
                 'athleteplaysinleague',
                 'athleteplayssport',
                 'organizationheadquarteredincity',
                 'organizationhiredperson',
                 'personborninlocation',
                 'personleadsorganization',
                 'teamplaysinleague',
                 'teamplayssport',
                 'worksfor']


    shell_data = []
    with open(input_dir + 'test_configs' + '/nell-templet.sh', "r") as f:
        shell_data = f.readlines()

    for task_id, task in enumerate(relations):
        shell_list = []
        for i, line in enumerate(shell_data):
            if i == 2:
                # print line.strip().split('/')
                cur_line = line.strip().split('/')
                cur_line[-2] = task
                cur_str = '/'.join(cur_line)+'\n'
                # print cur_str
                shell_list.append(cur_str)
            elif i == 16 or i == 14 or i == 3:
                cur_str = line.replace('nell-995',task)
                # print cur_str
                shell_list.append(cur_str)
            else:
                shell_list.append(line)

        with open(input_dir + 'test_configs/' + task + '.sh', "w") as f:
            for line in shell_list:
                f.write(line)


