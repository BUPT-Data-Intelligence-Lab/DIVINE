import os
from options import read_options


def read_task_options(dataset_str):
    if dataset_str == 'nell-995':
        relations = ['nell-995',
                    'agentbelongstoorganization',
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
    elif dataset_str == 'fb15k':
        relations = []
        pass
    else:
        relations = []
        print 'the input dataset_name is illegal'

    if len(relations) > 0:
        options = dict()
        for idx, rel in enumerate(relations):
            options[rel] = read_options(dataset_str, rel, idx)

        return options, relations


if __name__ == '__main__':
    # bash_file_dir = '../test_configs/'
    # with open(bash_file_dir + 'athletehomestadium.sh') as f:
    #     data = f.readlines()
    #     print data

    options = read_task_options('nell-995')
    print options['athletehomestadium']
