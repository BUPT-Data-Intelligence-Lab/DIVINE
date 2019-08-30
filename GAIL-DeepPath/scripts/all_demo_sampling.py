import os

tasks_dir_path = '../NELL-995/tasks/'

if __name__ == '__main__':
    relations = os.listdir(tasks_dir_path)
    # relations = ['concept_agentbelongstoorganization',
    #              'concept_athletehomestadium',
    #              'concept_athleteplaysforteam',
    #              'concept_athleteplaysinleague',
    #              'concept_athleteplayssport',
    #              'concept_organizationheadquarteredincity',
    #              'concept_organizationhiredperson',
    #              'concept_personborninlocation',
    #              'concept_personleadsorganization',
    #              'concept_teamplaysinleague',
    #              'concept_teamplayssport',
    #              'concept_worksfor']
    print('total tasks:', len(relations))
    for item in relations:
        relation = item
        print('current task:', relation)
        os.system('python demo_sampling.py ' + relation)
    print 'results for all demo sampling saved'
