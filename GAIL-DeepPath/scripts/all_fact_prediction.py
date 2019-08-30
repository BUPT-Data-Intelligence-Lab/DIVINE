import os
fact_results_path = '../NELL-995/results' + '/fact_results.txt'
tasks_dir_path = '../NELL-995/tasks/'

if __name__ == '__main__':
    # clear results file
    with open(fact_results_path, 'w') as f:
        pass
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
    print('total tasks:',len(relations))
    for item in relations:
        relation = item
        print('current task:',relation)
        os.system('python fact_prediction_eval.py ' + relation)
    print 'results for fact prediction saved'
