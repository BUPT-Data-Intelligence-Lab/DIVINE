import os

if __name__ == '__main__':
    # relations = ['agentbelongstoorganization',
    #              'athletehomestadium',
    #              'athleteplaysforteam',
    #              'athleteplaysinleague',
    #              'athleteplayssport',
    #              'organizationheadquarteredincity',
    #              'organizationhiredperson',
    #              'personborninlocation',
    #              'personleadsorganization',
    #              'teamplaysinleague',
    #              'teamplayssport',
    #              'worksfor']
    relations = ['agentbelongstoorganization','personborninlocation']
    print('total tasks:', len(relations))
    for i in range(10):
        for relation in relations:
            os.system('bash run.sh configs/' + relation + '.sh')


