import json

input_dir = 'data_preprocessed/'

if __name__ == '__main__':
    data_set = ['agentbelongstoorganization',
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

    for item in data_set:
        dump_dict = dict()
        with open(input_dir + item + '/relation2id.txt', "r") as f:
            rel_data = f.readlines()

            for line in rel_data:
                dump_dict[line.split()[0]] = int(line.split()[1])
        tail = len(dump_dict)
        dump_dict['PAD'] = tail
        dump_dict['DUMMY_START_RELATION'] = tail + 1
        dump_dict['NO_OP'] = tail + 2
        dump_dict['UNK'] = tail + 3
        # print len(dump_dict)

        with open(input_dir + item + '/vocab/relation_vocab.json', "w") as f:
            json.dump(dump_dict, f)

        dump_dict = dict()
        with open(input_dir + item + '/entity2id.txt', "r") as f:
            ent_data = f.readlines()

            for line in ent_data:
                dump_dict[line.split()[0]] = int(line.split()[1])
        tail = len(dump_dict)
        dump_dict['PAD'] = tail
        dump_dict['UNK'] = tail + 1
        # print len(dump_dict)

        with open(input_dir + item + '/vocab/entity_vocab.json', "w") as f:
            json.dump(dump_dict, f)


