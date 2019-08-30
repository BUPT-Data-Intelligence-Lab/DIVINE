import os

tasks_dir_path = '../NELL-995/tasks/'

if __name__ == '__main__':
    relations = os.listdir(tasks_dir_path)

    print('total tasks:', len(relations))
    for item in relations:
        relation = item
        print('current task:', relation)
        os.system('python pretrain.py ' + relation)
    print 'results for all pre-training saved'
