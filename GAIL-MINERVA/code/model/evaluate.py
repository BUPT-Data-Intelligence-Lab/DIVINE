import os
import sys
from code.model.nell_eval import nell_eval

model_answers_path = ""
correct_answers_path = ""
output_path = ""

if __name__ == '__main__':
    # clear results record
    with open(output_path, 'w') as f:
        pass
    relations = os.listdir(tasks_dir_path)
    nell_eval()