import os
import absl
import shutil
from pathlib import Path
import ntpath
import glob
import re

def get_bool(boolean_string):
    if boolean_string.lower() not in ['false', 'true']:
        raise ValueError('{0} is not a valid boolean string'.format(boolean_string))
    return boolean_string.lower() == 'true'

def create_dir_if_required(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def copy_file(full_file_name, copy_to_dir):
    create_dir_if_required(copy_to_dir)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, copy_to_dir)    
    else:
        absl.logging.error("file {0} does not exist.".format(full_file_name))

def move_file(file_name, soure, destination):
    source_file_fullname=os.path.join(soure,file_name)
    destination_file_fullname=os.path.join(destination, file_name)
    if os.path.isfile(source_file_fullname):
        if os.path.isfile(destination_file_fullname):
            os.remove(destination_file_fullname)
        shutil.move(source_file_fullname, destination_file_fullname)
    else:
        absl.logging.error("file {0} does not exist.".format(source_file_fullname))

def get_immediate_subdirnames(root_dir):
    return [name for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))]

def get_immediate_subdirs(root_dir):
    return [os.path.join(root_dir, name) for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))]

def get_immediate_files(root_dir):
    return [os.path.join(root_dir, name) for name in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, name))]

def get_parent_dir(path):
    path_finder = Path(path)
    return path_finder.parent

def get_current_dir_name(path):
    return os.path.basename(path)

def get_file_name_from_path(filepathname, exclude_extension = False):
    head, tail = ntpath.split(filepathname)
    filename = tail or ntpath.basename(head)
    if exclude_extension:
        filename = get_file_name_without_extension(filename)
    return filename

def get_path_from_filepathname(filepathname):
    head, tail = ntpath.split(filepathname)
    return head

def is_file(file_path):
    return True if os.path.isfile(file_path) else False

def delete_file_if_exists(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)

def delete_dir_and_all_contents(dir):
    shutil.rmtree(dir)
def get_file_extension(file_name):
    return os.path.splitext(file_name)[1]

def get_file_name_without_extension(file_name):
    return os.path.splitext(file_name)[0]

def add_postfix_to_filepathname(filepathname, postfix):
    filename = get_file_name_from_path(filepathname)
    filename_without_extension = get_file_name_without_extension(filename)
    filename = filename_without_extension + postfix + get_file_extension(filename)
    path = get_path_from_filepathname(filepathname)
    return os.path.join(path, filename)

def calc_recall(tp, fn, rounding_digits = None):
    recall = tp / (tp + fn)
    if rounding_digits != None:
        recall = round(recall, rounding_digits)
    return recall

def calc_precision(tp, fp, rounding_digits = None):
    precision = tp / (tp + fp)
    if rounding_digits != None:
        precision = round(precision, rounding_digits)
    return precision

def calc_f1(precision, recall, rounding_digits = None):
    f1 = 2 * precision * recall / (precision + recall)
    if rounding_digits != None:
        f1 = round(f1, rounding_digits)
    return f1

def count_file_lines(filename, exclude_first_line = False):
    with open(filename) as f:
        count = sum(1 for line in f)
    if exclude_first_line:
        count -=1
    return count

def get_files_in_dir(dir, extensionPattern = None):
    if(extensionPattern==None):
        return [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    else:
        return glob.glob(os.path.join(dir, extensionPattern))

def count_files_lines_in_dir(dir, extensionPattern = None, exclude_first_line = False):
    count = 0
    filenames = get_files_in_dir(dir, extensionPattern)
    for filename in filenames:
        count += count_file_lines(filename, exclude_first_line)
    return count

def IfStringRepresentsInt(s):
    try: 
        int(s)
        return s.isdigit()
    except ValueError:
        return False

def IfStringRepresentsFloat(s):
    try:
        float(s)
        return str(float(s)) == s
    except ValueError:
        return False


if __name__ == '__main__':
    #test:
    result=count_files_lines_in_dir(r'C:\H\PhD\ORIBA\Model\FileGen\OREBA.dis\64_std_uni_no_smo', '*.csv', True)
    print(result)
