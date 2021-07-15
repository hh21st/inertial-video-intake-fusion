import tensorflow as tf
import absl
import os
import logging
import traceback
import main
import utils
import eval
import eval2

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S', level=logging.INFO)

FLAGS = absl.app.flags.FLAGS
absl.app.flags.DEFINE_enum(
    name='eval_module', default='eval', enum_values=['eval', 'eval2'], 
    help='eval module to use')
absl.app.flags.DEFINE_string(
    name='root_dir', default=r'C:\H\temp\fusion.small_cnn.cl.resnet_slowfast.s16.f8.k3.n.dis.sh', help='root directory to find all .sh files')
absl.app.flags.DEFINE_boolean(
    name='overwrite', default=False, help='overwrite existing prob files and calculate the metrics if true')
absl.app.flags.DEFINE_boolean(
    name='is_initialized', default=False, help='true if the app been initialized (for internal use)')
absl.app.flags.DEFINE_integer(
    name='original_batch_size', default=32, help='Batch size used for training. (for internal use)')
absl.app.flags.DEFINE_integer(
    name='original_seq_length', default=128, help='Number of sequence elements. (for internal use)')
absl.app.flags.DEFINE_float(
    name='original_base_learning_rate', default=3e-4, help='Base learning rate (for internal use)')
absl.app.flags.DEFINE_float(
    name='original_decay_rate', default=0.93, help='Decay rate of the learning rate. (for internal use)')
absl.app.flags.DEFINE_enum(
    name='original_hand', default='both', enum_values=['both', 'dom', 'nondom'],
    help='specified data from which hands are included in the input (for internal use)')
absl.app.flags.DEFINE_enum(
    name='original_modality', default='both', enum_values=['both', 'accel', 'gyro'],
    help='specified data from what modalities are included in the input (for internal use)')
absl.app.flags.DEFINE_boolean(
    name='original_use_sequence_loss', default=True,
    help='Use sequence-to-sequence loss')
absl.app.flags.DEFINE_integer(
    name='original_small_num_filter_each_layer', default=8,
    help='number of convelutional filters for small cnn in each layer')
absl.app.flags.DEFINE_integer(
    name='original_small_kernel_size', default=6,
    help='Size of convelutional filters for small cnn')
absl.app.flags.DEFINE_integer(
    name='original_save_summary_steps', default=500,
    help='save_summary_steps')
absl.app.flags.DEFINE_integer(
    name='original_save_checkpoints_steps', default=500,
    help='save_checkpoints_steps')
absl.app.flags.DEFINE_boolean(
    name='original_use_threshold', default=True,
    help='use video and inertial validation threshold from stage 2 for training')

def get_indexes_from_model_dir(model_dir):
    indexes = []
    for index_file in os.listdir(model_dir):
        if index_file.endswith('.index'):
            indexes.append(index_file[:-6].split('.')[1].split('-')[1].strip())
    return indexes

def move_index_files(index, source_dir, destination_dir):
    utils.move_file('model.ckpt-{0}.data-00000-of-00001'.format(index), source_dir, destination_dir)
    utils.move_file('model.ckpt-{0}.index'.format(index), source_dir, destination_dir)
    utils.move_file('model.ckpt-{0}.meta'.format(index), source_dir, destination_dir)
    
def create_checkpoint_file(checkpoint_file_fullname, index):
    if os.path.isfile(checkpoint_file_fullname):
        os.remove(checkpoint_file_fullname)
    checkpoint_file = open(checkpoint_file_fullname,"w")
    checkpoint_file.write('model_checkpoint_path: "model.ckpt-{0}"'.format(index))
    checkpoint_file.write('\n')
    checkpoint_file.write('all_model_checkpoint_paths: "model.ckpt-{0}"'.format(index))
    checkpoint_file.close()

def prepare_index_checkpoint(model_dir_bests):
    indexes = get_indexes_from_model_dir(model_dir_bests)
    for index in indexes:
        index_dir=os.path.join(model_dir_bests,index)
        if not os.path.exists(index_dir):
            os.mkdir(index_dir)
            logging.info("folder {} was created".format(index_dir))
        else:
            logging.info("folder {} has already been created".format(index_dir))
        move_index_files(index, model_dir_bests, index_dir)
    index_dirs = utils.get_immediate_subdirs(model_dir_bests)
    for index_dir in index_dirs:
        create_checkpoint_file(os.path.join(index_dir,'checkpoint'), utils.get_current_dir_name(index_dir))
    return indexes

def write_f1score_header(f1score_file_fullname):
    if os.path.isfile(f1score_file_fullname):
        return
    f1score_file_header = 'model,checkpoint index,F1,UAR,TP,FN,FP type 1,FP type 2,Precision,Recall,Best threshold,model path'
    f1score_file = open(f1score_file_fullname,'w')
    f1score_file.write(f1score_file_header)
    f1score_file.write('\n')
    f1score_file.close()
    logging.info("{} was created".format(f1score_file_fullname))

def write_f1score_line(f1score_file_fullname,model_desciption, index, f1, uar, tp, fn, fp_1, fp_2, precision, recall, best_threshold, model_dir):
    try:
        f1score_file = open(f1score_file_fullname,'a')
    except:
        file_nane, file_extension = os.path.splitext(f1score_file_fullname)
        f1score_file_fullname = file_nane+'1'+file_extension
        write_f1score_header(f1score_file_fullname)
        f1score_file = open(f1score_file_fullname)
    f1score_file_line = model_desciption
    f1score_file_line += ',' + str(index)
    f1score_file_line += ',' + str(f1)
    f1score_file_line += ',' + str(uar)
    f1score_file_line += ',' + str(tp)
    f1score_file_line += ',' + str(fn)
    f1score_file_line += ',' + str(fp_1)
    f1score_file_line += ',' + str(fp_2)
    f1score_file_line += ',' + str(precision)
    f1score_file_line += ',' + str(recall)
    f1score_file_line += ',' + str(best_threshold)
    f1score_file_line += ',' + model_dir
    f1score_file.write(f1score_file_line)
    f1score_file.write('\n')
    f1score_file.close()
    logging.info("values for model {} - index {} was added to the file".format(model_desciption, str(index)))

def calc_f1(batch_size, eval_dir, model_dir, model, sub_mode, fusion, f_strategy, f_mode, f1score_file_fullname, seq_length, base_learning_rate, decay_rate, hand, modality, use_sequence_loss, small_num_filter_each_layer, small_kernel_size, save_summary_steps, save_checkpoints_steps, use_threshold):
    FLAGS.batch_size=batch_size
    FLAGS.model=model 
    FLAGS.sub_mode=sub_mode
    #FLAGS.decay_rate=.93 
    FLAGS.fusion=fusion
    FLAGS.f_strategy=f_strategy
    FLAGS.f_mode=f_mode
    FLAGS.seq_length = seq_length
    FLAGS.base_learning_rate = base_learning_rate
    FLAGS.decay_rate = decay_rate
    FLAGS.hand = hand
    FLAGS.modality = modality
    FLAGS.use_sequence_loss = use_sequence_loss
    FLAGS.small_num_filter_each_layer = small_num_filter_each_layer
    FLAGS.small_kernel_size = small_kernel_size
    FLAGS.save_summary_steps = save_summary_steps
    FLAGS.save_checkpoints_steps = save_checkpoints_steps
    FLAGS.use_threshold = use_threshold
    #
    FLAGS.min_dist = seq_length
    #
    FLAGS.mode ='predict_and_export_csv'
    model_desciption = utils.get_current_dir_name(model_dir)
    model_dir_bests = os.path.join(model_dir,'best_checkpoints')
    prepare_index_checkpoint(model_dir_bests)
    #
    eval_dir_sub=eval_dir+'_sub'
    eval_dir_sub = eval_dir+'_sub' if FLAGS.eval_mode == 'estimate' else eval_dir.replace('eval','test',1)+'_sub' 
    logging.info("eval_dir_sub: {}".format(eval_dir_sub))
    eval_sub_dirs = utils.get_immediate_subdirs(eval_dir_sub)
    index_dirs = utils.get_immediate_subdirs(model_dir_bests)
    prob_suffix = 'prob' if FLAGS.eval_mode == 'estimate' else 'prob_test' 
    for index_dir in index_dirs:
        FLAGS.prob_dir=os.path.join(index_dir, prob_suffix)
        FLAGS.model_dir = index_dir
        try:
            if not os.path.exists(FLAGS.prob_dir):
                os.mkdir(FLAGS.prob_dir)
            for eval_sub_dir in eval_sub_dirs:
                if not FLAGS.overwrite and os.path.isfile(os.path.join(FLAGS.prob_dir,utils.get_current_dir_name(eval_sub_dir)+'.csv')):
                    continue
                FLAGS.eval_dir=eval_sub_dir
                main.run_experiment()
                logging.info("probabilities file for validation folder {} was added to folder {}".format(FLAGS.eval_dir, FLAGS.prob_dir))
        
            if FLAGS.eval_module == 'eval':
                uar, tp, fn, fp_1, fp_2, precision, recall, f1, best_threshold = eval.main()
            elif FLAGS.eval_module == 'eval2':
                uar, tp, fn, fp_1, fp_2, precision, recall, f1, best_threshold = eval2.main()
            else:
                raise ValueError('FLAGS.eval_module is not implemented', FLAGS.eval_module)
            if uar != -1:
                write_f1score_line(f1score_file_fullname,model_desciption,utils.get_current_dir_name(index_dir),f1,uar,tp,fn,fp_1,fp_2,precision,recall,best_threshold,model_dir)
        except Exception as e:
            logging.error(traceback.format_exc()) 
            write_f1score_line(f1score_file_fullname,model_desciption,utils.get_current_dir_name(index_dir),0,0,0,0,0,0,0,0,0,model_dir)

def read_sh_file(sh_file_fullname):
    def read_python_line(words):
        if not FLAGS.is_initialized:
            FLAGS.original_batch_size = FLAGS.batch_size
            FLAGS.original_seq_length = FLAGS.seq_length
            FLAGS.original_base_learning_rate = FLAGS.base_learning_rate
            FLAGS.original_decay_rate = FLAGS.decay_rate
            FLAGS.original_hand = FLAGS.hand
            FLAGS.original_modality = FLAGS.modality
            FLAGS.original_use_sequence_loss = FLAGS.use_sequence_loss
            FLAGS.original_small_num_filter_each_layer = FLAGS.small_num_filter_each_layer
            FLAGS.original_small_kernel_size = FLAGS.small_kernel_size
            FLAGS.original_save_summary_steps = FLAGS.save_summary_steps
            FLAGS.original_save_checkpoints_steps = FLAGS.save_checkpoints_steps
            FLAGS.original_use_threshold = FLAGS.use_threshold
            FLAGS.is_initialized = True
        
        eval_dir=''
        model_dir = ''
        model = ''
        sub_mode = ''
        fusion = ''
        f_strategy = ''
        f_mode = ''
        batch_size = FLAGS.original_batch_size
        seq_length = FLAGS.original_seq_length
        base_learning_rate = FLAGS.original_base_learning_rate
        decay_rate = FLAGS.original_decay_rate
        hand = FLAGS.original_hand
        modality = FLAGS.original_modality
        use_sequence_loss = FLAGS.original_use_sequence_loss
        small_num_filter_each_layer = FLAGS.original_small_num_filter_each_layer
        small_kernel_size = FLAGS.original_small_kernel_size
        save_summary_steps = FLAGS.original_save_summary_steps
        save_checkpoints_steps = FLAGS.original_save_checkpoints_steps
        use_threshold = FLAGS.original_use_threshold

        for word in words:
            if word.startswith('--model_dir'):
                model_dir = word.split('=')[1]
            elif word.startswith('--batch_size'):
                batch_size = int(word.split('=')[1])
            elif word.startswith('--model'):
                model = word.split('=')[1]
            elif word.startswith('--sub_mode'):
                sub_mode = word.split('=')[1].strip('\"')
            elif word.startswith('--fusion'):
                fusion = word.split('=')[1]
            elif word.startswith('--f_strategy'):
                f_strategy = word.split('=')[1]
            elif word.startswith('--f_mode'):
                f_mode = word.split('=')[1]
            elif word.startswith('--eval_dir'):
                eval_dir = word.split('=')[1]
            elif word.startswith('--seq_length'):
                seq_length = int(word.split('=')[1])
            elif word.startswith('--base_learning_rate'):
                base_learning_rate = float(word.split('=')[1])
            elif word.startswith('--decay_rate'):
                decay_rate = float(word.split('=')[1])
            elif word.startswith('--hand'):
                hand = word.split('=')[1]
            elif word.startswith('--modality'):
                modality = word.split('=')[1]
            elif word.startswith('--use_sequence_loss'):
                use_sequence_loss = utils.get_bool(word.split('=')[1])
            elif word.startswith('--small_num_filter_each_layer'):
                small_num_filter_each_layer = int(word.split('=')[1])
            elif word.startswith('--small_kernel_size'):
                small_kernel_size = int(word.split('=')[1])
            elif word.startswith('--save_summary_steps'):
                save_summary_steps = int(word.split('=')[1])
            elif word.startswith('--save_checkpoints_steps'):
                save_checkpoints_steps = int(word.split('=')[1])
            elif word.startswith('--use_threshold'):
                use_threshold = utils.get_bool(word.split('=')[1])
        if fusion == None or fusion.strip() == '' :
            fusion = 'none' 
        return batch_size, eval_dir, model_dir, model, sub_mode, fusion, f_strategy, f_mode, seq_length, base_learning_rate, decay_rate, hand, modality, use_sequence_loss, small_num_filter_each_layer, small_kernel_size, save_summary_steps, save_checkpoints_steps, use_threshold

    sh_file = open(sh_file_fullname,'r')
    content = sh_file.read()
    lines = content.splitlines()
    base_dir =''
    for line in lines:
        if line.startswith('cd'):
            base_dir = line[3:]
        elif line.startswith('python'):
            words=line.split()
            if words[0] == 'python' and words[1] == 'main.py':
                batch_size, eval_dir, model_dir, model, sub_mode, fusion, f_strategy, f_mode, seq_length, base_learning_rate, decay_rate, hand, modality, use_sequence_loss, small_num_filter_each_layer, small_kernel_size, save_summary_steps, save_checkpoints_steps, use_threshold = read_python_line(words)
                model_dir = os.path.join(base_dir,model_dir)
                
                logging.info("read_sh_file - batch_size={}".format(batch_size))
                logging.info("read_sh_file - eval_dir={}".format(eval_dir))
                logging.info("read_sh_file - model_dir={}".format(model_dir))
                logging.info("read_sh_file - model={}".format(model))
                logging.info("read_sh_file - sub_mode={}".format(sub_mode))
                logging.info("read_sh_file - fusion={}".format(fusion))
                logging.info("read_sh_file - f_strategy={}".format(f_strategy))
                logging.info("read_sh_file - f_mode={}".format(f_mode))
                logging.info("read_sh_file - seq_length={}".format(seq_length))
                logging.info("read_sh_file - base_learning_rate={}".format(base_learning_rate))
                logging.info("read_sh_file - decay_rate={}".format(decay_rate))
                logging.info("read_sh_file - hand={}".format(hand))
                logging.info("read_sh_file - modality={}".format(modality))
                logging.info("read_sh_file - use_sequence_loss={}".format(use_sequence_loss))
                logging.info("read_sh_file - small_num_filter_each_layer={}".format(small_num_filter_each_layer))
                logging.info("read_sh_file - small_kernel_size={}".format(small_kernel_size))
                logging.info("read_sh_file - save_summary_steps={}".format(save_summary_steps))
                logging.info("read_sh_file - save_checkpoints_steps={}".format(save_checkpoints_steps))
                logging.info("read_sh_file - use_threshold={}".format(use_threshold))

                return batch_size, eval_dir, model_dir, model, sub_mode, fusion, f_strategy, f_mode, seq_length, base_learning_rate, decay_rate, hand, modality, use_sequence_loss, small_num_filter_each_layer, small_kernel_size, save_summary_steps, save_checkpoints_steps, use_threshold
    return '___','___','___','___','___','___','___','___','___','___','___','___','___','___'


def calc_for_sh_file(sh_file_fullname, f1score_file_fullname):
    batch_size, eval_dir, model_dir, model, sub_mode, fusion, f_strategy, f_mode, seq_length, base_learning_rate, decay_rate, hand, modality, use_sequence_loss, small_num_filter_each_layer, small_kernel_size, save_summary_steps, save_checkpoints_steps, use_threshold = read_sh_file(sh_file_fullname)

    if os.path.exists(model_dir):
        calc_f1(batch_size, eval_dir, model_dir, model, sub_mode, fusion, f_strategy, f_mode, f1score_file_fullname, seq_length, base_learning_rate, decay_rate, hand, modality, use_sequence_loss, small_num_filter_each_layer, small_kernel_size, save_summary_steps, save_checkpoints_steps, use_threshold)
    else:
        logging.info("path {} does not exists for file {}".format(model_dir, sh_file_fullname))

def calc(root_dir, eval_dir):
    dirs_info = [x for x in os.walk(root_dir)]
    f1score_file_name = 'f1scores.csv' if FLAGS.eval_mode == 'estimate' else 'f1scores_test.csv' 
    f1score_file_fullname = os.path.join(root_dir,f1score_file_name)
    write_f1score_header(f1score_file_fullname)
    for dir_info in dirs_info:
        dir=dir_info[0]
        files = dir_info[2]
        for file in files:
            try:
                if utils.get_file_extension(file) =='.sh':
                    sh_file_fullname = os.path.join(dir,file)
                    calc_for_sh_file(sh_file_fullname, f1score_file_fullname)
            except Exception as e:
                logging.error(traceback.format_exc())

def mainf1score(args=None):
    FLAGS.col_label = 1
    FLAGS.col_prob = 2

    if utils.is_file(FLAGS.root_dir):
        assert utils.get_file_extension(FLAGS.root_dir) == '.sh', 'root_dir is a file therefore its extension has to be .sh, file name={0}'.format(FLAGS.root_dir)
        f1score_file_name_suffix = '_f1score.csv' if FLAGS.eval_mode == 'estimate' else '_f1score_test.csv' 
        f1score_file_fullname = utils.get_file_name_without_extension(FLAGS.root_dir)+ f1score_file_name_suffix
        calc_for_sh_file(FLAGS.root_dir, f1score_file_fullname)
    else:
        calc(FLAGS.root_dir,FLAGS.eval_dir)

if __name__ == '__main__':

    absl.app.run(main=mainf1score)
