from easydict import EasyDict

FLAGS = EasyDict()

FLAGS.input_dir = None  # Directory where original shared task data is read from.
FLAGS.output_dir = None  # Directory where new data is written to.
FLAGS.model_dir = None
FLAGS.task = None  # Task to run for.
FLAGS.bert_vocab_file = None
FLAGS.bert_config_file = None
FLAGS.init_checkpoint = None
FLAGS.tapas_verbosity = None
FLAGS.use_tpu = False
FLAGS.tpu_name = None
FLAGS.tpu_zone = None
FLAGS.gcp_project = None
FLAGS.master = None
FLAGS.num_tpu_cores = 8
FLAGS.test_batch_size = 32
FLAGS.train_batch_size = None
FLAGS.gradient_accumulation_steps = 1
FLAGS.iterations_per_loop = 1000
FLAGS.test_mode = False
FLAGS.tf_random_seed = None
FLAGS.max_seq_length = 512
FLAGS.mode = None
FLAGS.loop_predict = True
FLAGS.compression_type = None
FLAGS.reset_position_index_per_cell = False
FLAGS.prune_columns = False
