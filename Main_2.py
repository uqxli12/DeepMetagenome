from Preprocessing_for_multiple_FASTA import *
from Preprocessing_for_Single_FASTA import *
from Deep_Model_ts_2 import *

#here define functions for simply using the modules given above. Users only need to type functions names and the input is only the training dataset and prediction dataset. File addresses and name can be changed in these functions for your specific scenario.
#all output files will be found in the work directory, normally the same place of the modules
#to excute the function, simply type the function name with the specified parameters. Also plese excute these functions in order, as the files are passed in order

def preprocessing_for_single_FASTA(file):
    print('Welcome to DeepMetaProteome!\n')
    _a = Fasta_Stat_for_singleFASTA(file)
    _a.seq_parse_single()
    _a.fasta_to_DataFrame_single()
    print('Prediction dataset preprocessing was done!\n')

def preprocessing_for_multi_FASTA(path):
    print('Welcome to DeepMetaProteome!\n')
    _b = Metagenome_Merger(path)
    _b.file_list()
    _b.file_list_creator()
    _b.metagenome_merger()
    _c = Fasta_Stat(path)
    _c.seq_parse()
    _c.fasta_to_DataFrame()
    print('Prediction dataset preprocessing was done!\n')

def training_dataset_preprocess(file):
    print('Welcome to DeepMetaProteome!\n')
    data1 = Seq_Preprocess(file)
    data1.file_reader()
    data1.plot_classes()
    print('Training dataset processing was done!\n')

def my_model():
    print('Welcome to DeepMetaProteome!\n')
    start_time = datetime.datetime.now() 
    data2 = MyModel()
    data2.BinaryY("df_duplicateout.csv")
    data2.model_builder()
    data2.call()
    data2.plot_prediction()
    data2.plt_ROC_curve()
    data2.plt_AUROC_AUPRC_curve()
        # 获取当前进程的内存使用信息
    process = psutil.Process()
    info = process.memory_info()
        # 打印内存使用详情
    print(f"进程内存使用: 共计 {info.rss / 1024/1024 ** 2:.2f} MB")
        
    end_time = datetime.datetime.now()
    log_time = "训练总时间: " + str((end_time - start_time).seconds / 60) + "m"
    time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
    print(log_time)
    
#     print(f"进程内存使用: 共计 {info.rss / 1024 ** 2:.2f} MB")
    print('Model building was done!\n')

def model_pred():
    print('Welcome to DeepMetaProteome!\n')
    data3 = ModelPred('df_duplicateout.csv',"predictionset.csv")
    data3.prot_pred()
    data3.result_processing()
    print('Prediction was done!\n')
