#coding: utf-8
import tensorflow as tf
import pandas as pd
from Bio import SeqIO
import pandas as pd
import os

"""
This modual depends on biopython and tensorflow. 
conda install biopython
Define MaxLen or MaxLen first
"""
#define MaxLen or MinLen first
MaxLen = 200
class Fasta_Reader():
    def __init__(self,file):
        self.file = file  
        #here file is the FASTA name as a string
    def fasta_reader(self):
        fasta_df = pd.read_csv(self.file, sep='>', lineterminator='>', header=None)
        fasta_df[['Accession', 'Sequence']] = fasta_df[0].str.split('\n', 1, expand=True)
        fasta_df.drop(0, axis=1, inplace=True)
        fasta_df['Sequence'] = fasta_df['Sequence'].replace('\n', '', regex=True)
        return fasta_df

class Metagenome_Merger():
    """
    The Metagenome_Merger provides functions to merge a number of protein datasets in FASTA format.
    All metagenomes file should be placed in a separate folder which should not be current work directory
    """

     
    def __init__(self, path):
        self.path=path
        self.file = open('train_list.txt','w')
         #the file can be given by file=open('train_list.txt','w'), better the folder storing this file should not be the one of self.path 
    def file_list(self):
        for parent, dirnames, filenames in os.walk(self.path):
            for filename in filenames:
                print(filename)
                self.file.write(filename+ '\n')
            self.file.flush()
            
    def file_list_creator(self):
        with open('train_list.txt') as f:  #'train_list.csv' is now in the current work directory
            self.list=[line.rstrip() for line in f]
        print(self.list)
    def metagenome_merger(self):
           
        #instantiation of the class of fasta_reader for using its method
        outfile = pd.concat(Fasta_Reader(self.path+'/'+i).fasta_reader() for i in self.list) 
        #by default it is a vertial appending (axis=0)
        #to make the file callable, the address needs to be precisely and properly expressed
        print(outfile.head())
        outfile['Accession'] = '>' + outfile['Accession']
        outfile.to_csv(self.path+'/'+'combined.fa', sep='\n', index=None, header=None)  
        #if you want to save the combined file to a specific path, use the string addition operation



class Fasta_Stat():
    def __init__(self, path):  
        #seq here is the file name of the outfile, in the format of tuple
        self.path = path
        self.seq = (path+ '/'+'combined.fa')
    def seq_parse(self):
        record = list(SeqIO.parse(self.seq, "fasta"))
        print (len(record))
    def fasta_to_DataFrame(self):
        #parse the fasta dataset and save the features to lists
        meta=[]
        sequence=[]
        length=[]
        CHfrq=[]  #the frequency of residues of Cys and His
        for seq_record in SeqIO.parse(self.seq, "fasta"):
            meta.append(str(seq_record.id))
            sequence.append(str(seq_record.seq))
            length.append(len(seq_record))
            CHfrq.append(int(str(seq_record.seq).count('C'))+int(str(seq_record.seq).count('H')))
        print(sequence)
        #create a DataFrame with the results in lists
        df= pd.DataFrame(data ={'Meta':meta,'Sequence':sequence,'length':length,'CHfrq':CHfrq})
        print(df)
        df.to_csv("fasta_stas.csv",index=None)
        #a csv file naming 'fasta_stas.csv' will appear in in the current work directory.
        print(df.head())
        print(df.describe())
        #remove the sequences with a length>MaxLen aa so that the subsequent searching can be minimized.
        #<MaxLen or >MinLen, change the expression
        df_filtered=df[df['length']<MaxLen]
        df_filtered.to_csv("predictionset.csv",index=None)  
        #a csv file naming "predictionset.csv" will appear in the current work directory. 
        df_filtered.head()

        






