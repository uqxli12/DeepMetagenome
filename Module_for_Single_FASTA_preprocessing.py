#coding: utf-8
import tensorflow as tf
import pandas as pd
from Bio import SeqIO
import pandas as pd
import os
import Global_var_settings

"""
This modual depends on biopython and tensorflow. 
conda install biopython
Define MaxLen or MaxLen first
"""

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

#define MaxLen or MinLen First 
MaxLen = 200
class Fasta_Stat_for_singleFASTA():
    def __init__(self, file):  
        #seq here is the file name of the outfile, in the format of tuple
       
        self.seq = (file)
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
        df= pd.DataFrame(data ={'Meta':meta,'SequenceID':sequence,'length':length,'CHfrq':CHfrq})
        print(df)
        df.to_csv("fasta_stas.csv",index=None)
        #a csv file naming 'fasta_stas.csv' will appear in in the current work directory.
        print(df.head())
        print(df.describe())
        #remove the sequences with a length>MaxLen aa so that the subsequent searching can be minimized
        df_filtered=df[df['length']< ]
        df_filtered.to_csv("predictionset.csv",index=None)  
        #a csv file naming "predictionset.csv" will appear in the current work directory. 
        df_filtered.head()
