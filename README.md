# SWAFN
Code for Paper "SWAFN: Sentimental Words Aware Fusion Network for Multimodal Sentiment Analysis", COLING2020


### requires:  
python > 3.6,  
pytorch == 1.4.0,  
sklearn,  
h5py  

The YouTube dataset is already in the YouTube folder. The MOSI dataset and MOSEI dataset can be downlowded at http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/cmu-mosi/seq_length_20/ and http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/cmu-mosei/seq_length_20/data/ respectively. Put the dataset to the corresponding folder.(same as that of YouTube)


To train and test on CMU-MOSI dataset, run:     
`python MOSI/mosi_model.py`
    
To train and test on CMU-MOSEI dataset, run:   
`python MOSEI/mosei_model.py`
    
To train and test on YouTube dataset, run:     
`python YouTube/youtube_model.py`
