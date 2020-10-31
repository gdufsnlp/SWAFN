# SWAFN
Code for Paper "SWAFN: Sentimental Words Aware Fusion Network for Multimodal Sentiment Analysis", COLING2020


### Requires:  
python > 3.6,  
pytorch == 1.4.0,  
sklearn,  
h5py  

### Datasets:
The MOSI dataset, MOSEI dataset and YouTube dataset can be downlowded at http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/cmu-mosi/seq_length_20/, http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/cmu-mosei/seq_length_20/data/ and https://github.com/pliang279/MFN/tree/master/new_data/youtube respectively. Put the downloaded datasets to the corresponding folder.
### Run the code:
To train and test on CMU-MOSI dataset, run:     
`python MOSI/mosi_model.py`
    
To train and test on CMU-MOSEI dataset, run:   
`python MOSEI/mosei_model.py`
    
To train and test on YouTube dataset, run:     
`python YouTube/youtube_model.py`
