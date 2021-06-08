class physio_net(object):
    def __init__(self,path):
        self.subject = os.listdir(path)
        self.path = path
        #self.subject = self.subject[0:10]
        self.Fs = 160.0
        self.t_size = 640
        self.left_vs_right = ['03', '04', '07', '08' , '11', '12']
        self.hand_vs_feet  = ['05', '06', '09', '10', '13', '14']
    
    def convert(self,output,test_valid_size = 0.35,valid_test_split = False,batch_size = 10):
        for part in split_seq(self.subject, batch_size): 
        #this loop break downs the dataset so it can be loaded into RAM
            for subject in part :
                #iterate for each subject
                LR_anotation = []
                FH_anotation = []
                LR_data      = []
                FH_data      = []
            
                #left_vs_right data
                for trial in self.left_vs_right:
                    edf_data = mne.io.read_raw_edf(self.path+"/"+str(subject)+"/"+str(subject)+"R"+str(trial)+".edf",verbose=False)
                    anotation = pd.DataFrame({
                    'onsest':edf_data.annotations.onset,
                    'duration':edf_data.annotations.duration,
                    'type':edf_data.annotations.description}
                    )
                    LR_data.append(edf_data.get_data()[:,:19920]) # [:,:19920] because the last 80 sample is 0 and must be deleted
                    LR_anotation.append(anotation)

                L_seg_data = np.zeros((0,64,self.t_size))
                R_seg_data = np.zeros((0,64,self.t_size))
                Labels = []
            
                for i_session in range(len(LR_anotation)):
                    for i_trial in range(LR_anotation[i_session].shape[0]):
                        if(LR_anotation[i_session].iloc[i_trial][2] == 'T0'):
                            continue
                        duration = self.Fs * (LR_anotation[i_session].iloc[i_trial][1])
                        onset    = self.Fs * (LR_anotation[i_session].iloc[i_trial][0])
                        selected = LR_data[i_session][:,int(onset):int(onset+self.t_size)]
                        try:
                            if(LR_anotation[i_session].iloc[i_trial][2] == 'T1'):#motion of left hand 
                                L_seg_data = np.vstack((L_seg_data,selected.reshape((1,selected.shape[0],selected.shape[1]))))
                            if(LR_anotation[i_session].iloc[i_trial][2] == 'T2'):#motion of right hand
                                R_seg_data = np.vstack((R_seg_data,selected.reshape((1,selected.shape[0],selected.shape[1]))))
                        except:
                            pass

                if(not valid_test_split):
                    try:
                        os.makedirs(output+'/train/Right')
                        os.makedirs(output+'/train/Left')
                    except OSError:
                        pass
                    #train
                    for R in range(R_seg_data.shape[0]):
                        np.save((output+'/train/Right/'+str(subject)+'_'+ str(R)),R_seg_data[R,:,:])
                    for L in range(L_seg_data.shape[0]):
                        np.save((output+'/train/Left/'+str(subject)+'_'+ str(L)),L_seg_data[L,:,:])

                else :
                    try:
                        os.makedirs(output+'/train/Right')
                        os.makedirs(output+'/train/Left')
                    except OSError:
                        pass
                    try:
                        os.makedirs(output+'/validation/Right')
                        os.makedirs(output+'/validation/Left')
                    except OSError:
                        pass
                    try:
                        os.makedirs(output+'/test/Right')
                        os.makedirs(output+'/test/Left')
                    except OSError:
                        pass
                    
                #split into train test validation
                R_train , R_v_t = train_test_split(R_seg_data,test_size = test_valid_size)
                R_valid , R_test = train_test_split(R_v_t,test_size = 0.5)
                
                L_train , L_v_t = train_test_split(L_seg_data,test_size = test_valid_size)
                L_valid , L_test = train_test_split(L_v_t,test_size = 0.5)
                
                #train
                for R in range(R_train.shape[0]):
                    np.save((output+'/train/Right/'+str(subject)+'_'+ str(R)),R_train[R,:,:])
                for L in range(L_train.shape[0]):
                    np.save((output+'/train/Left/'+str(subject)+'_'+ str(L)),L_train[L,:,:])
                #test
                for R in range(R_test.shape[0]):
                    np.save((output+'/test/Right/'+str(subject)+'_'+ str(R)),R_test[R,:,:])
                for L in range(L_test.shape[0]):
                    np.save((output+'/test/Left/'+str(subject)+'_'+ str(L)),L_test[L,:,:])
                #validation
                for R in range(R_valid.shape[0]):
                    np.save((output+'/validation/Right/'+str(subject)+'_'+ str(R)),R_valid[R,:,:])
                for L in range(L_valid.shape[0]):
                    np.save((output+'/validation/Left/'+str(subject)+'_'+ str(L)),L_valid[L,:,:])
                
            
                #hand_vs_feet data
                for trial in self.hand_vs_feet:
                    edf_data = mne.io.read_raw_edf(self.path+"/"+str(subject)+"/"+str(subject)+"R"+str(trial)+".edf",verbose=False)
                    anotation = pd.DataFrame({
                    'onsest':edf_data.annotations.onset,
                    'duration':edf_data.annotations.duration,
                    'type':edf_data.annotations.description}
                    )
                    FH_data.append(edf_data.get_data()[:,:19920])
                    FH_anotation.append(anotation)
            
                F_seg_data = np.zeros((0,64,self.t_size))
                H_seg_data = np.zeros((0,64,self.t_size))
                Labels = []
                for i_session in range(len(FH_anotation)):
                    for i_trial in range(FH_anotation[i_session].shape[0]):
                        if(FH_anotation[i_session].iloc[i_trial][2] == 'T0'):
                            continue
                        duration = self.Fs * (FH_anotation[i_session].iloc[i_trial][1])
                        onset    = self.Fs * (FH_anotation[i_session].iloc[i_trial][0])
                        selected = FH_data[i_session][:,int(onset):int(onset+self.t_size)]
                        try:
                            if(FH_anotation[i_session].iloc[i_trial][2] == 'T1'):#motion of Hand hand 
                                H_seg_data = np.vstack((H_seg_data,selected.reshape((1,selected.shape[0],selected.shape[1]))))
                            if(FH_anotation[i_session].iloc[i_trial][2] == 'T2'):#motion of Feet hand
                                F_seg_data = np.vstack((F_seg_data,selected.reshape((1,selected.shape[0],selected.shape[1]))))
                        except:
                            pass
            
            
                if(not valid_test_split):
                    try:
                        os.makedirs(output+'/train/Feet')
                        os.makedirs(output+'/train/Hands')
                    except:
                        pass                
                    #train
                    for F in range(F_seg_data.shape[0]):
                        np.save((output+'/train/Feet/'+str(subject)+'_'+ str(R)),F_seg_data[F,:,:])
                    for H in range(H_seg_data.shape[0]):
                        np.save((output+'/train/Hands/'+str(subject)+'_'+ str(L)),H_seg_data[H,:,:])

                else :
                    try:
                        os.makedirs(output+'/train/Feet')
                        os.makedirs(output+'/train/Hands')
                    except OSError:
                        pass
                    try:
                        os.makedirs(output+'/validation/Feet')
                        os.makedirs(output+'/validation/Hands')
                    except OSError:
                        pass
                    try:
                        os.makedirs(output+'/test/Feet')
                        os.makedirs(output+'/test/Hands')
                    except OSError:
                        pass

                    #split into train test validation
                    F_train , F_v_t = train_test_split(F_seg_data,test_size = 0.35)
                    F_valid , F_test = train_test_split(R_v_t,test_size = 0.5)
                    
                    H_train , H_v_t = train_test_split(H_seg_data,test_size = 0.35)
                    H_valid , H_test = train_test_split(H_v_t,test_size = 0.5)
                    
                    #train
                    for F in range(F_train.shape[0]):
                        np.save((output+'/train/Feet/' +str(subject)+'_'+str(F)),F_train[F,:,:])
                    for H in range(H_train.shape[0]):
                        np.save((output+'/train/Hands/'+str(subject)+'_'+str(H)),H_train[H,:,:])
                    #validation
                    for F in range(F_valid.shape[0]):
                        np.save((output+'/test/Feet/' +str(subject)+'_'+str(F)) ,F_valid[F,:,:])
                    for H in range(H_valid.shape[0]):
                        np.save((output+'/test/Hands/'+str(subject)+'_'+str(H)),H_valid[H,:,:])
                    #test
                    for F in range(F_test.shape[0]):
                        np.save((output+'/validation/Feet/' +str(subject)+'_'+str(F)) ,F_test[F,:,:])
                    for H in range(H_test.shape[0]):
                        np.save((output+'/validation/Hands/'+str(subject)+'_'+str(H)),H_test[H,:,:])   

                del H_seg_data,F_seg_data,R_seg_data,L_seg_data, FH_anotation,F_train , F_v_t,F_valid , F_test,H_train , H_v_t,H_valid , H_test,R_train , R_v_t,R_valid , R_test,L_valid , L_test , L_train , L_v_t


if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser(description='Physionet Data Creator')
    parser.add_argument('--path', 
                            action="store",
                            dest = "path", 
                            default="./physionet",
                            help="path to the dataset folder ")
    parser.add_argument('--output', 
                            action="store",
                            dest = "output", 
                            default="./output_data",
                            help="path to the converted data ")
    parser.add_argument('--valid_test_split', 
                            action="store_true",
                            dest = "valid_test_split", 
                            help="wheter to split train data into validation and test")

    args = parser.parse_args()
    
    data_holder = physio_net(path= args.path)
    data.convert(args.output, valid_test_split=args.valid_test_split)

    sys.exit()