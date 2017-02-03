import hw_utils
import numpy as np
import timeit

X_tr,y_tr,X_te,y_te= ([] for i in range(4))

def load_and_normalize_Data():
    global X_tr,y_tr,X_te,y_te
    X_tr,y_tr,X_te,y_te = hw_utils.loaddata("MiniBooNE_PID.txt")
    X_tr, X_te = hw_utils.normalize(X_tr, X_te)
    


def run_experiments():
    global X_tr,y_tr,X_te,y_te
    print type(X_tr)
   
    start_time = timeit.default_timer() 
    hw_utils.testmodels(np.asarray(X_tr), np.asarray(y_tr), np.asarray(X_te), np.asarray(y_te), [[50,2],[50,50,2],[50,50,50,2],[50,50,50,50,2]], actfn='linear', last_act='softmax', reg_coeffs=[0.0], 
                    num_epoch=30, batch_size=1000, sgd_lr=0.001, sgd_decays=[0.0], sgd_moms=[0.0], 
                        sgd_Nesterov=False, EStop=False, verbose=0)
    elapsed = timeit.default_timer() - start_time
    print 'Time taken to train:', elapsed
      
    print "\nExperiment part d(b) with linear activation and different architectures"
      
    start_time = timeit.default_timer()
    hw_utils.testmodels(np.asarray(X_tr), np.asarray(y_tr), np.asarray(X_te), np.asarray(y_te), [[50,50,2],[50,500,2],[50,500,300,2],[50,800,500,300,2],[50,800,800,500,300,2]], actfn='linear', last_act='softmax', reg_coeffs=[0.0], 
                    num_epoch=30, batch_size=1000, sgd_lr=0.001, sgd_decays=[0.0], sgd_moms=[0.0], 
                        sgd_Nesterov=False, EStop=False, verbose=0)
    elapsed = timeit.default_timer() - start_time
    print 'Time taken to train:', elapsed
      
      
    start_time = timeit.default_timer() 
    print "\nExperiment part e with sigmoid activation (e) part"
    hw_utils.testmodels(np.asarray(X_tr), np.asarray(y_tr), np.asarray(X_te), np.asarray(y_te), [[50,50,2],[50,500,2],[50,500,300,2],[50,800,500,300,2],[50,800,800,500,300,2]], actfn='sigmoid', last_act='softmax', reg_coeffs=[0.0], 
                    num_epoch=30, batch_size=1000, sgd_lr=0.001, sgd_decays=[0.0], sgd_moms=[0.0], 
                        sgd_Nesterov=False, EStop=False, verbose=0)
    elapsed = timeit.default_timer() - start_time
    print 'Time taken to train:', elapsed
      
    start_time = timeit.default_timer() 
    print "\nExperiment part e with Relu activation (f) part"
    hw_utils.testmodels(np.asarray(X_tr), np.asarray(y_tr), np.asarray(X_te), np.asarray(y_te), [[50,50,2],[50,500,2],[50,500,300,2],[50,800,500,300,2],[50,800,800,500,300,2]], actfn='relu', last_act='softmax', reg_coeffs=[0.0], 
                    num_epoch=30, batch_size=1000, sgd_lr=5e-4, sgd_decays=[0.0], sgd_moms=[0.0], 
                        sgd_Nesterov=False, EStop=False, verbose=0)
      
    elapsed = timeit.default_timer() - start_time
    print 'Time taken to train:', elapsed
      
     
    start_time = timeit.default_timer()  
    print "\nExperiment part e with L2 Regularization activation (g) part"
    best_config_g = hw_utils.testmodels(np.asarray(X_tr), np.asarray(y_tr), np.asarray(X_te), np.asarray(y_te), [[50,800,500,300,2]], actfn='relu', last_act='softmax', reg_coeffs=[1e-7, 5e-7, 1e-6, 5e-6, 1e-5], 
                    num_epoch=30, batch_size=1000, sgd_lr=5e-4, sgd_decays=[0.0], sgd_moms=[0.0], 
                        sgd_Nesterov=False, EStop=False, verbose=0)
    elapsed = timeit.default_timer() - start_time
    print 'Time taken to train:', elapsed
     
    start_time = timeit.default_timer() 
    print "\nExperiment part e with Early Stopping and L2 Regularization activation (h) part"
    best_config_h = hw_utils.testmodels(np.asarray(X_tr), np.asarray(y_tr), np.asarray(X_te), np.asarray(y_te), [[50,800,500,300,2]], actfn='relu', last_act='softmax', reg_coeffs=[1e-7, 5e-7, 1e-6, 5e-6, 1e-5], 
                    num_epoch=30, batch_size=1000, sgd_lr=5e-4, sgd_decays=[0.0], sgd_moms=[0.0], 
                        sgd_Nesterov=False, EStop=True, verbose=0)
      
    elapsed = timeit.default_timer() - start_time
    print 'Time taken to train:', elapsed
     
    start_time = timeit.default_timer() 
    print "\nExperiment part SGD with weight decay (i) part"
    best_config_1 = hw_utils.testmodels(np.asarray(X_tr), np.asarray(y_tr), np.asarray(X_te), np.asarray(y_te), [[50,800,500,300,2]], actfn='relu', last_act='softmax', reg_coeffs=[5e-7], 
                    num_epoch=100, batch_size=1000, sgd_lr=1e-5, sgd_decays=[1e-5, 5e-5, 1e-4, 3e-4, 7e-4, 1e-3], sgd_moms=[0.0], 
                        sgd_Nesterov=False, EStop=False, verbose=0)
    elapsed = timeit.default_timer() - start_time
    print 'Time taken to train:', elapsed
      
    start_time = timeit.default_timer() 
    print "\nExperiment part Momentum (j) part"
    print 'moment', best_config_1[2]
    best_config_2 = hw_utils.testmodels(np.asarray(X_tr), np.asarray(y_tr), np.asarray(X_te), np.asarray(y_te), [[50,800,500,300,2]], actfn='relu', last_act='softmax', reg_coeffs=[0.0], 
                    num_epoch=50, batch_size=1000, sgd_lr=1e-5, sgd_decays=[float(best_config_1[2])], sgd_moms=[0.99, 0.98, 0.95, 0.9, 0.85], 
                        sgd_Nesterov=True, EStop=False, verbose=0)
    elapsed = timeit.default_timer() - start_time
    print 'Time taken to train:', elapsed
     
    start_time = timeit.default_timer()
    print "\nExperiment part Momentum (k) part,TODOOOO:: decay best value found in the previous part, moment????"
#     print 'reg', float(best_config_g[1])
#     print  'sgd_moms', float(best_config_2[3])
#     print 'sgd_decays', float(best_config_1[2])
    best_config_3 = hw_utils.testmodels(np.asarray(X_tr), np.asarray(y_tr), np.asarray(X_te), np.asarray(y_te), [[50,800,500,300,2]], actfn='relu', last_act='softmax', reg_coeffs=[float(best_config_g[1])], 
                    num_epoch=100, batch_size=1000, sgd_lr=1e-5, sgd_decays=[float(best_config_1[2])], sgd_moms=[float(best_config_2[3])], 
                        sgd_Nesterov=True, EStop=True, verbose=0)
    elapsed = timeit.default_timer() - start_time
    print 'Time taken to train:', elapsed
     
     
    start_time = timeit.default_timer() 
    print "\nGrid search with cross-validation"
    best_config_4 = hw_utils.testmodels(np.asarray(X_tr), np.asarray(y_tr), np.asarray(X_te), np.asarray(y_te), [[50,50,2],[50,500,2],[50,500,300,2],[50,800,500,300,2],[50,800,800,500,300,2]], actfn='relu', last_act='softmax', reg_coeffs=[1e-7, 5e-7, 1e-6, 5e-6, 1e-5], 
                    num_epoch=100, batch_size=1000, sgd_lr=1e-5, sgd_decays=[1e-5, 5e-5, 1e-4], sgd_moms=[0.99], 
                        sgd_Nesterov=True, EStop=True, verbose=0)
    elapsed = timeit.default_timer() - start_time
    print 'Time taken to train:', elapsed
     
     
     