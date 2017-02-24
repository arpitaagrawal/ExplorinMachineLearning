import Bias_Variance_tradeOff
import SVM_Kernel

## Part 1(a) : Plot MSE for 10 samples each in 100 datasets. Print Bias^2, and Variace as well
Bias_Variance_tradeOff.data_analysis_bias_variance([0,1,2,3,4,5], [0], 100, 10, True )

## Part 1(b) : Plot MSE for 100 samples each in 100 datasets. Print Bias^2, and Variace as well
Bias_Variance_tradeOff.data_analysis_bias_variance([0,1,2,3,4,5], [0], 100, 100, True )

## Part 1(d) :  Bias^2, and Variance analysis for 100 samples each in 100 datasets.
print 'Bias-Variance Alamysis with regularization for diff lamndas'
Bias_Variance_tradeOff.data_analysis_bias_variance([2], [ 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0], 100, 100, False )

SVM_Kernel.load_dataset()    
SVM_Kernel.data_processing()
SVM_Kernel.train_svm_model()
SVM_Kernel.train_kernelised_svm_model()

#SVM_Kernel.load_dataset()